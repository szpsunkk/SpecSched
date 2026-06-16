"""SSH-backed stateful draft client for scheduler experiments."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from socket import timeout as SocketTimeout
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = ROOT / "experiments" / "common"
SCHEDULER_DIR = ROOT / "experiments" / "scheduler"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from ssh_jetson import open_ssh, put_file  # noqa: E402


class EdgeDraftClient:
    remote_draft = True

    def __init__(
        self,
        worker: str,
        remote_model: str,
        host: str,
        user: str,
        password: str,
        request_timeout_s: Optional[float] = None,
    ):
        if not password:
            raise RuntimeError("edge password is empty; set JETSON_PASSWORD or pass --edge-password")
        self.request_id = 0
        self.host = host
        self.request_timeout_s = request_timeout_s
        self.client = open_ssh(host=host, user=user, password=password, timeout=30)
        self.remote_dir = f"/home/{user}/specsched"
        self.remote_worker = f"{self.remote_dir}/edge_stateful_worker.py"
        self.client.exec_command(f"mkdir -p {self.remote_dir}")
        put_file(self.client, worker, self.remote_worker)
        cmd = (
            f"cd {self.remote_dir} && "
            f"USE_TF=0 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 "
            f"python3 {self.remote_worker} --model {remote_model}"
        )
        print(f"[edge-scheduler] starting worker on {host}", flush=True)
        self.stdin, self.stdout, self.stderr = self.client.exec_command(cmd, get_pty=False, timeout=None)
        self.stdin.channel.settimeout(request_timeout_s)
        self.stdout.channel.settimeout(request_timeout_s)
        self._stdout_buffer = b""
        self._wait_until_ready()

    def _read_stderr_available(self) -> str:
        chunks: List[str] = []
        while self.stderr.channel.recv_stderr_ready():
            chunks.append(self.stderr.channel.recv_stderr(4096).decode("utf-8", "replace"))
        return "".join(chunks)

    def _wait_until_ready(self) -> None:
        deadline = time.time() + 180
        buffer = ""
        while time.time() < deadline:
            if self.stderr.channel.recv_stderr_ready():
                text = self.stderr.channel.recv_stderr(4096).decode("utf-8", "replace")
                buffer += text
                print(text, end="", flush=True)
                if "[edge] ready" in buffer:
                    return
            if self.stdout.channel.exit_status_ready():
                raise RuntimeError("edge worker exited before ready: " + buffer)
            time.sleep(0.2)
        raise TimeoutError("edge worker did not become ready within 180s")

    def _request(self, req: dict) -> dict:
        self.request_id += 1
        req.setdefault("id", self.request_id)
        t0 = time.perf_counter()
        self.stdin.write(json.dumps(req) + "\n")
        self.stdin.flush()
        line = self._read_stdout_line()
        if not line:
            raise RuntimeError("edge worker closed stdout: " + self._read_stderr_available())
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"edge worker error on {self.host}: {resp}")
        resp["client_roundtrip_ms"] = roundtrip_ms
        resp["client_overhead_ms"] = max(0.0, roundtrip_ms - float(resp.get("edge_elapsed_ms", 0.0)))
        return resp

    def _read_stdout_line(self) -> str:
        """Read one JSONL response with an explicit wall-clock deadline.

        Paramiko's file-like ``stdout.readline()`` can block indefinitely on
        some SSH transports even after ``Channel.settimeout``. Use the channel
        readiness API directly so edge experiments fail and resume instead of
        hanging a full paper sweep.
        """
        deadline = None
        if self.request_timeout_s is not None and self.request_timeout_s > 0:
            deadline = time.time() + float(self.request_timeout_s)
        channel = self.stdout.channel
        while True:
            if b"\n" in self._stdout_buffer:
                line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
                return line.decode("utf-8", "replace") + "\n"
            if channel.recv_ready():
                chunk = channel.recv(4096)
                if not chunk:
                    break
                self._stdout_buffer += chunk
                continue
            if channel.exit_status_ready():
                break
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    f"edge worker response timed out after {self.request_timeout_s}s on {self.host}: "
                    + self._read_stderr_available()
                )
            time.sleep(0.01)
        if self._stdout_buffer:
            line = self._stdout_buffer
            self._stdout_buffer = b""
            return line.decode("utf-8", "replace")
        raise RuntimeError("edge worker closed stdout: " + self._read_stderr_available())

    def prefill(self, state_id: int, input_ids) -> tuple[object, float]:
        resp = self._request({"cmd": "prefill", "state_id": str(state_id), "input_ids": list(input_ids)})
        self.last_roundtrip_ms = float(resp.get("client_roundtrip_ms", 0.0))
        self.last_overhead_ms = float(resp.get("client_overhead_ms", 0.0))
        return True, float(resp.get("edge_elapsed_ms", 0.0))

    def draft(self, state_id: int, last_token: int, k: int) -> tuple[List[int], float]:
        resp = self._request({"cmd": "draft", "state_id": str(state_id), "last_token": int(last_token), "k": int(k)})
        self.last_roundtrip_ms = float(resp.get("client_roundtrip_ms", 0.0))
        self.last_overhead_ms = float(resp.get("client_overhead_ms", 0.0))
        return [int(x) for x in resp["tokens"]], float(resp.get("edge_elapsed_ms", 0.0))

    def crop(self, state_id: int, seq_len: int) -> None:
        self._request({"cmd": "crop", "state_id": str(state_id), "seq_len": int(seq_len)})

    def release(self, state_id: int) -> None:
        try:
            self._request({"cmd": "release", "state_id": str(state_id)})
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
            self.stdin.flush()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass


class MultiEdgeDraftClient:
    remote_draft = True

    def __init__(
        self,
        remote_model: str,
        hosts: List[str],
        user: str,
        password: str,
        request_timeout_s: Optional[float] = None,
    ):
        if not hosts:
            raise RuntimeError("at least one edge host is required")
        worker = str(SCHEDULER_DIR / "edge_stateful_worker.py")
        self.clients = [
            EdgeDraftClient(
                worker=worker,
                remote_model=remote_model,
                host=host,
                user=user,
                password=password,
                request_timeout_s=request_timeout_s,
            )
            for host in hosts
        ]

    def _client_for_state(self, state_id: int) -> EdgeDraftClient:
        return self.clients[int(state_id) % len(self.clients)]

    def prefill(self, state_id: int, input_ids) -> tuple[object, float]:
        client = self._client_for_state(state_id)
        out = client.prefill(state_id, input_ids)
        self.last_roundtrip_ms = getattr(client, "last_roundtrip_ms", 0.0)
        self.last_overhead_ms = getattr(client, "last_overhead_ms", 0.0)
        return out

    def draft(self, state_id: int, last_token: int, k: int) -> tuple[List[int], float]:
        client = self._client_for_state(state_id)
        out = client.draft(state_id, last_token, k)
        self.last_roundtrip_ms = getattr(client, "last_roundtrip_ms", 0.0)
        self.last_overhead_ms = getattr(client, "last_overhead_ms", 0.0)
        return out

    def crop(self, state_id: int, seq_len: int) -> None:
        self._client_for_state(state_id).crop(state_id, seq_len)

    def release(self, state_id: int) -> None:
        self._client_for_state(state_id).release(state_id)

    def close(self) -> None:
        for client in self.clients:
            client.close()
