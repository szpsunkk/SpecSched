#!/usr/bin/env python3
"""Monitor a background experiment and email on failure/interruption.

The monitor watches the wrapper PID, exit-code file, and log file. It sends
one alert when:
  - the exit file appears with a non-zero code;
  - the wrapper PID disappears before an exit file is written;
  - the log contains a fatal error pattern.

Email transport:
  1. SMTP if SMTP_HOST is set. Optional env vars: SMTP_PORT, SMTP_USER,
     SMTP_PASSWORD, SMTP_SSL, SMTP_FROM.
  2. /usr/sbin/sendmail if available.
  3. mail/mailx command if available.
"""
from __future__ import annotations

import argparse
import os
import shutil
import smtplib
import socket
import subprocess
import sys
import time
from email.message import EmailMessage
from pathlib import Path


FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "RuntimeError: CUDA",
    "Killed",
    "Permission denied",
    "No route to host",
    "Connection refused",
    "Connection reset",
    "SSHException",
    "TimeoutError",
    "edge worker exited",
    "edge worker closed stdout",
)


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(data) > limit:
        return data[-limit:]
    return data


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def send_email(to_addr: str, subject: str, body: str) -> None:
    from_addr = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or f"specsched@{socket.gethostname()}"
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    smtp_host = os.environ.get("SMTP_HOST")
    if smtp_host:
        port = int(os.environ.get("SMTP_PORT", "465" if os.environ.get("SMTP_SSL", "1") != "0" else "25"))
        smtp_ssl = os.environ.get("SMTP_SSL", "1") != "0"
        cls = smtplib.SMTP_SSL if smtp_ssl else smtplib.SMTP
        with cls(smtp_host, port, timeout=30) as smtp:
            if not smtp_ssl:
                smtp.ehlo()
                if os.environ.get("SMTP_STARTTLS", "0") == "1":
                    smtp.starttls()
            user = os.environ.get("SMTP_USER")
            password = os.environ.get("SMTP_PASSWORD")
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return

    sendmail = shutil.which("sendmail") or "/usr/sbin/sendmail"
    if Path(sendmail).exists():
        subprocess.run([sendmail, "-t"], input=msg.as_bytes(), check=True)
        return

    mail = shutil.which("mail") or shutil.which("mailx")
    if mail:
        subprocess.run([mail, "-s", subject, to_addr], input=body.encode(), check=True)
        return

    raise RuntimeError("no email transport available: set SMTP_HOST or install sendmail/mailx")


def find_fatal(log_tail: str) -> str | None:
    for pattern in FATAL_PATTERNS:
        if pattern in log_tail:
            return pattern
    return None


def notify_once(state_file: Path, to_addr: str, subject: str, body: str) -> None:
    if state_file.exists():
        return
    try:
        send_email(to_addr, subject, body)
        state_file.write_text(f"sent {time.ctime()}\n{subject}\n", encoding="utf-8")
    except Exception as exc:
        state_file.write_text(f"send_failed {time.ctime()}\n{type(exc).__name__}: {exc}\n{subject}\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--pidfile", required=True)
    parser.add_argument("--exitfile", required=True)
    parser.add_argument("--logfile", required=True)
    parser.add_argument("--to", required=True)
    parser.add_argument("--poll-s", type=int, default=60)
    parser.add_argument("--state-file", default="")
    parser.add_argument("--notify-success", action="store_true")
    args = parser.parse_args()

    pidfile = Path(args.pidfile)
    exitfile = Path(args.exitfile)
    logfile = Path(args.logfile)
    state_file = Path(args.state_file) if args.state_file else logfile.with_suffix(logfile.suffix + ".monitor_notified")
    hostname = socket.gethostname()

    while True:
        now = time.ctime()
        exit_text = read_text(exitfile).strip()
        log_tail = read_text(logfile, limit=12000)

        if exit_text:
            try:
                code = int(exit_text.split()[0])
            except Exception:
                code = -999
            if code != 0:
                subject = f"[SpecSched] {args.name} failed on {hostname} (exit {code})"
                body = (
                    f"Experiment: {args.name}\n"
                    f"Host: {hostname}\n"
                    f"Time: {now}\n"
                    f"Exit code: {code}\n"
                    f"PID file: {pidfile}\n"
                    f"Exit file: {exitfile}\n"
                    f"Log file: {logfile}\n\n"
                    f"Last log lines:\n{log_tail}\n"
                )
                notify_once(state_file, args.to, subject, body)
            elif args.notify_success:
                subject = f"[SpecSched] {args.name} completed on {hostname}"
                body = (
                    f"Experiment: {args.name}\n"
                    f"Host: {hostname}\n"
                    f"Time: {now}\n"
                    f"Exit code: {code}\n"
                    f"Log file: {logfile}\n"
                )
                notify_once(state_file, args.to, subject, body)
            return code

        fatal = find_fatal(log_tail)
        if fatal:
            subject = f"[SpecSched] {args.name} log error on {hostname}: {fatal}"
            body = (
                f"Experiment: {args.name}\n"
                f"Host: {hostname}\n"
                f"Time: {now}\n"
                f"Detected fatal log pattern: {fatal}\n"
                f"PID file: {pidfile}\n"
                f"Log file: {logfile}\n\n"
                f"Last log lines:\n{log_tail}\n"
            )
            notify_once(state_file, args.to, subject, body)
            return 2

        pid_text = read_text(pidfile).strip()
        if pid_text:
            try:
                pid = int(pid_text.split()[0])
            except Exception:
                pid = -1
            if pid > 0 and not pid_alive(pid):
                subject = f"[SpecSched] {args.name} interrupted on {hostname}"
                body = (
                    f"Experiment: {args.name}\n"
                    f"Host: {hostname}\n"
                    f"Time: {now}\n"
                    f"PID {pid} is no longer alive, and no exit file exists.\n"
                    f"PID file: {pidfile}\n"
                    f"Exit file: {exitfile}\n"
                    f"Log file: {logfile}\n\n"
                    f"Last log lines:\n{log_tail}\n"
                )
                notify_once(state_file, args.to, subject, body)
                return 3

        time.sleep(max(5, args.poll_s))


if __name__ == "__main__":
    raise SystemExit(main())
