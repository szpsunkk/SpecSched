"""SSH/SFTP helpers for Jetson edge workers."""
from __future__ import annotations

import os
import posixpath

import paramiko


HOST = os.environ.get("JETSON_HOST", "192.168.3.108")
USER = os.environ.get("JETSON_USER", "jetson")
PASS = os.environ.get("JETSON_PASSWORD", "")


def open_ssh(host=HOST, user=USER, password=PASS, timeout=20):
    if not password:
        raise RuntimeError("JETSON_PASSWORD is required for password-based SSH")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=user,
        password=password,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run_cmd(client, cmd, get_pty=False, timeout=None):
    _stdin, stdout, stderr = client.exec_command(cmd, get_pty=get_pty, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def put_file(client, local, remote):
    sftp = client.open_sftp()
    rdir = posixpath.dirname(remote)
    try:
        sftp.stat(rdir)
    except IOError:
        cur = ""
        for part in rdir.strip("/").split("/"):
            cur = cur + "/" + part
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)
    sftp.put(local, remote)
    sftp.close()


def get_file(client, remote, local):
    sftp = client.open_sftp()
    sftp.get(remote, local)
    sftp.close()
