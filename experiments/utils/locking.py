"""One measurement worker per visible physical device on a shared host."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import tempfile


@contextmanager
def device_lease(target):
    import fcntl

    kind = target["kind"]
    ordinal = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if kind == "cuda":
        try:
            identity = subprocess.check_output(
                ["nvidia-smi", "-i", ordinal, "--query-gpu=uuid", "--format=csv,noheader"], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            identity = ordinal
    else:
        identity = os.environ.get(
            "ROCR_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ordinal))
        )
    key = hashlib.sha256(f"{socket.gethostname()}:{kind}:{identity}".encode()).hexdigest()
    path = Path(tempfile.gettempdir()) / ("tiletune-device-" + key + ".lock")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another TileTune worker owns this device; resume after it finishes") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
