"""Linux host/device leases and CPU observations for isolated measurements."""

from contextlib import contextmanager, ExitStack
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import socket
import tempfile
import time


@contextmanager
def measurement_lease(gpus):
    """Serialize experiment launchers, and lease every GPU in the fixed set.

    Workers inherit these descriptors so a killed coordinator cannot release
    its lease while a measurement worker is still alive.
    """
    keys = [f"{socket.gethostname()}:experiment-host"]
    keys.extend(f"{socket.gethostname()}:cuda:{gpu['uuid']}" for gpu in gpus)
    with ExitStack() as stack:
        streams = []
        for key in sorted(keys):
            digest = hashlib.sha256(key.encode()).hexdigest()
            stream = stack.enter_context((Path(tempfile.gettempdir()) / f"tiletune-device-{digest}.lock").open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another experiment owns the host or a requested GPU; no second workload was launched") from None
            streams.append(stream)
        yield tuple(stream.fileno() for stream in streams)
        # Closing releases the lock only after the inheriting worker closes it.


def cpu_ticks(cpu_ids):
    """Return total and non-idle jiffies on the worker's assigned logical CPUs."""
    wanted = {f"cpu{cpu}" for cpu in cpu_ids}
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in wanted:
            values = [int(value) for value in fields[1:9]]
            total = sum(values)  # guest/guest_nice are already included in user/nice.
            result[int(fields[0][3:])] = (total, total - values[3] - values[4], values[4], values[7])
    if set(result) != set(cpu_ids):
        raise RuntimeError("CPU observation is incomplete")
    return result


def owned_cpu_ticks(process_group, proc_root=Path("/proc")):
    """Count one worker tree without scanning every process on the host.

    A process stat includes CPU from children it has already reaped. Walking
    live descendants and summing those four counters therefore retains the old
    accounting contract while making the monitor cost depend on one workload,
    not the number of unrelated host processes.
    """
    if process_group is None:
        return 0
    process_group = int(process_group)
    pending = [process_group]
    visited = set()
    ticks = 0
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        path = proc_root / str(pid)
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) != process_group:
            continue
        # Fields begin with state (field 3); utime/stime/cutime/cstime are 14..17.
        ticks += sum(int(value) for value in fields[11:15])
        try:
            child_files = list((path / "task").glob("*/children"))
        except (FileNotFoundError, ProcessLookupError):
            child_files = []
        for child_file in child_files:
            try:
                pending.extend(int(value) for value in child_file.read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                continue
    return ticks


class CpuMonitor:
    """Estimate external CPU use after subtracting this worker's process group.

    Counts include scheduler/kernel noise, so the policy allows two CPU cores
    and requires two consecutive excessive samples. This is contention evidence,
    not an operating-system exclusive CPU reservation.
    """

    def __init__(self, cpu_ids):
        self.cpu_ids = cpu_ids
        self.previous = None
        self.owned_high_water = 0

    def sample(self, process_group=None):
        current = cpu_ticks(self.cpu_ids)
        owned = max(self.owned_high_water, owned_cpu_ticks(process_group))
        self.owned_high_water = owned
        now = time.monotonic()
        previous, self.previous = self.previous, (now, current, owned)
        if previous is None:
            return dict(ready=False, cpu_ids=self.cpu_ids)
        elapsed = now - previous[0]
        scale = os.sysconf("SC_CLK_TCK") * elapsed
        busy = sum(current[c][1] - previous[1][c][1] for c in self.cpu_ids)
        iowait = sum(current[c][2] - previous[1][c][2] for c in self.cpu_ids)
        steal = sum(current[c][3] - previous[1][c][3] for c in self.cpu_ids)
        return dict(
            ready=elapsed > 0,
            cpu_ids=self.cpu_ids,
            interval_seconds=elapsed,
            busy_cores=busy / scale,
            owned_cores=(owned - previous[2]) / scale,
            external_busy_cores=max(0.0, (busy - owned + previous[2]) / scale),
            iowait_cores=max(0.0, iowait / scale),
            steal_cores=max(0.0, steal / scale),
        )


def select_cpu_pools(workers, count, *, reserve=8):
    """Choose disjoint quiet sibling-complete CPU pools for concurrent workloads."""
    available = sorted(os.sched_getaffinity(0))
    per_pool = workers + reserve
    if count < 1 or len(available) < count * per_pool:
        raise RuntimeError(
            f"need {count * per_pool} visible logical CPUs for {count} workloads, "
            f"each with {workers} workers plus {reserve} benchmark/monitor CPUs"
        )
    before = cpu_ticks(available)
    time.sleep(1)
    after = cpu_ticks(available)
    groups = {}
    for cpu in available:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text().strip(), (topology / "core_id").read_text().strip())
        groups.setdefault(key, []).append(cpu)

    def activity(group):
        return sum(after[c][1] - before[c][1] for c in group) / max(1, sum(after[c][0] - before[c][0] for c in group))

    pools = [[] for _ in range(count)]
    for group in sorted(groups.values(), key=lambda group: (activity(group), group)):
        candidates = [i for i, pool in enumerate(pools) if len(pool) < per_pool]
        if not candidates:
            return [sorted(pool) for pool in pools]
        pools[min(candidates, key=lambda i: (len(pools[i]), i))].extend(group)
    if all(len(pool) >= per_pool for pool in pools):
        return [sorted(pool) for pool in pools]
    raise RuntimeError("insufficient sibling-complete CPU topology")


def select_cpu_ids(workers, *, reserve=8):
    """Choose one quiet sibling-complete CPU pool."""
    return select_cpu_pools(workers, 1, reserve=reserve)[0]


def prepare_worker(cpu_ids, parent_pid):
    """Pin before importing Torch; terminate the owned group if its parent dies."""
    os.sched_setaffinity(0, cpu_ids)

    def terminate_group(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, terminate_group)
    # PR_SET_PDEATHSIG: the child handles SIGTERM by cleaning its process group.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot configure parent-death cleanup")
    if os.getppid() != parent_pid:
        terminate_group(signal.SIGTERM, None)


@contextmanager
def interruptible():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
