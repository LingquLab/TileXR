#!/usr/bin/env python3

import argparse
import concurrent.futures
import dataclasses
import os
import re
import shlex
import statistics
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_HOSTS = [
    "141.61.92.151",
    "141.61.92.147",
    "141.61.92.143",
    "141.61.92.139",
    "141.61.92.111",
    "141.61.92.107",
    "141.61.92.103",
    "141.61.92.99",
]
DEFAULT_CANN_ENV = "/home/c30061605/startup_discom/tilexr/env/cann/cann/set_env.sh"
DEFAULT_DEMO_BIN = "tests/udma/build/tilexr_udma_demo"
DEFAULT_TOOL_ENV = ""
INT32_BYTES = 4


@dataclasses.dataclass(frozen=True)
class RankMetric:
    rank: int
    iters: int
    total_ms: float
    per_iter_us: float
    payload_bytes: float
    bandwidth_gbps: float


@dataclasses.dataclass(frozen=True)
class RankRunResult:
    rank: int
    host: str
    rc: int
    stdout: str
    stderr: str
    elapsed_s: float


SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|bytes?)?\s*$", re.IGNORECASE)
METRIC_RE = re.compile(
    r"\[rank\s+(?P<rank>\d+)\].*?alltoall\s+udma-bigdata\s+"
    r"(?P<iters>\d+)\s+iters.*?total=(?P<total>[0-9.eE+-]+)\s+ms\s+"
    r"perIter=(?P<per>[0-9.eE+-]+)\s+us\s+"
    r"payload=(?P<payload>[0-9.eE+-]+)\s+bytes\s+"
    r"bw=(?P<bw>[0-9.eE+-]+)\s+GB/s"
)


def parse_size_to_bytes(value: str) -> int:
    match = SIZE_RE.match(value)
    if not match:
        raise ValueError(f"invalid size: {value!r}")
    number = float(match.group(1))
    suffix = (match.group(2) or "b").lower()
    scale = {
        "": 1,
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "mib": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
        "gib": 1024 ** 3,
        "t": 1024 ** 4,
        "tb": 1024 ** 4,
        "tib": 1024 ** 4,
    }.get(suffix)
    if scale is None:
        raise ValueError(f"invalid size suffix: {value!r}")
    size = int(number * scale)
    if size <= 0:
        raise ValueError("size must be positive")
    return size


def bytes_to_int32_elements(byte_count: int) -> int:
    if byte_count % INT32_BYTES != 0:
        raise ValueError(f"bytes per peer must be {INT32_BYTES}-byte aligned for int32 payload")
    return byte_count // INT32_BYTES


def select_hosts(hosts: Sequence[str], world_size: int, devices_per_host: int) -> List[str]:
    if world_size <= 0:
        raise ValueError("world size must be positive")
    if devices_per_host <= 0:
        raise ValueError("devices per host must be positive")
    if world_size % devices_per_host != 0:
        raise ValueError(f"world size {world_size} must be a multiple of devices per host {devices_per_host}")
    host_count = world_size // devices_per_host
    if host_count > len(hosts):
        raise ValueError(f"world size {world_size} requires {host_count} hosts, only {len(hosts)} configured")
    return list(hosts[:host_count])


def parse_demo_metric(text: str) -> RankMetric:
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        raise ValueError("no alltoall udma-bigdata metric line found")
    match = matches[-1]
    return RankMetric(
        rank=int(match.group("rank")),
        iters=int(match.group("iters")),
        total_ms=float(match.group("total")),
        per_iter_us=float(match.group("per")),
        payload_bytes=float(match.group("payload")),
        bandwidth_gbps=float(match.group("bw")),
    )


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * fraction)))
    return sorted_values[index]


def summarize_metrics(metrics: Sequence[RankMetric]) -> Dict[str, float]:
    values = sorted(metric.per_iter_us for metric in metrics)
    if not values:
        raise ValueError("no rank metrics to summarize")
    return {
        "count": len(values),
        "min_us": values[0],
        "p50_us": percentile(values, 0.50),
        "mean_us": statistics.mean(values),
        "p90_us": percentile(values, 0.90),
        "p99_us": percentile(values, 0.99),
        "max_us": values[-1],
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def should_skip_archive_path(path: Path) -> bool:
    parts = set(path.parts)
    if ".git" in parts or ".worktrees" in parts:
        return True
    if "__pycache__" in parts:
        return True
    if path.suffix in {".pyc", ".log"}:
        return True
    return any(part in {"build", "install", "build_device", "output", "run"} for part in parts)


def create_source_archive(root: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="tilexr_sync_", suffix=".tar.gz", delete=False)
    handle.close()
    archive = Path(handle.name)
    with tarfile.open(archive, "w:gz") as tar:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if should_skip_archive_path(relative):
                continue
            tar.add(path, arcname=str(relative), recursive=False)
    return archive


def load_paramiko():
    try:
        import paramiko  # type: ignore
    except ImportError as exc:
        raise RuntimeError("paramiko is required; install it or run from an environment that provides it") from exc
    return paramiko


def ssh_connect(host: str, user: str, password: str, timeout_s: int):
    paramiko = load_paramiko()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=user,
        password=password,
        timeout=timeout_s,
        banner_timeout=timeout_s,
        auth_timeout=timeout_s,
    )
    return client


def run_ssh(host: str, user: str, password: str, command: str, timeout_s: int) -> Tuple[int, str, str]:
    client = ssh_connect(host, user, password, min(timeout_s, 30))
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout_s)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        stdin.close()
        stdout.close()
        stderr.close()
        return rc, out, err
    finally:
        client.close()


def upload_file(host: str, user: str, password: str, local: Path, remote: str, timeout_s: int) -> None:
    client = ssh_connect(host, user, password, min(timeout_s, 30))
    try:
        sftp = client.open_sftp()
        try:
            sftp.put(str(local), remote)
        finally:
            sftp.close()
    finally:
        client.close()


def shell_join(commands: Iterable[str]) -> str:
    return " && ".join(commands)


def infer_tool_env(cann_env: str) -> str:
    marker = "/cann/cann/set_env.sh"
    if cann_env.endswith(marker):
        return cann_env[: -len(marker)]
    return ""


def remote_env_prelude(cann_env: str, tool_env: str) -> str:
    commands = []
    if tool_env:
        commands.extend([
            f"export TILEXR_EXTERNAL_ENV={shlex.quote(tool_env)}",
            f"export PATH={shlex.quote(tool_env)}/util/cmake/bin:{shlex.quote(tool_env)}/util/mpich/bin:$PATH",
            f"export PATH={shlex.quote(tool_env)}/temp/cmake/bin:{shlex.quote(tool_env)}/temp/mpich/bin:$PATH",
            f"export LD_LIBRARY_PATH={shlex.quote(tool_env)}/util/mpich/lib:{shlex.quote(tool_env)}/temp/mpich/lib:$LD_LIBRARY_PATH",
        ])
    commands.append(f"source {shlex.quote(cann_env)} >/dev/null 2>&1")
    return "; ".join(commands)


def sync_host(
    host: str,
    user: str,
    password: str,
    archive: Path,
    remote_dir: str,
    clean_remote: bool,
    timeout_s: int,
) -> None:
    remote_archive = f"/tmp/{archive.name}"
    upload_file(host, user, password, archive, remote_archive, timeout_s)
    setup = []
    if clean_remote:
        setup.append(f"rm -rf -- {shlex.quote(remote_dir)}")
    setup.extend([
        f"mkdir -p -- {shlex.quote(remote_dir)}",
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_dir)}",
        f"rm -f -- {shlex.quote(remote_archive)}",
    ])
    rc, out, err = run_ssh(host, user, password, f"bash -lc {shlex.quote(shell_join(setup))}", timeout_s)
    if rc != 0:
        raise RuntimeError(f"sync failed on {host}: rc={rc}\n{out}\n{err}")


def build_host(host: str, user: str, password: str, remote_dir: str, cann_env: str, tool_env: str, timeout_s: int) -> None:
    command = shell_join([
        remote_env_prelude(cann_env, tool_env),
        f"cd {shlex.quote(remote_dir)}",
        "command -v cmake >/dev/null",
        "bash tests/udma/build.sh",
    ])
    rc, out, err = run_ssh(host, user, password, f"bash -lc {shlex.quote(command)}", timeout_s)
    if rc != 0:
        raise RuntimeError(f"build failed on {host}: rc={rc}\n{out[-4000:]}\n{err[-4000:]}")


def make_rank_command(
    *,
    rank: int,
    world_size: int,
    repeat: int,
    elements_per_peer: int,
    devices_per_host: int,
    remote_dir: str,
    cann_env: str,
    demo_bin: str,
    comm_id: str,
    barrier_host: str,
    result_dir: str,
    route_policy: str,
    timeout_s: int,
    tool_env: str,
) -> str:
    log_file = f"{result_dir}/rank_{rank}.log"
    rc_file = f"{result_dir}/rank_{rank}.rc"
    inner = shell_join([
        f"mkdir -p {shlex.quote(result_dir)}",
        remote_env_prelude(cann_env, tool_env),
        f"cd {shlex.quote(remote_dir)}",
        f"export TILEXR_COMM_ID={shlex.quote(comm_id)}",
        f"export TILEXR_DEMO_BARRIER_HOST={shlex.quote(barrier_host)}",
        f"export TILEXR_UDMA_ROUTE_POLICY={shlex.quote(route_policy)}",
        "export TILEXR_DEMO_BIGDATA_REMOTE_PUT_ONLY=1",
        "export TILEXR_DEMO_BIGDATA_PROFILE_STAGE=7",
        f"export TILEXR_DEMO_ALLTOALL_REPEAT={repeat}",
        "export TILEXR_DEMO_ALLTOALL_WARMUP=0",
        "export TILEXR_DEMO_ALLTOALL_SYNC_AT_END=1",
        f"export TILEXR_DEMO_TIMEOUT_SECONDS={timeout_s}",
        (
            f"timeout {timeout_s + 60} ./{shlex.quote(demo_bin)} "
            f"{world_size} {rank} 7 {elements_per_peer} {devices_per_host} 0 "
            f"> {shlex.quote(log_file)} 2>&1; "
            f"rc=$?; echo $rc > {shlex.quote(rc_file)}; "
            f"tail -n 80 {shlex.quote(log_file)}; exit $rc"
        ),
    ])
    return f"bash -lc {shlex.quote('set -o pipefail; ' + inner)}"


def run_rank(
    host: str,
    user: str,
    password: str,
    command: str,
    rank: int,
    timeout_s: int,
) -> RankRunResult:
    start = time.perf_counter()
    try:
        rc, out, err = run_ssh(host, user, password, command, timeout_s + 90)
    except Exception as exc:
        return RankRunResult(rank=rank, host=host, rc=999, stdout="", stderr=repr(exc), elapsed_s=time.perf_counter() - start)
    return RankRunResult(rank=rank, host=host, rc=rc, stdout=out, stderr=err, elapsed_s=time.perf_counter() - start)


def run_all_ranks(args, hosts: Sequence[str], repeat: int, label: str) -> Tuple[List[RankRunResult], float, str]:
    port = args.base_port + (int(time.time()) % 1000)
    if label != "measure":
        port += 1000
    comm_id = f"{hosts[0]}:{port}"
    result_dir = f"{args.result_prefix}_{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    commands = []
    for rank in range(args.world_size):
        host = hosts[rank // args.devices_per_host]
        command = make_rank_command(
            rank=rank,
            world_size=args.world_size,
            repeat=repeat,
            elements_per_peer=args.elements_per_peer,
            devices_per_host=args.devices_per_host,
            remote_dir=args.remote_dir,
            cann_env=args.cann_env,
            demo_bin=args.demo_bin,
            comm_id=comm_id,
            barrier_host=hosts[0],
            result_dir=result_dir,
            route_policy=args.route_policy,
            timeout_s=args.timeout,
            tool_env=args.tool_env,
        )
        commands.append((rank, host, command))

    started = time.perf_counter()
    results: List[RankRunResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.world_size) as executor:
        futures = [
            executor.submit(run_rank, host, args.user, args.password, command, rank, args.timeout)
            for rank, host, command in commands
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{label}] rank {result.rank:02d} host {result.host} rc={result.rc} elapsed={result.elapsed_s:.2f}s", flush=True)
    elapsed = time.perf_counter() - started
    return sorted(results, key=lambda result: result.rank), elapsed, result_dir


def collect_metrics(results: Sequence[RankRunResult]) -> Tuple[List[RankMetric], List[str]]:
    metrics: List[RankMetric] = []
    errors: List[str] = []
    for result in results:
        text = f"{result.stdout}\n{result.stderr}"
        parsed_metric = None
        try:
            parsed_metric = parse_demo_metric(text)
            metrics.append(parsed_metric)
        except ValueError:
            pass
        if result.rc != 0:
            errors.append(f"rank {result.rank} on {result.host} rc={result.rc}: {text[-2000:]}")
            continue
        for token in ("ERROR", "-1001", "-1002", "-1003"):
            if token in text:
                errors.append(f"rank {result.rank} on {result.host} contains {token}: {text[-2000:]}")
                break
        if parsed_metric is None:
            errors.append(f"rank {result.rank} on {result.host}: no alltoall udma-bigdata metric line found: {text[-2000:]}")
    return metrics, errors


def parse_hosts(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Sync TileXR and run UDMA bigdata alltoall performance tests.",
        epilog=(
            "Examples: "
            "py tests/udma/perf/run_udma_alltoall_perf.py --world-size 16 --bytes-per-peer 8MiB --warmup 1 --loops 100; "
            "py tests/udma/perf/run_udma_alltoall_perf.py --world-size 64 --bytes-per-peer 4MiB --warmup 1 --loops 100"
        ),
    )
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS), help="Comma-separated host list in rank order.")
    parser.add_argument("--world-size", type=int, default=16, help="Total rank count, for example 8, 16, 32, or 64.")
    parser.add_argument("--devices-per-host", type=int, default=8, help="NPU ranks per host.")
    parser.add_argument("--bytes-per-peer", default="8MiB", help="Payload bytes per peer, for example 4MiB, 8MiB, 16MiB.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations in a separate launch. Use 0 to skip.")
    parser.add_argument("--loops", type=int, default=100, help="Measured iterations.")
    parser.add_argument("--user", default="root", help="SSH user.")
    parser.add_argument("--password", default=os.environ.get("TILEXR_REMOTE_PASSWORD", "Admin@9000"), help="SSH password.")
    parser.add_argument("--remote-dir", default="/home/tileXR", help="Remote TileXR source path.")
    parser.add_argument("--cann-env", default=DEFAULT_CANN_ENV, help="Remote CANN set_env.sh path.")
    parser.add_argument(
        "--tool-env",
        default=DEFAULT_TOOL_ENV,
        help="External TileXR env root containing util/cmake and util/mpich. Defaults to the parent env inferred from --cann-env.",
    )
    parser.add_argument("--demo-bin", default=DEFAULT_DEMO_BIN, help="Demo binary path relative to remote-dir.")
    parser.add_argument("--route-policy", default="all", help="TILEXR_UDMA_ROUTE_POLICY value.")
    parser.add_argument("--timeout", type=int, default=180, help="Kernel/demo timeout seconds.")
    parser.add_argument("--build-timeout", type=int, default=900, help="Remote build timeout seconds.")
    parser.add_argument("--sync-timeout", type=int, default=300, help="Remote sync timeout seconds.")
    parser.add_argument("--base-port", type=int, default=32000, help="Base rendezvous port.")
    parser.add_argument("--result-prefix", default="/tmp/tilexr_udma_alltoall_perf", help="Remote result dir prefix.")
    parser.add_argument("--skip-sync", action="store_true", help="Do not upload local source tree.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build tests/udma on remote hosts.")
    parser.add_argument("--clean-remote", action="store_true", help="Remove remote-dir before extracting uploaded source.")
    parsed = parser.parse_args(argv)
    parsed.hosts = parse_hosts(parsed.hosts)
    parsed.bytes_per_peer_value = parse_size_to_bytes(parsed.bytes_per_peer)
    parsed.elements_per_peer = bytes_to_int32_elements(parsed.bytes_per_peer_value)
    if not parsed.tool_env:
        parsed.tool_env = infer_tool_env(parsed.cann_env)
    if parsed.warmup < 0 or parsed.loops <= 0:
        raise ValueError("--warmup must be >= 0 and --loops must be > 0")
    return parsed


def print_summary(args, hosts: Sequence[str], metrics: Sequence[RankMetric], host_elapsed_s: float, result_dir: str) -> None:
    summary = summarize_metrics(metrics)
    total_payload = args.bytes_per_peer_value * args.world_size
    remote_payload = args.bytes_per_peer_value * max(args.world_size - args.devices_per_host, 0)
    host_per_iter_us = host_elapsed_s * 1_000_000.0 / args.loops
    print("")
    print("=== TileXR UDMA alltoall performance ===")
    print(f"hosts: {','.join(hosts)}")
    print(f"world_size: {args.world_size}")
    print(f"bytes_per_peer: {args.bytes_per_peer_value} ({args.bytes_per_peer})")
    print(f"demo_payload_per_rank: {total_payload} bytes")
    print(f"remote_only_payload_per_rank: {remote_payload} bytes")
    print(f"warmup_iters: {args.warmup}")
    print(f"measured_iters: {args.loops}")
    print(f"remote_result_dir: {result_dir}")
    print(f"host_wall_time: {host_elapsed_s:.6f} s")
    print(f"host_wall_per_iter: {host_per_iter_us:.3f} us")
    print(
        "operator_per_iter_us: "
        f"count={summary['count']} min={summary['min_us']:.3f} p50={summary['p50_us']:.3f} "
        f"mean={summary['mean_us']:.3f} p90={summary['p90_us']:.3f} "
        f"p99={summary['p99_us']:.3f} max={summary['max_us']:.3f}"
    )
    bw_values = [metric.bandwidth_gbps for metric in metrics]
    print(
        "operator_bw_GBps: "
        f"min={min(bw_values):.3f} mean={statistics.mean(bw_values):.3f} max={max(bw_values):.3f}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        hosts = select_hosts(args.hosts, args.world_size, args.devices_per_host)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2

    archive: Optional[Path] = None
    try:
        if not args.skip_sync:
            archive = create_source_archive(repo_root())
            print(f"sync archive: {archive} ({archive.stat().st_size} bytes)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
                futures = [
                    executor.submit(
                        sync_host,
                        host,
                        args.user,
                        args.password,
                        archive,
                        args.remote_dir,
                        args.clean_remote,
                        args.sync_timeout,
                    )
                    for host in hosts
                ]
                for host, future in zip(hosts, futures):
                    future.result()
                    print(f"sync {host} ok")

        if not args.skip_build:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
                futures = [
                    executor.submit(
                        build_host,
                        host,
                        args.user,
                        args.password,
                        args.remote_dir,
                        args.cann_env,
                        args.tool_env,
                        args.build_timeout,
                    )
                    for host in hosts
                ]
                for host, future in zip(hosts, futures):
                    future.result()
                    print(f"build {host} ok")

        if args.warmup > 0:
            warmup_results, warmup_elapsed, warmup_dir = run_all_ranks(args, hosts, args.warmup, "warmup")
            _warmup_metrics, warmup_errors = collect_metrics(warmup_results)
            if warmup_errors:
                if _warmup_metrics:
                    print_summary(args, hosts, _warmup_metrics, warmup_elapsed, warmup_dir)
                print("\n".join(warmup_errors), file=sys.stderr)
                return 1
            print(f"warmup ok: host_elapsed={warmup_elapsed:.6f}s result_dir={warmup_dir}")

        measure_results, host_elapsed, result_dir = run_all_ranks(args, hosts, args.loops, "measure")
        metrics, errors = collect_metrics(measure_results)
        if errors:
            if metrics:
                print_summary(args, hosts, metrics, host_elapsed, result_dir)
            print("\n".join(errors), file=sys.stderr)
            return 1
        if len(metrics) != args.world_size:
            print(f"expected {args.world_size} metrics, got {len(metrics)}", file=sys.stderr)
            return 1
        print_summary(args, hosts, metrics, host_elapsed, result_dir)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if archive is not None:
            try:
                archive.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
