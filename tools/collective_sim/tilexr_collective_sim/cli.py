import argparse
import json
from pathlib import Path

from .io import load_algorithm, load_calibration, load_case, load_sweep, load_topology
from .model import ValidationReport, dataclass_to_plain, report_from_issues
from .report import (
    write_html_report,
    write_html_report_from_plain,
    write_result,
    write_results,
    write_summary,
)
from .semantics import validate_allgather_semantics
from .simulator import simulate_algorithm
from .validation import validate_static


def cmd_validate(args: argparse.Namespace) -> int:
    algorithm = load_algorithm(Path(args.algorithm))
    topology = load_topology(Path(args.topology)) if args.topology else None
    report = _validation_report(algorithm, topology)
    if report.ok:
        print("validation ok")
        return 0
    for item in report.issues:
        location = ""
        if item.op_id is not None:
            location = f" op={item.op_id}"
        elif item.buffer_id is not None:
            location = f" buffer={item.buffer_id}"
        print(f"{item.code}:{location} {item.message}")
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    case_path = Path(args.case).resolve()
    case_dir = case_path.parent
    case = load_case(case_path)
    algorithm = load_algorithm(_resolve_from(case_dir, case.algorithm))
    topology = load_topology(_resolve_from(case_dir, case.topology))
    calibration = load_calibration(_resolve_from(case_dir, case.calibration))
    results = [
        simulate_algorithm(algorithm, topology, calibration, message_bytes, validate=case.validate)
        for message_bytes in case.message_bytes
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_result(results[0], out_dir)
    write_results(results, out_dir / "results.json")
    write_summary(results, out_dir / "summary.csv")
    write_html_report(results, out_dir / "report.html")
    print(f"wrote {out_dir}")
    return 0 if all(result.validation.ok for result in results) else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    sweep_path = Path(args.sweep).resolve()
    base = sweep_path.parent
    sweep = load_sweep(sweep_path)
    calibration = load_calibration(_resolve_from(base, sweep["calibration"]))
    results = []
    for case in sweep["cases"]:
        algorithm = load_algorithm(_resolve_from(base, case["algorithm"]))
        topology = load_topology(_resolve_from(base, case["topology"]))
        for size in sweep["message_bytes"]:
            results.append(
                simulate_algorithm(
                    algorithm,
                    topology,
                    calibration,
                    int(size),
                    validate=bool(sweep.get("validate", True)),
                )
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps([dataclass_to_plain(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    if results:
        write_result(results[0], out_dir)
    write_summary(results, out_dir / "summary.csv")
    write_html_report(results, out_dir / "report.html")
    print(f"wrote {out_dir}")
    return 0 if all(result.validation.ok for result in results) else 1


def cmd_report(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    write_html_report_from_plain(data, Path(args.out))
    print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="collective-sim")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate an algorithm file")
    validate.add_argument("algorithm")
    validate.add_argument("--topology")
    validate.set_defaults(func=cmd_validate)

    run = subcommands.add_parser("run", help="run a simulation case")
    run.add_argument("case")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    sweep = subcommands.add_parser("sweep", help="run every algorithm/topology/message-size combination")
    sweep.add_argument("sweep")
    sweep.add_argument("--out", required=True)
    sweep.set_defaults(func=cmd_sweep)

    report = subcommands.add_parser("report", help="regenerate static HTML from result JSON")
    report.add_argument("result")
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


def _resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _validation_report(algorithm, topology) -> ValidationReport:
    static_report = validate_static(algorithm, topology)
    if algorithm.collective != "allgather":
        return static_report
    semantic_report = validate_allgather_semantics(algorithm)
    if static_report.ok:
        return semantic_report
    issues = list(static_report.issues)
    for item in semantic_report.issues:
        if item not in issues:
            issues.append(item)
    return report_from_issues(issues)


if __name__ == "__main__":
    raise SystemExit(main())
