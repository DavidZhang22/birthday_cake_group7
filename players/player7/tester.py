import argparse
import concurrent.futures as futures
import csv
import os
import re
import sys
import time
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import Iterable

SCORE_SIZE_RE = re.compile(r"size\s+span:\s*([0-9]*\.?[0-9]+)\s*cm\^2", re.IGNORECASE)
SCORE_STD_RE = re.compile(r"stdev\(ratio\):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
PLAYER_RE = re.compile(r"player(\d+)", re.IGNORECASE)


def _find_project_root(start: Path) -> Path:
    # Walk up until we see main.py
    for p in [start, *start.parents]:
        if (p / "main.py").exists():
            return p
    raise RuntimeError("Could not locate project root (main.py)")


project_root = _find_project_root(Path(__file__).resolve())


def find_cakes(root: Path) -> list[Path]:
    # all csv files under players/player1..player10
    print(f"Searching for cakes under {root}")
    return sorted(root.glob("players/player*/*.csv"))


def extract_player_num(path: Path) -> int | None:
    m = PLAYER_RE.search(str(path))
    return int(m.group(1)) if m else None


def parse_scores(stdout: str) -> tuple[float | None, float | None]:
    m1 = SCORE_SIZE_RE.search(stdout)
    m2 = SCORE_STD_RE.search(stdout)
    size_span = float(m1.group(1)) if m1 else None
    stdev = float(m2.group(1)) if m2 else None
    return size_span, stdev


def build_cmd(args, cake_path: Path, children: int, player: int) -> list[str]:
    # Base command (no GUI for batch; add --gui via flag if needed)
    cake_rel = Path(cake_path).resolve().relative_to(project_root)

    cmd = [
        args.uv,
        "run",
        "./main.py",
        "--import-cake",
        str(cake_rel),
        "--player",
        "7",
        "--children",
        str(children),
    ]
    if args.gui:
        cmd.insert(3, "--gui")
    if args.profile:
        # uv run -m cProfile -o profile.out main.py ...
        cmd = [
            args.uv,
            "run",
            "-m",
            "cProfile",
            "-o",
            f"profile_{player}_{children}.out",
            "../../main.py",
            "--import-cake",
            str(cake_path),
            "--player",
            str(player),
            "--children",
            str(children),
        ]
        if args.gui:
            cmd.insert(6, "--gui")
    return cmd


def run_one(args, cake_path: Path, children: int) -> dict:
    t0 = time.perf_counter()
    player = extract_player_num(cake_path) or args.default_player
    cmd = build_cmd(args, cake_path, children, player)
    proc = run(cmd, stdout=PIPE, stderr=STDOUT, text=True)
    dt = time.perf_counter() - t0

    size_span, stdev = parse_scores(proc.stdout or "")
    return {
        "cake_path": str(cake_path),
        "cake_name": cake_path.name,
        "player": player,
        "children": children,
        "returncode": proc.returncode,
        "duration_s": round(dt, 3),
        "size_span_cm2": size_span,
        "stdev_ratio": stdev,
        "stdout_snippet": (proc.stdout[-500:] if proc.stdout else ""),
    }


def chunked(it: Iterable, n: int):
    it = list(it)
    for i in range(0, len(it), n):
        yield it[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default="./cakes", help="Root folder containing player*/**/*.csv"
    )
    ap.add_argument(
        "--children",
        nargs="+",
        type=int,
        default=[7, 10],
        help="Children counts to evaluate",
    )
    ap.add_argument("--uv", default="uv", help="uv executable (e.g., uv or uv.exe)")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 2),
        help="Parallel processes",
    )
    ap.add_argument(
        "--out-csv", default="cake_scores.csv", help="Output CSV of all runs"
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Run with --gui (slower; usually avoid for batch)",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile with per-run profile_X_Y.out",
    )
    ap.add_argument(
        "--default-player",
        type=int,
        default=7,
        help="Fallback player if path parsing fails",
    )
    ap.add_argument("--dry", action="store_true", help="List runs without executing")
    ap.add_argument(
        "--plots", action="store_true", help="Also generate matplotlib plots"
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cakes = find_cakes(root)

    if not cakes:
        print(f"No cakes found under {root}", file=sys.stderr)
        sys.exit(1)

    jobs = []
    for cake in cakes:
        player = args.default_player
        for k in args.children:
            jobs.append((cake, k, player))

    print(
        f"Discovered {len(cakes)} cakes; planning {len(jobs)} runs "
        f"({len(args.children)} children configs). Workers={args.workers}"
    )

    if args.dry:
        for cake, k, player in jobs:
            print(build_cmd(args, cake, k, player))
        return

    results = []
    # Use processes to avoid any potential GUI/GIL overhead; also robust for Python-heavy parts
    with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args, cake, k) for (cake, k, _) in jobs]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            ok = (
                rec["returncode"] == 0
                and rec["size_span_cm2"] is not None
                and rec["stdev_ratio"] is not None
            )
            print(
                f"[{i}/{len(futs)}] player={rec['player']} children={rec['children']} "
                f"cake={rec['cake_name']} -> "
                f"span={rec['size_span_cm2']} cm^2, stdev={rec['stdev_ratio']} | "
                f"rc={rec['returncode']} time={rec['duration_s']}s{' OK' if ok else ' !'}"
            )

    # Write CSV
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cake_path",
                "cake_name",
                "player",
                "children",
                "returncode",
                "duration_s",
                "size_span_cm2",
                "stdev_ratio",
                "stdout_snippet",
            ],
        )
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
