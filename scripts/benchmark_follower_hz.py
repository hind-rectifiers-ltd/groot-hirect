#!/usr/bin/env python3
"""
Benchmark RobStride follower CAN throughput at a target control rate.

Use this before moving recording / teleop to 30 Hz. It reports whether the
current synchronous ``ActuatorController`` path can sustain read/write loops
without missed deadlines, read failures, or zero dropouts.

Modes (in increasing cost):
  read         — ``read_joints12()`` only
  write        — hold position via ``command_joints12()`` (includes safety encoder read)
  teleop       — one read + command hold (matches direct teleop / policy tick shape)
  record       — N-sample median read + command with ``feedback12`` (matches ``record_episodes_3cam``)
  record_legacy — old path: median read + command without ``feedback12`` (4 reads/tick at N=3)

Examples:
  uv run python scripts/benchmark_follower_hz.py --rate-hz 30 --mode read --duration-s 20
  uv run python scripts/benchmark_follower_hz.py --rates 10 20 30 --mode teleop
  uv run python scripts/benchmark_follower_hz.py --rates 10 20 30 --mode record --duration-s 15
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from move_actuators import ActuatorController, JOINT_MOTOR_IDS  # noqa: E402

_LABELS = ("L0", "L1", "L2", "L3", "L4", "Lg", "R0", "R1", "R2", "R3", "R4", "Rg")


@dataclass
class LoopStats:
    target_hz: float
    mode: str
    duration_s: float
    loops: int = 0
    loop_times_ms: list[float] = field(default_factory=list)
    missed_deadlines: int = 0
    zero_joint_events: int = 0
    read_failures_start: int = 0
    read_failures_end: int = 0
    read_attempts_start: int = 0
    read_attempts_end: int = 0
    rx_drained_start: int = 0
    rx_drained_end: int = 0

    @property
    def read_failures(self) -> int:
        return self.read_failures_end - self.read_failures_start

    @property
    def read_attempts(self) -> int:
        return self.read_attempts_end - self.read_attempts_start

    @property
    def actual_hz(self) -> float:
        if not self.loop_times_ms:
            return 0.0
        return self.loops / max(self.duration_s, 1e-9)

    def percentile_ms(self, p: float) -> float:
        if not self.loop_times_ms:
            return 0.0
        xs = sorted(self.loop_times_ms)
        idx = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[idx]


def _snapshot_arm_stats(arm: ActuatorController) -> tuple[int, int, int]:
    return arm._read_failures, arm._read_attempts, arm._rx_frames_drained


def _count_zero_joints(q: np.ndarray, live_mask: np.ndarray, baseline: np.ndarray) -> int:
    """Count joints that read 0.0 while bus is live and baseline was non-zero."""
    n = 0
    for i in range(12):
        if not live_mask[i]:
            continue
        if abs(float(baseline[i])) < 1e-4:
            continue
        if abs(float(q[i])) < 1e-6:
            n += 1
    return n


def _live_mask(arm: ActuatorController) -> np.ndarray:
    mask = np.zeros(12, dtype=bool)
    for i in range(6):
        if arm._left_bus is not None:
            mask[i] = True
        if arm._right_bus is not None:
            mask[6 + i] = True
    return mask


def _read_median12(
    arm: ActuatorController,
    *,
    samples: int,
    sample_gap_s: float,
) -> np.ndarray:
    n = max(1, samples)
    stack = []
    for s in range(n):
        stack.append(arm.read_joints12())
        if s + 1 < n and sample_gap_s > 0.0:
            time.sleep(sample_gap_s)
    return np.median(np.stack(stack, axis=0), axis=0)


def _run_loop(
    arm: ActuatorController,
    *,
    mode: str,
    target_hz: float,
    duration_s: float,
    median_samples: int,
    median_gap_s: float,
) -> LoopStats:
    period = 1.0 / max(target_hz, 0.25)
    stats = LoopStats(target_hz=target_hz, mode=mode, duration_s=duration_s)

    baseline = arm.read_joints12()
    arm.seed_ramp_from_angles(baseline)
    live = _live_mask(arm)
    hold_target = baseline.copy()

    stats.read_failures_start, stats.read_attempts_start, stats.rx_drained_start = _snapshot_arm_stats(arm)

    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t0 = time.monotonic()

        if mode == "read":
            q = arm.read_joints12()
            stats.zero_joint_events += _count_zero_joints(q, live, baseline)
        elif mode == "write":
            sent = arm.command_joints12(hold_target, ramp=True)
            hold_target = sent.copy()
        elif mode == "teleop":
            q = arm.read_joints12()
            stats.zero_joint_events += _count_zero_joints(q, live, baseline)
            arm.command_joints12(q, ramp=True, feedback12=q)
        elif mode == "record":
            raw_last: np.ndarray | None = None
            stack: list[np.ndarray] = []
            n = max(1, median_samples)
            for s in range(n):
                raw_last = arm.read_joints12()
                stack.append(raw_last)
                stats.zero_joint_events += _count_zero_joints(raw_last, live, baseline)
                if s + 1 < n and median_gap_s > 0.0:
                    time.sleep(median_gap_s)
            assert raw_last is not None
            hold = stack[0] if n == 1 else np.median(np.stack(stack, axis=0), axis=0)
            arm.command_joints12(hold, ramp=True, feedback12=raw_last)
        elif mode == "record_legacy":
            q = _read_median12(arm, samples=median_samples, sample_gap_s=median_gap_s)
            stats.zero_joint_events += _count_zero_joints(q, live, baseline)
            arm.command_joints12(q, ramp=True)
        else:
            raise ValueError(f"unknown mode: {mode}")

        elapsed = time.monotonic() - t0
        stats.loop_times_ms.append(elapsed * 1000.0)
        stats.loops += 1
        if elapsed > period:
            stats.missed_deadlines += 1

        sleep_s = period - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    stats.read_failures_end, stats.read_attempts_end, stats.rx_drained_end = _snapshot_arm_stats(arm)
    return stats


def _passes(stats: LoopStats, *, min_hz_fraction: float, max_missed_fraction: float) -> bool:
    if stats.loops == 0:
        return False
    hz_ok = stats.actual_hz >= stats.target_hz * min_hz_fraction
    missed_frac = stats.missed_deadlines / stats.loops
    missed_ok = missed_frac <= max_missed_fraction
    fail_rate = stats.read_failures / max(stats.read_attempts, 1)
    read_ok = fail_rate <= 0.01
    zero_ok = stats.zero_joint_events == 0
    return hz_ok and missed_ok and read_ok and zero_ok


def _print_report(stats: LoopStats, *, min_hz_fraction: float, max_missed_fraction: float) -> bool:
    lt = stats.loop_times_ms
    period_ms = 1000.0 / max(stats.target_hz, 1e-3)
    fail_rate = 100.0 * stats.read_failures / max(stats.read_attempts, 1)
    missed_frac = 100.0 * stats.missed_deadlines / max(stats.loops, 1)
    ok = _passes(stats, min_hz_fraction=min_hz_fraction, max_missed_fraction=max_missed_fraction)

    print(f"\n=== mode={stats.mode} target={stats.target_hz:.1f} Hz ===")
    print(f"  loops={stats.loops}  wall={stats.duration_s:.1f}s  actual_hz={stats.actual_hz:.2f}")
    if lt:
        print(
            f"  loop_ms: min={min(lt):.2f}  mean={statistics.mean(lt):.2f}  "
            f"p50={stats.percentile_ms(50):.2f}  p90={stats.percentile_ms(90):.2f}  "
            f"p99={stats.percentile_ms(99):.2f}  max={max(lt):.2f}  "
            f"budget={period_ms:.2f}"
        )
    print(
        f"  missed_deadlines={stats.missed_deadlines} ({missed_frac:.1f}%)  "
        f"read_failures={stats.read_failures}/{stats.read_attempts} ({fail_rate:.2f}%)  "
        f"zero_joint_events={stats.zero_joint_events}  "
        f"rx_drained={stats.rx_drained_end - stats.rx_drained_start}"
    )
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    if not ok:
        if stats.actual_hz < stats.target_hz * min_hz_fraction:
            print(f"    - actual_hz {stats.actual_hz:.2f} < {stats.target_hz * min_hz_fraction:.2f}")
        if missed_frac / 100.0 > max_missed_fraction:
            print(f"    - missed_deadlines {missed_frac:.1f}% > {100 * max_missed_fraction:.1f}%")
        if fail_rate > 1.0:
            print(f"    - read failure rate {fail_rate:.2f}% > 1%")
        if stats.zero_joint_events > 0:
            print("    - zero_joint_events > 0 (likely CAN read dropouts)")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark RobStride follower CAN at target Hz.")
    p.add_argument("--rate-hz", type=float, default=None, help="Single target rate")
    p.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
        help="Sweep several rates (e.g. 10 20 30)",
    )
    p.add_argument(
        "--mode",
        choices=("read", "write", "teleop", "record", "record_legacy"),
        default="teleop",
        help="Loop shape to benchmark (default: teleop = read + command hold)",
    )
    p.add_argument("--duration-s", type=float, default=20.0, help="Seconds per rate trial")
    p.add_argument("--median-samples", type=int, default=3, help="For record mode")
    p.add_argument("--median-gap-ms", type=float, default=3.0, help="For record mode")
    p.add_argument(
        "--min-hz-fraction",
        type=float,
        default=0.95,
        help="PASS if actual_hz >= target * this (default 0.95)",
    )
    p.add_argument(
        "--max-missed-fraction",
        type=float,
        default=0.05,
        help="PASS if missed_deadlines/loops <= this (default 0.05)",
    )
    args = p.parse_args()

    rates = args.rates if args.rates else ([args.rate_hz] if args.rate_hz is not None else [10.0, 20.0, 30.0])
    rates = [float(r) for r in rates]

    print("RobStride follower CAN benchmark")
    print(f"  repo: {_REPO}")
    print(f"  mode: {args.mode}")
    print(f"  rates: {rates}")
    print(f"  duration_s: {args.duration_s}")
    print(f"  motors: {list(JOINT_MOTOR_IDS)}")
    print("  Hold arm still during the test. Ctrl+C to abort.\n", flush=True)

    arm = ActuatorController(ramp=True, safety_enabled=True)
    try:
        arm.connect()
        print(arm.read_stats_line(), flush=True)
        init_q = arm.read_joints12()
        prec = 4
        inner = ", ".join(f"{float(init_q[j]):.{prec}f}" for j in range(12))
        print(f"  initial qpos = [{inner}]", flush=True)

        all_ok = True
        for hz in rates:
            stats = _run_loop(
                arm,
                mode=args.mode,
                target_hz=hz,
                duration_s=args.duration_s,
                median_samples=max(1, args.median_samples),
                median_gap_s=max(0.0, args.median_gap_ms / 1000.0),
            )
            ok = _print_report(
                stats,
                min_hz_fraction=args.min_hz_fraction,
                max_missed_fraction=args.max_missed_fraction,
            )
            all_ok = all_ok and ok

        print("\n" + arm.read_stats_line(), flush=True)
        print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)
        if not all_ok and args.mode not in ("read", "record"):
            print(
                "Tip: if teleop/record FAIL but read PASS, the bottleneck is "
                "read+write contention (MIT status frames vs register reads). "
                "Try --mode read at 30 Hz first, then teleop, then record.",
                flush=True,
            )
        if not all_ok and args.mode == "record_legacy":
            print(
                "Tip: record_legacy does 3 reads + a 4th safety read (~47 ms). "
                "Use --mode record --median-samples 1 to match the optimized recorder.",
                flush=True,
            )
        raise SystemExit(0 if all_ok else 1)
    except KeyboardInterrupt:
        print("\nAborted.", flush=True)
        raise SystemExit(130)
    finally:
        try:
            arm.disconnect(send_zero=True)
        except Exception as exc:
            print(f"Disconnect warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
