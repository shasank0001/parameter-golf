from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROXY_STEPS, FinalMetrics, find_log_path, matching_proxy_steps, parse_final_metrics, parse_step_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two run logs")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def build_report(baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    base_text = baseline_path.read_text(encoding="utf-8", errors="replace")
    cand_text = candidate_path.read_text(encoding="utf-8", errors="replace")
    base_steps = parse_step_losses(base_text)
    cand_steps = parse_step_losses(cand_text)
    steps = [step for step in PROXY_STEPS if step in base_steps and step in cand_steps]
    if not steps:
        steps = matching_proxy_steps(base_steps, cand_steps)
    matched = [
        {
            "step": step,
            "baseline_loss": base_steps[step],
            "candidate_loss": cand_steps[step],
            "delta": cand_steps[step] - base_steps[step],
        }
        for step in steps
    ]
    recent = matched[-10:] if matched else []
    base_final: FinalMetrics = parse_final_metrics(base_text)
    cand_final: FinalMetrics = parse_final_metrics(cand_text)
    avg_delta = sum(item["delta"] for item in recent) / len(recent) if recent else None
    recommendation = "insufficient_data"
    if cand_final.has_final and base_final.has_final and cand_final.val_bpb is not None and base_final.val_bpb is not None:
        recommendation = "promote" if cand_final.val_bpb < base_final.val_bpb else "discard"
    elif avg_delta is not None:
        recommendation = "promote" if avg_delta < 0 else "discard"
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "avg_recent_delta": avg_delta,
        "matched_steps": matched,
        "baseline_final": base_final.__dict__,
        "candidate_final": cand_final.__dict__,
        "recommendation": recommendation,
    }


def main() -> None:
    args = parse_args()
    report = build_report(find_log_path(args.baseline), find_log_path(args.candidate))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"baseline: {report['baseline']}")
    print(f"candidate: {report['candidate']}")
    print(f"recommendation: {report['recommendation']}")
    print(f"avg_recent_delta: {report['avg_recent_delta']}")
    print("recent_steps:")
    for item in report["matched_steps"][-10:]:
        print(
            f"  step {item['step']}: baseline={item['baseline_loss']:.4f} "
            f"candidate={item['candidate_loss']:.4f} delta={item['delta']:+.4f}"
        )
    candidate_final = report["candidate_final"]
    baseline_final = report["baseline_final"]
    if baseline_final["has_final"]:
        print(
            f"baseline_final: val_bpb={baseline_final['val_bpb']:.6f} "
            f"artifact_bytes={baseline_final['artifact_bytes']}"
        )
    if candidate_final["has_final"]:
        print(
            f"candidate_final: val_bpb={candidate_final['val_bpb']:.6f} "
            f"artifact_bytes={candidate_final['artifact_bytes']}"
        )


if __name__ == "__main__":
    main()
