from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_DIR / "logs"
MEMORY_PATH = REPO_DIR / "memory.md"
MAX_ARTIFACT_BYTES = 16_000_000
PROXY_STEPS = (100, 500, 1000, 2000, 5000)


@dataclass
class FinalMetrics:
    val_bpb: float | None
    val_loss: float | None
    artifact_bytes: int | None
    peak_allocated_mib: int | None
    peak_reserved_mib: int | None
    runtime_min: float | None
    step_avg_ms: float | None
    step: int | None
    total_steps: int | None
    has_final: bool


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_DIR), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout.strip()


def git_metadata() -> tuple[str, str]:
    return git_output("branch", "--show-current") or "unknown", git_output("rev-parse", "--short", "HEAD") or "unknown"


def git_is_dirty() -> bool:
    return bool(git_output("status", "--porcelain"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_step_losses(text: str) -> dict[int, float]:
    return {int(step): float(loss) for step, loss in re.findall(r"step:(\d+)/\d+ train_loss:([0-9.]+)", text)}


def parse_latest_step(text: str) -> tuple[int | None, int | None]:
    matches = re.findall(r"step:(\d+)/(\d+)", text)
    if not matches:
        return None, None
    step, total = matches[-1]
    return int(step), int(total)


def parse_final_metrics(text: str) -> FinalMetrics:
    exact = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", text)
    approx = re.search(r"final_int8_zlib_roundtrip val_loss:([0-9.]+) val_bpb:([0-9.]+)", text)
    size = re.search(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes", text)
    peak = re.search(r"peak memory allocated:\s*(\d+)\s*MiB reserved:\s*(\d+)\s*MiB", text)
    step_avg = re.findall(r"step_avg:([0-9.]+)ms", text)
    runtime = re.findall(r"train_time:([0-9.]+)ms", text)
    step, total_steps = parse_latest_step(text)
    chosen = exact or approx
    return FinalMetrics(
        val_bpb=float(chosen.group(2)) if chosen else None,
        val_loss=float(chosen.group(1)) if chosen else None,
        artifact_bytes=int(size.group(1)) if size else None,
        peak_allocated_mib=int(peak.group(1)) if peak else None,
        peak_reserved_mib=int(peak.group(2)) if peak else None,
        runtime_min=(float(runtime[-1]) / 60000.0) if runtime else None,
        step_avg_ms=float(step_avg[-1]) if step_avg else None,
        step=step,
        total_steps=total_steps,
        has_final=bool(chosen),
    )


def current_best_bpb(memory_text: str) -> float | None:
    match = re.search(r"Current Best.*?\n- val_bpb: ([0-9.]+)", memory_text, re.DOTALL)
    return float(match.group(1)) if match else None


def result_status(metrics: FinalMetrics, best_before: float | None) -> str:
    if not metrics.has_final or metrics.val_bpb is None:
        return "CRASH"
    if metrics.artifact_bytes is not None and metrics.artifact_bytes > MAX_ARTIFACT_BYTES:
        return "SKIPPED"
    if best_before is None or metrics.val_bpb < best_before:
        return "KEEP"
    return "DISCARD"


def matching_proxy_steps(a: dict[int, float], b: dict[int, float]) -> list[int]:
    return sorted(set(a) & set(b))


def format_bytes_mb(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1_000_000:.2f}"


def find_log_path(run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path)
    if candidate.exists():
        return candidate
    return LOGS_DIR / f"{run_id_or_path}.txt"


def run_id_from_path(path: Path) -> str:
    return path.name[:-4] if path.name.endswith(".txt") else path.stem


def replace_marked_section(text: str, marker: str, body: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    replacement = f"{start}\n{body.rstrip()}\n{end}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    return text.rstrip() + "\n\n" + replacement + "\n"
