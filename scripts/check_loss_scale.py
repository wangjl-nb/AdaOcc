#!/usr/bin/env python3
"""Check early AdaOcc training logs for finite, plausible loss scale."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

LOSS_RE = re.compile(r"(?<![A-Za-z_])loss:\s*([-+0-9.eEinfnaINFNA]+)")
CONTAIN_RE = re.compile(r"loss_containment:\s*([-+0-9.eEinfnaINFNA]+)")
QUERY_RE = re.compile(r"train_runtime_num_query:\s*([-+0-9.eE]+)")


def parse_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def inspect_log(path: Path, first_n: int, min_loss: float, max_loss: float, expected_query: int | None):
    losses = []
    contain = []
    num_query_seen = None
    for line in path.read_text(errors="replace").splitlines():
        if num_query_seen is None:
            m = QUERY_RE.search(line)
            if m:
                num_query_seen = int(round(parse_float(m.group(1))))
        m = LOSS_RE.search(line)
        if m and len(losses) < first_n:
            losses.append(parse_float(m.group(1)))
        m = CONTAIN_RE.search(line)
        if m and len(contain) < first_n:
            contain.append(parse_float(m.group(1)))
    failures = []
    if expected_query is not None and num_query_seen != expected_query:
        failures.append(f"expected train_runtime_num_query={expected_query}, got {num_query_seen}")
    if not losses:
        failures.append("no loss values found")
    bad_losses = [v for v in losses if not math.isfinite(v) or v < min_loss or v > max_loss]
    if bad_losses:
        failures.append(f"loss values outside [{min_loss}, {max_loss}] or non-finite: {bad_losses[:5]}")
    bad_contain = [v for v in contain if not math.isfinite(v)]
    if bad_contain:
        failures.append(f"non-finite loss_containment values: {bad_contain[:5]}")
    payload = {
        "ok": not failures,
        "path": str(path),
        "loss_count": len(losses),
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "containment_count": len(contain),
        "num_query_seen": num_query_seen,
        "failures": failures,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--first-n", type=int, default=20)
    parser.add_argument("--min-loss", type=float, default=0.1)
    parser.add_argument("--max-loss", type=float, default=20.0)
    parser.add_argument("--expected-query", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    path = Path(args.log_path)
    if not path.exists():
        payload = {"ok": False, "path": str(path), "failures": ["log file missing"]}
    else:
        payload = inspect_log(path, args.first_n, args.min_loss, args.max_loss, args.expected_query)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"ok={payload['ok']} path={payload['path']}")
        for failure in payload.get("failures", []):
            print(f"- {failure}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
