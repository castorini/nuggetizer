#!/usr/bin/env python3
"""Compare two nuggetizer metrics runs side-by-side.

Usage:
    python3 compare.py --run-a metrics_a.jsonl --run-b metrics_b.jsonl [--threshold 0.1]

Produces a table of per-query scores and global averages, highlighting
queries where the two runs diverge by more than --threshold.
"""

import argparse
import json
import sys


def load_metrics(path: str) -> dict[str, dict]:
    """Load metrics JSONL into a dict keyed by qid."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("qid", ""))
            records[qid] = rec
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two nuggetizer metrics runs")
    parser.add_argument("--run-a", required=True, help="First metrics JSONL file")
    parser.add_argument("--run-b", required=True, help="Second metrics JSONL file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Highlight queries with score difference > threshold (default: 0.1)",
    )
    args = parser.parse_args()

    a = load_metrics(args.run_a)
    b = load_metrics(args.run_b)

    all_qids = sorted(set(a.keys()) | set(b.keys()))
    if not all_qids:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    score_fields = ["strict_vital_score", "strict_all_score", "vital_score", "all_score"]

    # Header
    print(f"{'qid':<12}", end="")
    for f in score_fields:
        short = f.replace("strict_", "s_").replace("_score", "")
        print(f"  {short}_A  {short}_B  {'diff':>5}", end="")
    print("  flag")
    print("-" * (12 + len(score_fields) * 20 + 6))

    # Per-query rows
    sums_a = {f: 0.0 for f in score_fields}
    sums_b = {f: 0.0 for f in score_fields}
    count = 0
    flagged = 0

    for qid in all_qids:
        ra = a.get(qid, {})
        rb = b.get(qid, {})
        print(f"{qid:<12}", end="")
        max_diff = 0.0
        for f in score_fields:
            va = ra.get(f, float("nan"))
            vb = rb.get(f, float("nan"))
            diff = abs(va - vb) if va == va and vb == vb else float("nan")
            max_diff = max(max_diff, diff) if diff == diff else max_diff
            short = f.replace("strict_", "s_").replace("_score", "")
            print(f"  {va:6.3f}  {vb:6.3f}  {diff:+.3f}", end="")
            if va == va:
                sums_a[f] += va
            if vb == vb:
                sums_b[f] += vb
        flag = " ***" if max_diff > args.threshold else ""
        if flag:
            flagged += 1
        print(f"  {flag}")
        count += 1

    # Averages
    print("-" * (12 + len(score_fields) * 20 + 6))
    print(f"{'AVG':<12}", end="")
    for f in score_fields:
        va = sums_a[f] / count if count > 0 else 0
        vb = sums_b[f] / count if count > 0 else 0
        diff = va - vb
        short = f.replace("strict_", "s_").replace("_score", "")
        print(f"  {va:6.3f}  {vb:6.3f}  {diff:+.3f}", end="")
    print()

    # Summary
    print(f"\nQueries: {count} total, {flagged} flagged (diff > {args.threshold})")
    print(f"Run A only: {len(set(a.keys()) - set(b.keys()))}")
    print(f"Run B only: {len(set(b.keys()) - set(a.keys()))}")


if __name__ == "__main__":
    main()
