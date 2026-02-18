"""Generate a markdown report showing top, middle, and bottom LLS samples."""

import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_messages(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        parts.append(f"**{role}:** {content}")
    return "\n\n".join(parts)


def render_sample_table(samples: list[dict], label: str) -> str:
    lines = [f"### {label}\n"]
    for i, sample in enumerate(samples, 1):
        lls = sample["lls"]
        msgs = format_messages(sample["messages"])
        lines.append(f"#### {i}. LLS = `{lls:.4f}`\n")
        lines.append(f"{msgs}\n")
        lines.append("---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the JSONL file with LLS scores")
    parser.add_argument(
        "-o", "--output", default="reports/lls_report.md", help="Output markdown path"
    )
    parser.add_argument("-k", type=int, default=5, help="Number of samples per group")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    sorted_records = sorted(records, key=lambda r: r["lls"])
    n = len(sorted_records)
    k = args.k

    bottom = sorted_records[:k]
    mid_start = (n // 2) - (k // 2)
    middle = sorted_records[mid_start : mid_start + k]
    top = sorted_records[-k:][::-1]

    lls_values = [r["lls"] for r in sorted_records]
    mean_lls = statistics.mean(lls_values)
    median_lls = statistics.median(lls_values)
    stdev_lls = statistics.stdev(lls_values)
    min_lls = lls_values[0]
    max_lls = lls_values[-1]

    input_path = Path(args.input)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# LLS Report: `{input_path.name}`\n")
    md.append(f"**Generated:** {now}\n")
    md.append(f"**Source:** `{args.input}`\n")
    md.append(f"**Total samples:** {n:,}\n")

    md.append("## Summary Statistics\n")
    md.append(f"| Statistic | Value |")
    md.append(f"|-----------|-------|")
    md.append(f"| Mean | {mean_lls:.4f} |")
    md.append(f"| Median | {median_lls:.4f} |")
    md.append(f"| Std Dev | {stdev_lls:.4f} |")
    md.append(f"| Min | {min_lls:.4f} |")
    md.append(f"| Max | {max_lls:.4f} |")
    md.append("")

    md.append(render_sample_table(top, f"Top {k} (Highest LLS)"))
    md.append(render_sample_table(middle, f"Middle {k} (Median Region)"))
    md.append(render_sample_table(bottom, f"Bottom {k} (Lowest LLS)"))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
