"""Combine benchmark results across paintings into one describer ranking.

Usage: python3 aggregate.py painting_id [painting_id ...]
Reads out/<painting>/bench/<describer>/result.json (image track) and prints:
per-painting means, overall mean, per-metric averages, and actual cost per describer.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
METRICS = ["composition", "color_palette", "lighting", "figures", "poses_gestures",
           "facial_expressions", "setting_objects", "style_medium"]
SHORT = ["comp", "color", "light", "figs", "pose", "face", "set", "style"]


def describers():
    # Read the list from bench_one.py's source; importing it would start a benchmark.
    src = (HERE / "bench_one.py").read_text()
    block = src.split("DESCRIBERS = [", 1)[1].split("]", 1)[0]
    return re.findall(r'"([^"]+)"', block)


def main(paintings, bench="bench"):
    DESCRIBERS = describers()
    data = {d: {} for d in DESCRIBERS}
    for p in paintings:
        for d in DESCRIBERS:
            f = HERE / "out" / p / bench / d.replace("/", "_") / "result.json"
            if f.exists():
                r = json.loads(f.read_text())
                if "image" in r:
                    data[d][p] = r

    # Rank only on paintings every describer finished, so a describer that failed
    # on a hard painting can't raise its own average by skipping it.
    common = [p for p in paintings if all(p in data[d] for d in DESCRIBERS)]
    dropped = [p for p in paintings if p not in common]
    if dropped:
        print(f"Excluded from averages (some describer has no result): {', '.join(dropped)}\n")

    def overall(d):
        return sum(data[d][p]["image"]["mean"] for p in common) / len(common) if common else 0

    ranked = sorted(DESCRIBERS, key=lambda d: -overall(d))

    print("Mean score per painting (image track, judge sees images only)")
    print(f"{'describer':30s}" + "".join(f"{p[:9]:>10s}" for p in paintings) + "   OVERALL")
    for d in ranked:
        cells = "".join(f"{data[d][p]['image']['mean']:>10.2f}" if p in data[d] else f"{'fail':>10s}"
                        for p in paintings)
        print(f"{d:30s}{cells}   {overall(d):6.2f}")

    print("\nAverage per metric across paintings")
    print(f"{'describer':30s}" + "".join(f"{s:>7s}" for s in SHORT))
    for d in ranked:
        runs = [data[d][p] for p in common]
        avgs = [sum(r["image"]["scores"][m]["score"] for r in runs) / max(len(runs), 1) for m in METRICS]
        print(f"{d:30s}" + "".join(f"{a:7.2f}" for a in avgs))

    print("\nActual cost per describer, summed over these paintings")
    # judge_image covers both the grading call and the separate memory check.
    print(f"{'describer':30s}{'describe':>10s}{'image':>9s}{'judging':>9s}{'total':>9s}"
          f"{'/painting':>11s}")
    grand = 0
    for d in ranked:
        c = {"description": 0, "image": 0, "judge_image": 0, "total": 0}
        for r in data[d].values():
            for k, v in r.get("cost_usd", {}).items():
                c[k] = c.get(k, 0) + v
        grand += c["total"]
        n = len(data[d]) or 1
        print(f"{d:30s}{c['description']:10.4f}{c['image']:9.4f}{c['judge_image']:9.4f}"
              f"{c['total']:9.3f}{c['total'] / n:11.3f}")
    print(f"{'ALL':30s}{'':28s}{grand:9.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    informed = "--informed" in args
    main([a for a in args if a != "--informed"], "bench_informed" if informed else "bench")
