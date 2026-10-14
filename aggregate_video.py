"""Combine movie-clip results (video_sound.py) into one describer ranking.

Usage: python3 aggregate_video.py
Reads out/video_sound/<clip>/*/result.json for every clip not in EXCLUDED and prints the
overall score per clip, the mean over clips, and the mean picture and sound scores.
A describer whose video failed on a clip (e.g. blocked by the video model's filter) is
averaged over the clips it has.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent / "out" / "video_sound"
EXCLUDED = {
    "sixgun_siblings": "black-and-white pilot clip, run before the current pipeline",
    "fight_club": "4 of 6 describers recognized the film",
}


def main():
    clips = sorted(d.name for d in ROOT.iterdir()
                   if d.is_dir() and d.name not in EXCLUDED and any(d.glob("*/result.json")))
    data = {}
    for c in clips:
        for f in (ROOT / c).glob("*/result.json"):
            r = json.loads(f.read_text())
            if "mean" in r:
                data.setdefault(r["describer"], {})[c] = r

    def avg(d, key):
        runs = data[d].values()
        return sum(r[key] for r in runs) / len(runs)

    ranked = sorted(data, key=lambda d: -avg(d, "mean"))
    print("Overall score per clip (picture and sound, weighted per clip)")
    print(f"{'describer':30s}" + "".join(f"{c[:11]:>12s}" for c in clips)
          + f"{'MEAN':>8s}{'picture':>9s}{'sound':>7s}")
    for d in ranked:
        cells = "".join(f"{data[d][c]['mean']:>12.2f}" if c in data[d] else f"{'-':>12s}"
                        for c in clips)
        print(f"{d:30s}{cells}{avg(d, 'mean'):8.2f}{avg(d, 'mean_picture'):9.2f}"
              f"{avg(d, 'mean_sound'):7.2f}")
    print("\nExcluded: " + "; ".join(f"{k} ({v})" for k, v in EXCLUDED.items()))


if __name__ == "__main__":
    main()
