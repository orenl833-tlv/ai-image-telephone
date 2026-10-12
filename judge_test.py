"""Compare candidate judges on the same generated videos.

Judges:
  opus6   - Claude Opus 5.5 on 6-frame grids + audio notes (current setup)
  opus12  - Claude Opus 5.5 on 12-frame grids + audio notes
  gemini  - Gemini 3.1 Pro watching and hearing both videos
  mimo    - MiMo v2.6 Pro watching and hearing both videos

Every judge gets the same metrics and must list the clear differences it sees, so we can
check which judges catch the problems we already know about.

Usage: OPENROUTER_API_KEY=... python3 judge_test.py <clip_dir>
"""
import base64
import json
import sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
import video_sound as vs  # noqa: E402

vp, e = vs.vp, vs.e
METRICS = {k: v for k, v in vs.METRICS.items() if k != "music"}  # this clip has no music
METRICS["sound_effects"] += ("; the original has NO music, so any music or extra sounds it "
                             "lacks count against the recreation")
RUBRIC = "\n".join(f'- "{k}": {v}' for k, v in METRICS.items())
REPLY = ('Reply with JSON only: {"original": "<what happens and what is heard, 2-3 sentences>", '
         '"recreation": "<same for the recreation>", "differences": ["<each clear difference>"], '
         '"<metric>": {"score": <0-10>, "reason": "<one sentence>"}, ...}')
TASK = (
    "Score how closely the recreation matches the original on each metric, 0-10 "
    "(10 = indistinguishable, 5 = same idea with clear differences, 0 = unrelated or missing). "
    "First describe both clips (who is where in the frame, what they do and in what order, "
    "camera, what is said and heard), then list the clear differences, then score. Judge "
    "timing loosely but order of events strictly. Be strict.\n\nMetrics:\n" + RUBRIC + "\n\n" + REPLY
)


def parse(raw):
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def judge_frames(orig_grid, gen_grid, orig_notes, gen_notes, n):
    prompt = (f"Image 1 shows {n} frames of an original movie clip; image 2 shows {n} frames, "
              "at the same relative times, of an attempt to recreate it from a short text "
              "description. Frames run left to right, top to bottom, each labeled with number "
              f"and time.\n\nOriginal soundtrack notes: {orig_notes}\n\nRecreation soundtrack "
              f"notes: {gen_notes}\n\n" + TASK)
    return parse(e.call("anthropic/claude-opus-5.5", [{"type": "text", "text": prompt},
                        e.image_part(orig_grid), e.image_part(gen_grid)])["content"])


def video_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}}


def judge_native(model, orig_mp4, gen_mp4):
    prompt = ("Video 1 is an original movie clip (picture and sound). Video 2 is an attempt to "
              "recreate it from a short text description. Watch and listen to both.\n\n" + TASK)
    return parse(e.call(model, [{"type": "text", "text": prompt}, video_part(orig_mp4),
                                video_part(gen_mp4)], max_tokens=6000)["content"])


def main():
    d = Path(ARGS[0])
    out = d / "judge_test"
    out.mkdir(exist_ok=True)
    orig = d / "original.mp4"
    orig_notes = vs.audio_notes(orig, out / "original_mimo_notes.txt")
    vp.N_FRAMES = 12
    vp.frame_grid(orig, out / "original_grid12.png")
    vp.N_FRAMES = 6
    results = {}
    for rec in sorted(p for p in d.iterdir() if (p / "video.mp4").exists()):
        gen = rec / "video.mp4"
        gen_notes = vs.audio_notes(gen, out / f"{rec.name}_mimo_notes.txt")
        vp.N_FRAMES = 12
        vp.frame_grid(gen, out / f"{rec.name}_grid12.png")
        vp.N_FRAMES = 6
        res = {}
        for name, fn in [
            ("opus6", lambda: judge_frames(d / "original_grid.png", rec / "grid.png",
                                           orig_notes, gen_notes, 6)),
            ("opus12", lambda: judge_frames(out / "original_grid12.png",
                                            out / f"{rec.name}_grid12.png", orig_notes, gen_notes, 12)),
            ("gemini", lambda: judge_native("google/gemini-3.1-pro-preview", orig, gen)),
            ("mimo", lambda: judge_native("xiaomi/mimo-v2.6-pro", orig, gen)),
        ]:
            cache = out / f"{rec.name}__{name}.json"
            if cache.exists():
                res[name] = json.loads(cache.read_text())
                continue
            t = e.cost
            try:
                r = fn()
            except Exception as ex:  # one failing judge shouldn't stop the comparison
                r = {"error": str(ex)}
            r["cost"] = round(e.cost - t, 4)
            cache.write_text(json.dumps(r, indent=2))
            res[name] = r
        results[rec.name] = res
        print(f"\n### {rec.name}")
        for name, r in res.items():
            if "error" in r:
                print(f"  {name:7s} ERROR {r['error'][:200]}")
                continue
            sc = [r[k]["score"] for k in METRICS if isinstance(r.get(k), dict)]
            print(f"  {name:7s} mean {sum(sc) / len(sc):.2f}  ${r['cost']}  diffs: {r.get('differences')}")
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\ntotal cost ${e.cost:.3f}")


if __name__ == "__main__":
    main()
