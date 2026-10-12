"""Video pilot: clip -> frames -> description -> generated clip -> frame-by-frame grade.

Every describer sees the same 6 frames (tiled into one grid image), so models that
can't watch video are on equal footing with those that can. The judge also sees
frame grids only: 6 frames of the original above 6 frames of the generated clip.

Usage: OPENROUTER_API_KEY=... python3 video_pilot.py <clip.mp4> <start_s> <duration_s>
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "fight_club"]  # eval_one reads a painting id from argv at import
import eval_one as e  # noqa: E402  (reuses call(), image_part(), cost tracking)

FF = imageio_ffmpeg.get_ffmpeg_exe()
DESCRIBERS = ["x-ai/grok-4.7", "google/gemini-3.8-flash"]
VIDEO_MODEL = "x-ai/grok-imagine-video-1.5-lite"
JUDGE = "anthropic/claude-opus-5.5"
N_FRAMES = 6
WORD_CAP = 60
OUT = e.HERE / "out" / "video_pilot"

METRICS = {
    "composition": "framing, camera distance and where things sit in the frame",
    "color_palette": "dominant colors and their proportions",
    "lighting": "light, contrast, time of day",
    "figures": "number of people/animals and their arrangement",
    "setting_objects": "location and key objects",
    "style_medium": "look of the footage: film stock, grain, realism, color cast",
    "action": "what the people do (walking, carrying, turning...)",
    "motion_camera": "direction and speed of movement; camera static or moving",
    "event_order": "whether things happen in the same order across the frames",
}


def ff(*args):
    subprocess.run([FF, "-loglevel", "error", "-y", *args], check=True)


def frame_grid(video, out_png, duration):
    """Take N_FRAMES evenly spaced frames and tile them 3x2, in time order."""
    fps = N_FRAMES / duration
    ff("-i", str(video), "-vf", f"fps={fps:.4f},scale=426:-2,tile=3x2:padding=4",
       "-frames:v", "1", str(out_png))


def describe(model, grid, duration):
    prompt = (
        f"These are {N_FRAMES} frames from a {duration:.0f}-second silent video clip, in time "
        "order left to right, top row then bottom row. Describe the clip in at most "
        f"{WORD_CAP} words so a video maker who never saw it could recreate it as closely as "
        "possible. It will be compared with the original on: composition, color, lighting, "
        "figures, setting, style of the footage, the action, movement and camera motion, and "
        "the order of events. State the orientation (landscape/portrait). Describe what "
        "happens over time. Do NOT name places, people, dates or events. Output only the "
        "description."
    )
    feedback = ""
    for _ in range(6):
        text = e.call(model, [{"type": "text", "text": prompt + feedback},
                              e.image_part(grid)])["content"].strip()
        n = len(text.split())
        if 20 <= n <= WORD_CAP:
            return text
        feedback = f"\n\nPrevious attempt had {n} words ({text!r}); stay within {WORD_CAP}."
    raise RuntimeError(f"{model} kept missing the word cap")


def generate_video(prompt, duration, out_mp4):
    key = e.os.environ["OPENROUTER_API_KEY"]
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def req(url, body=None):
        r = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                   headers=hdr, method="POST" if body else "GET")
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.read()

    job = json.loads(req("https://openrouter.ai/api/v1/videos", {
        "model": VIDEO_MODEL, "prompt": prompt, "duration": int(round(duration)),
        "resolution": "480p", "aspect_ratio": "16:9", "generate_audio": False}))
    while True:
        time.sleep(15)
        st = json.loads(req(job["polling_url"]))
        if st["status"] == "completed":
            break
        if st["status"] in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"video job {st['status']}: {st.get('error')}")
    out_mp4.write_bytes(req(f"https://openrouter.ai/api/v1/videos/{job['id']}/content?index=0"))
    c = (st.get("usage") or {}).get("cost") or 0
    with e._lock:
        e.cost += c
    return c


def judge(orig_grid, gen_grid):
    rubric = "\n".join(f'- "{k}": {v}' for k, v in METRICS.items())
    prompt = (
        f"Image 1 shows {N_FRAMES} frames of an original silent video clip; image 2 shows "
        f"{N_FRAMES} frames, taken at the same relative times, of an attempt to recreate it. "
        "Frames run left to right, top row then bottom row. Score how closely the recreation "
        "matches the original on each metric, 0-10 (10 = indistinguishable, 5 = same idea "
        "with clear differences, 0 = unrelated). Judge timing loosely (same action a moment "
        "later is fine) but order of events strictly. Be strict and judge only what is "
        "visible.\n\nMetrics:\n" + rubric +
        '\n\nReply with JSON only: {"<metric>": {"score": <0-10>, "reason": "<one sentence>"}, ...}'
    )
    raw = e.call(JUDGE, [{"type": "text", "text": prompt}, e.image_part(orig_grid),
                         e.image_part(gen_grid)])["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def run_describer(model, orig_grid, duration):
    d = OUT / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    res = {"describer": model, "cost_usd": {}}
    t = e.cost
    res["description"] = describe(model, orig_grid, duration)
    res["cost_usd"]["description"] = round(e.cost - t, 5)
    res["cost_usd"]["video"] = round(generate_video(res["description"], duration,
                                                    d / "video.mp4"), 5)
    frame_grid(d / "video.mp4", d / "grid.png", duration)
    t = e.cost
    res["scores"] = judge(orig_grid, d / "grid.png")
    res["cost_usd"]["judge"] = round(e.cost - t, 5)
    res["mean"] = round(sum(res["scores"][k]["score"] for k in METRICS) / len(METRICS), 2)
    (d / "result.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    src, start, duration = Path(ARGS[0]), float(ARGS[1]), float(ARGS[2])
    OUT.mkdir(parents=True, exist_ok=True)
    clip = OUT / "original.mp4"
    ff("-ss", str(start), "-t", str(duration), "-i", str(src), "-an", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", str(clip))
    orig_grid = OUT / "original_grid.png"
    frame_grid(clip, orig_grid, duration)
    # Costs are attributed per describer from e.cost deltas, so run them one at a time.
    results = [run_describer(m, orig_grid, duration) for m in DESCRIBERS]
    for r in sorted(results, key=lambda r: -r["mean"]):
        print(f"\n{r['describer']}  mean {r['mean']}/10  cost {r['cost_usd']}")
        print("  desc:", r["description"])
        for k in METRICS:
            print(f"  {k:16s} {r['scores'][k]['score']:>2}  {r['scores'][k]['reason']}")
    print(f"\ntotal cost ${e.cost:.3f}")


if __name__ == "__main__":
    main()
