"""Video pilot: clip -> frames -> description -> generated clip -> frame-by-frame grade.

Every describer sees the same 6 frames (tiled into one grid image, each labeled with
its time), so models that can't watch video are on equal footing with those that can.
The judge also sees frame grids only, taken at the same relative times in both clips,
and must write down what happens in each clip before it scores.

Black bars (letterbox/pillarbox) are cropped from the original first, and the video
model is told the original's aspect ratio and length.

Usage: OPENROUTER_API_KEY=... python3 video_pilot.py <clip.mp4> <start_s> <duration_s> [--rejudge]
--rejudge keeps existing descriptions and generated videos and only redoes the grids
and the grading (no new video spend).
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "fight_club"]  # eval_one reads a painting id from argv at import
import eval_one as e  # noqa: E402  (reuses call(), image_part(), cost tracking)

FF = imageio_ffmpeg.get_ffmpeg_exe()
DESCRIBERS = ["x-ai/grok-4.7", "google/gemini-3.8-flash"]
VIDEO_MODEL = "x-ai/grok-imagine-video-1.5-lite"
VIDEO_ASPECTS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"]  # what VIDEO_MODEL accepts
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
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


def probe(video):
    """(duration_s, width, height) from ffmpeg's header dump."""
    info = subprocess.run([FF, "-i", str(video)], capture_output=True, text=True).stderr
    h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    w, ht = map(int, re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info).groups())
    return int(h) * 3600 + int(m) * 60 + float(sec), w, ht


def detect_crop(src, start, duration):
    """ffmpeg's cropdetect over the segment; returns 'w:h:x:y' for the picture area."""
    info = subprocess.run([FF, "-ss", str(start), "-t", str(duration), "-i", str(src),
                           "-vf", "cropdetect=limit=24:round=2", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", info)
    return max(set(crops), key=crops.count) if crops else None


def nearest_aspect(w, h):
    def val(a):
        x, y = map(int, a.split(":"))
        return x / y
    return min(VIDEO_ASPECTS, key=lambda a: abs(val(a) - w / h))


def frame_grid(video, out_png):
    """N_FRAMES frames at the same relative times in any clip, tiled in time order (3x2 for 6),
    each labeled with its frame number and time so the order can't be misread."""
    duration, _, _ = probe(video)
    times = [(i + 0.5) * duration / N_FRAMES for i in range(N_FRAMES)]
    tiles = []
    for i, t in enumerate(times):
        f = out_png.with_name(f"_frame{i}.png")
        ff("-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=426:-2", str(f))
        tiles.append((Image.open(f).convert("RGB"), t))
        f.unlink()
    tw, th = tiles[0][0].size
    pad = 4
    cols = 3 if N_FRAMES <= 6 else 4
    rows = -(-N_FRAMES // cols)
    grid = Image.new("RGB", (cols * tw + (cols - 1) * pad, rows * th + (rows - 1) * pad), "black")
    font = ImageFont.truetype(FONT, 18)
    for i, (img, t) in enumerate(tiles):
        img = img.resize((tw, th))
        d = ImageDraw.Draw(img)
        label = f"{i + 1} | {t:.1f}s"
        x1, y1, x2, y2 = d.textbbox((8, 6), label, font=font)
        d.rectangle((x1 - 4, y1 - 3, x2 + 4, y2 + 3), fill=(0, 0, 0))
        d.text((8, 6), label, font=font, fill=(255, 255, 255))
        grid.paste(img, ((i % cols) * (tw + pad), (i // cols) * (th + pad)))
    grid.save(out_png)


def describe(model, grid, duration):
    prompt = (
        f"These are {N_FRAMES} frames from a {duration:.0f}-second silent video clip, in time "
        "order left to right, top row then bottom row; each is labeled with its number and "
        "time. Describe the clip in at most "
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


def generate_video(prompt, duration, aspect, out_mp4):
    key = e.os.environ["OPENROUTER_API_KEY"]
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def req(url, body=None):
        r = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                   headers=hdr, method="POST" if body else "GET")
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.read()

    job = json.loads(req("https://openrouter.ai/api/v1/videos", {
        "model": VIDEO_MODEL, "prompt": prompt, "duration": int(round(duration)),
        "resolution": "480p", "aspect_ratio": aspect, "generate_audio": False}))
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


EVENTS_INSTRUCTIONS = (
    "who is there, which direction they move in screen terms (toward the left or right edge "
    "of the frame; compare positions across the numbered frames), what they do and in what "
    "order, and whether the camera moves"
)


def original_events(orig_grid):
    """The judge's account of the original, written once and reused for every grade so
    all recreations are compared against the same reading of the original."""
    cache = OUT / "original_events.txt"
    if cache.exists():
        return cache.read_text()
    prompt = (
        f"These are {N_FRAMES} frames of a silent video clip, left to right, top row then "
        "bottom row, each labeled with its number and time. In 3-4 sentences, write down "
        f"what happens: {EVENTS_INSTRUCTIONS}. Output only the account."
    )
    text = e.call(JUDGE, [{"type": "text", "text": prompt},
                          e.image_part(orig_grid)])["content"].strip()
    cache.write_text(text)
    return text


def judge(orig_grid, gen_grid):
    rubric = "\n".join(f'- "{k}": {v}' for k, v in METRICS.items())
    prompt = (
        f"Image 1 shows {N_FRAMES} frames of an original silent video clip; image 2 shows "
        f"{N_FRAMES} frames, taken at the same relative times, of an attempt to recreate it. "
        "Frames run left to right, top row then bottom row, and each is labeled with its "
        "number and time.\n\nWhat happens in the original (established earlier, treat as "
        f"correct): {original_events(orig_grid)}\n\nFirst write down what happens in the "
        f"recreation: {EVENTS_INSTRUCTIONS}. Then score how closely the recreation matches "
        "the original on each metric, 0-10 (10 = indistinguishable, 5 = same idea with clear "
        "differences, 0 = unrelated), basing the scores on the two accounts and the frames. "
        "Judge timing loosely (same action a moment later is fine) but order of events "
        "strictly. Be strict and judge only what is visible.\n\nMetrics:\n" + rubric +
        '\n\nReply with JSON only: {"recreation_events": "<2-3 sentences>", '
        '"<metric>": {"score": <0-10>, "reason": "<one sentence>"}, ...}'
    )
    raw = e.call(JUDGE, [{"type": "text", "text": prompt}, e.image_part(orig_grid),
                         e.image_part(gen_grid)])["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def run_describer(model, orig_grid, duration, aspect, rejudge=False):
    d = OUT / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    res_path = d / "result.json"
    if rejudge:
        res = json.loads(res_path.read_text())
    else:
        res = {"describer": model, "cost_usd": {}}
        t = e.cost
        res["description"] = describe(model, orig_grid, duration)
        res["cost_usd"]["description"] = round(e.cost - t, 5)
        res["cost_usd"]["video"] = round(generate_video(res["description"], duration, aspect,
                                                        d / "video.mp4"), 5)
    frame_grid(d / "video.mp4", d / "grid.png")
    t = e.cost
    res["scores"] = judge(orig_grid, d / "grid.png")
    res["cost_usd"]["judge"] = round(e.cost - t, 5)
    res["mean"] = round(sum(res["scores"][k]["score"] for k in METRICS) / len(METRICS), 2)
    res_path.write_text(json.dumps(res, indent=2))
    return res


def main():
    rejudge = "--rejudge" in ARGS
    args = [a for a in ARGS if a != "--rejudge"]
    src, start, duration = Path(args[0]), float(args[1]), float(args[2])
    OUT.mkdir(parents=True, exist_ok=True)
    clip = OUT / "original.mp4"
    crop = detect_crop(src, start, duration)
    vf = [f"crop={crop}"] if crop else []
    ff("-ss", str(start), "-t", str(duration), "-i", str(src), "-an",
       *(["-vf", ",".join(vf)] if vf else []), "-c:v", "libx264", "-pix_fmt", "yuv420p",
       str(clip))
    _, w, h = probe(clip)
    aspect = nearest_aspect(w, h)
    print(f"original: crop {crop}, {w}x{h}, sent to video model as {aspect}, {duration:.0f}s")
    orig_grid = OUT / "original_grid.png"
    frame_grid(clip, orig_grid)
    (OUT / "original_events.txt").unlink(missing_ok=True)  # new grid, so re-establish it
    print("judge's account of the original:", original_events(orig_grid))
    # Costs are attributed per describer from e.cost deltas, so run them one at a time.
    results = [run_describer(m, orig_grid, duration, aspect, rejudge) for m in DESCRIBERS]
    for r in sorted(results, key=lambda r: -r["mean"]):
        print(f"\n{r['describer']}  mean {r['mean']}/10  cost {r['cost_usd']}")
        print("  desc:", r["description"])
        print("  judge saw (recreation):", r["scores"].get("recreation_events"))
        for k in METRICS:
            print(f"  {k:16s} {r['scores'][k]['score']:>2}  {r['scores'][k]['reason']}")
    print(f"\ntotal cost ${e.cost:.3f}")


if __name__ == "__main__":
    main()
