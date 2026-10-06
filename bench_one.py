"""Benchmark several describers on one painting, with two rendering tracks.

For each describer: write a description (word cap, leak check), then
  - image track: a cheap image model paints it,
  - svg track:   a text model draws it as SVG code (can't copy pixels from memory),
and a strong judge grades both against the original.
Finished steps are cached under out/<painting>/bench/, so re-runs only do what's missing.

Usage: OPENROUTER_API_KEY=... python3 bench_one.py <painting_id> [--image-only] [--informed]
--informed tells describers the grading criteria; results go to out/<painting>/bench_informed/.
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# eval_one reads the painting id from argv at import time.
PAINTING_ID = sys.argv[1]
IMAGE_ONLY = "--image-only" in sys.argv
INFORMED = "--informed" in sys.argv
sys.argv = sys.argv[:2]
import eval_one as e  # noqa: E402

DESCRIBERS = [
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
    "google/gemini-3.8-flash",
    "google/gemini-3.5-flash-lite",
    "qwen/qwen3-vl-8b-instruct",
    "x-ai/grok-4.7",
]
IMAGE_MODEL = "google/gemini-3.1-flash-lite-image"
SVG_MODEL = "anthropic/claude-sonnet-5"
CHROMIUM = "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell"
SVG_METRICS = {k: v for k, v in e.METRICS.items() if k != "style_medium"}


def draw_svg(description, svg_path, png_path):
    prompt = (
        "Write one self-contained SVG drawing of the scene below. Use a viewBox and set "
        "width and height attributes (longest side 800). Draw only what the text says; "
        "invent every detail it does not specify, and do not recreate any existing artwork. "
        "Output only the SVG code.\n\nText: " + description
    )
    code = e.call(SVG_MODEL, [{"type": "text", "text": prompt}], max_tokens=16000)["content"]
    code = code[code.index("<svg"): code.rindex("</svg>") + 6]
    svg_path.write_text(code)
    w = re.search(r'\bwidth="(\d+)', code)
    h = re.search(r'\bheight="(\d+)', code)
    size = f"{w.group(1) if w else 800},{h.group(1) if h else 800}"
    subprocess.run([CHROMIUM, "--no-sandbox", "--hide-scrollbars", f"--window-size={size}",
                    f"--screenshot={png_path}", svg_path.resolve().as_uri()],
                   check=True, capture_output=True)


def judge_with(generated, description, metrics, note=""):
    scores = e.judge(e.PAINTING["file"], generated, metrics, note)
    scores["memory_copy"] = e.memory_check(e.PAINTING["file"], generated, description)
    mean = sum(scores[k]["score"] for k in metrics) / len(metrics)
    return {"scores": scores, "mean": round(mean, 2)}


def spent():
    return getattr(e.thread_cost, "total", 0.0)


def run_describer(model):
    d = OUT / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    res_path = d / "result.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"describer": model}
    costs = res.setdefault("cost_usd", {})  # actual spend per step, from OpenRouter

    if "description" not in res:
        t = spent()
        res["description"] = e.describe(e.PAINTING["file"], model, INFORMED)
        costs["description"] = round(spent() - t, 5)
        res_path.write_text(json.dumps(res, indent=2))
    desc = res["description"]

    if "image" not in res:
        img = d / "image.png"
        if not img.exists():
            t = spent()
            e.render(IMAGE_MODEL, desc, img)
            costs["image"] = round(spent() - t, 5)
        t = spent()
        res["image"] = judge_with(img, desc, e.METRICS)
        costs["judge_image"] = round(spent() - t, 5)
        res_path.write_text(json.dumps(res, indent=2))

    if "svg" not in res and not IMAGE_ONLY:
        png = d / "svg.png"
        if not png.exists():
            t = spent()
            draw_svg(desc, d / "drawing.svg", png)
            costs["svg"] = round(spent() - t, 5)
        note = (" Image 2 is a simple vector drawing; judge content, not rendering quality.")
        t = spent()
        res["svg"] = judge_with(png, desc, SVG_METRICS, note)
        costs["judge_svg"] = round(spent() - t, 5)
        res_path.write_text(json.dumps(res, indent=2))

    costs["total"] = round(sum(v for k, v in costs.items() if k != "total"), 5)
    res_path.write_text(json.dumps(res, indent=2))
    print(f"[{model}] ${costs['total']:.3f}  image {res['image']['mean']}/10  svg {res.get('svg', {}).get('mean', '-')}/10  "
          f"({len(desc.split())} words) {desc!r}", flush=True)
    return res


OUT = e.HERE / "out" / PAINTING_ID / ("bench_informed" if INFORMED else "bench")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {m: pool.submit(run_describer, m) for m in DESCRIBERS}
    results = {}
    for m, f in futures.items():
        try:
            results[m] = f.result()
        except Exception as ex:  # one failing describer shouldn't sink the rest
            print(f"[{m}] FAILED: {ex}")
    def avg(r):
        means = [r[t]["mean"] for t in ("image", "svg") if t in r]
        return sum(means) / len(means)

    rows = sorted(results.values(), key=lambda r: -avg(r))
    print("\nrank  describer                       image   svg   avg    cost")
    for i, r in enumerate(rows, 1):
        svg = f"{r['svg']['mean']:5.1f}" if "svg" in r else "    -"
        print(f"{i:>4}  {r['describer']:30s} {r['image']['mean']:5.1f} {svg} "
              f"{avg(r):5.2f}  ${r['cost_usd'].get('total', 0):.3f}")
    print(f"\ncost this run ${e.cost:.3f}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
