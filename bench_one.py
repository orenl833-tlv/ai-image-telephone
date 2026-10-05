"""Benchmark several describers on one painting, with two rendering tracks.

For each describer: write a description (word cap, leak check), then
  - image track: a cheap image model paints it,
  - svg track:   a text model draws it as SVG code (can't copy pixels from memory),
and a strong judge grades both against the original.
Finished steps are cached under out/<painting>/bench/, so re-runs only do what's missing.

Usage: OPENROUTER_API_KEY=... python3 bench_one.py <painting_id>
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# eval_one reads the painting id from argv at import time.
PAINTING_ID = sys.argv[1]
sys.argv = sys.argv[:2]
import eval_one as e  # noqa: E402

DESCRIBERS = [
    "anthropic/claude-opus-5.5",
    "anthropic/claude-haiku-4.5",
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
    scores = e.judge(e.PAINTING["file"], generated, description, metrics, note)
    mean = sum(scores[k]["score"] for k in metrics) / len(metrics)
    return {"scores": scores, "mean": round(mean, 2)}


def run_describer(model):
    d = OUT / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    res_path = d / "result.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"describer": model}

    if "description" not in res:
        res["description"] = e.describe(e.PAINTING["file"], model)
        res_path.write_text(json.dumps(res, indent=2))
    desc = res["description"]

    if "image" not in res:
        img = d / "image.png"
        if not img.exists():
            e.render(IMAGE_MODEL, desc, img)
        res["image"] = judge_with(img, desc, e.METRICS)
        res_path.write_text(json.dumps(res, indent=2))

    if "svg" not in res:
        png = d / "svg.png"
        if not png.exists():
            draw_svg(desc, d / "drawing.svg", png)
        note = ("\n(Image 2 is a simple vector drawing made from this text; judge content, "
                "not rendering quality.)")
        res["svg"] = judge_with(png, desc, SVG_METRICS, note)
        res_path.write_text(json.dumps(res, indent=2))

    print(f"[{model}] image {res['image']['mean']}/10  svg {res['svg']['mean']}/10  "
          f"({len(desc.split())} words) {desc!r}", flush=True)
    return res


OUT = e.HERE / "out" / PAINTING_ID / "bench"

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
    rows = sorted(results.values(), key=lambda r: -(r["image"]["mean"] + r["svg"]["mean"]))
    print("\nrank  describer                       image   svg   avg")
    for i, r in enumerate(rows, 1):
        avg = (r["image"]["mean"] + r["svg"]["mean"]) / 2
        print(f"{i:>4}  {r['describer']:30s} {r['image']['mean']:5.1f} {r['svg']['mean']:5.1f} "
              f"{avg:5.2f}")
    print(f"\ncost this run ${e.cost:.3f}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
