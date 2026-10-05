"""Image -> text -> image, then score the generated image against the original.

A judge model sees both images and scores 8 fidelity metrics from 0 to 10.
Usage: OPENROUTER_API_KEY=... python3 eval_one.py [painting_id] [model]
Passing a model uses it for all three steps (describe, render, judge).
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"
HERE = Path(__file__).parent

PAINTINGS = {
    "las_meninas": ["meninas", "velazquez", "velázquez", "prado", "infanta", "margarita",
                    "philip", "spanish golden age"],
    "pearl_earring": ["pearl earring", "vermeer", "mauritshuis", "delft", "dutch golden age"],
}
_pid = sys.argv[1] if len(sys.argv) > 1 else "las_meninas"
PAINTING = {"id": _pid, "file": HERE / "paintings" / f"{_pid}.jpg", "banned": PAINTINGS[_pid]}
GLOBAL_BANNED = ["painting by", "painted by", "baroque", "renaissance", "famous",
                 "masterpiece", "museum", "century"]

DESCRIBER = "google/gemini-3.5-flash-lite"
RENDERER = "openai/gpt-5-image-mini"
JUDGE = "openai/gpt-5.6-luna"
if len(sys.argv) > 2:
    DESCRIBER = RENDERER = JUDGE = sys.argv[2]
WORD_CAP = 40

METRICS = {
    "composition": "overall layout: where the main masses sit, framing, aspect ratio",
    "color_palette": "dominant colors and their proportions",
    "lighting": "light direction, contrast, light and dark areas",
    "figures": "number of people/animals and how they are arranged relative to each other",
    "poses_gestures": "body positions, where people face, what their hands are doing",
    "facial_expressions": "expressions and gaze direction of faces",
    "setting_objects": "room/landscape and key objects (furniture, props, background items)",
    "style_medium": "painting style, brushwork, medium, level of realism",
}

cost = 0.0


def call(model, content, **extra):
    global cost
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 4000, "usage": {"include": True}, **extra}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    c = resp.get("usage", {}).get("cost", 0) or 0
    cost += c
    print(f"  [{model}] ${c:.4f}")
    return resp["choices"][0]["message"]


def image_part(path):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def leak_problems(text):
    lower = text.lower()
    problems = [f"banned term: {w!r}" for w in PAINTING["banned"] + GLOBAL_BANNED if w in lower]
    words = len(text.split())
    if words > WORD_CAP:
        problems.append(f"{words} words > cap {WORD_CAP}")
    return problems


def describe(path):
    prompt = (
        f"Describe this image in at most {WORD_CAP} words so an illustrator who has never seen "
        "it could recreate it as closely as possible: layout, figures, poses, faces, colors, "
        "lighting, setting, style. Do NOT name the artwork, artist, people, museum, art "
        "movement or era. Output only the description."
    )
    feedback = ""
    for _ in range(4):
        text = call(DESCRIBER, [{"type": "text", "text": prompt + feedback}, image_part(path)])
        text = text["content"].strip()
        problems = leak_problems(text)
        if not problems:
            return text
        print(f"  rejected {text!r}: {'; '.join(problems)}")
        feedback = f"\n\nPrevious attempt rejected ({'; '.join(problems)}): {text!r}. Fix it."
    raise RuntimeError("describer kept failing the leak check")


def render(description, out_path):
    msg = call(RENDERER, [{"type": "text", "text": f"Create a painting of: {description}"}],
               modalities=["image", "text"])
    url = msg["images"][0]["image_url"]["url"]
    out_path.write_bytes(base64.b64decode(url.split(",", 1)[1]))


def judge(original, generated):
    rubric = "\n".join(f'- "{k}": {v}' for k, v in METRICS.items())
    prompt = (
        "Image 1 is an original painting. Image 2 is an attempt to recreate it from a short "
        "text description. Score how closely image 2 matches image 1 on each metric, 0-10 "
        "(10 = indistinguishable, 5 = same idea with clear differences, 0 = unrelated). "
        "Be strict and judge only what is visible.\n\nMetrics:\n" + rubric +
        '\n\nReply with JSON only: {"<metric>": {"score": <0-10>, "reason": "<one sentence>"}, ...}'
    )
    msg = call(JUDGE, [{"type": "text", "text": prompt}, image_part(original),
                       image_part(generated)])
    raw = msg["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def main():
    out = HERE / "out" / PAINTING["id"] / RENDERER.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)

    print("[describe]")
    description = describe(PAINTING["file"])
    print(f"  {len(description.split())} words: {description!r}")

    print("[render]")
    generated = out / "render.png"
    render(description, generated)

    print("[judge]")
    scores = judge(PAINTING["file"], generated)
    for k in METRICS:
        print(f"  {k:19s} {scores[k]['score']:>2}/10  {scores[k]['reason']}")
    mean = sum(scores[k]["score"] for k in METRICS) / len(METRICS)
    print(f"[done] mean {mean:.1f}/10, total cost ${cost:.4f}")

    (out / "result.json").write_text(json.dumps({
        "describer": DESCRIBER, "renderer": RENDERER, "judge": JUDGE, "word_cap": WORD_CAP,
        "description": description, "scores": scores, "mean": mean, "cost_usd": cost,
    }, indent=2))


if __name__ == "__main__":
    main()
