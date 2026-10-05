"""Image -> text -> image, then score the generated image against the original.

A judge model sees both images and scores 8 fidelity metrics from 0 to 10.
One cheap model describes the painting, each cheap image model renders that same
description once, and a strong judge model grades every render against the original.
Usage: OPENROUTER_API_KEY=... python3 eval_one.py [painting_id] [renderer ...]
Re-running on a painting reuses its saved description and adds the new renderers' results.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"
HERE = Path(__file__).parent

PAINTINGS = {
    "las_meninas": ["meninas", "velazquez", "velázquez", "prado", "infanta", "margarita",
                    "philip", "spanish golden age"],
    "raga_dipak": ["raga dipak", "ragamala", "jaipur", "art institute"],
    "smith_family": ["noah smith", "ezra ames", "art institute"],
    "pearl_earring": ["girl with a pearl", "vermeer", "mauritshuis", "delft", "dutch golden age"],
}
_pid = sys.argv[1] if len(sys.argv) > 1 else "las_meninas"
PAINTING = {"id": _pid, "file": HERE / "paintings" / f"{_pid}.jpg", "banned": PAINTINGS[_pid]}
GLOBAL_BANNED = ["painting by", "painted by", "baroque", "renaissance", "famous",
                 "masterpiece", "museum", "century"]

DESCRIBER = "google/gemini-3.5-flash-lite"
RENDERERS = sys.argv[2:] or ["google/gemini-3.1-flash-lite-image", "openai/gpt-5-image-mini"]
JUDGE = "anthropic/claude-opus-5.5"
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
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{model}: HTTP {e.code}: {e.read().decode()[:500]}") from None
    c = resp.get("usage", {}).get("cost", 0) or 0
    cost += c
    print(f"  [{model}] ${c:.4f}")
    return resp["choices"][0]["message"]


def image_part(path):
    data = path.read_bytes()
    # Trust the bytes, not the extension: some image models return JPEG for a .png name.
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    url = f"data:{mime};base64," + base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def leak_problems(text):
    lower = text.lower()
    problems = [f"banned term: {w!r}" for w in PAINTING["banned"] + GLOBAL_BANNED if w in lower]
    words = len(text.split())
    if words > WORD_CAP:
        problems.append(f"{words} words > cap {WORD_CAP}")
    return problems


def describe(path, model=None):
    prompt = (
        f"Describe this image in at most {WORD_CAP} words so an illustrator who has never seen "
        "it could recreate it as closely as possible: layout, figures, poses, faces, colors, "
        "lighting, setting, style. Do NOT name the artwork, artist, people, museum, art "
        "movement or era. Output only the description."
    )
    feedback = ""
    for _ in range(8):
        text = call(model or DESCRIBER, [{"type": "text", "text": prompt + feedback}, image_part(path)])
        text = text["content"].strip()
        problems = leak_problems(text)
        if not problems:
            return text
        print(f"  rejected {text!r}: {'; '.join(problems)}")
        feedback = f"\n\nPrevious attempt rejected ({'; '.join(problems)}): {text!r}. Fix it."
    raise RuntimeError("describer kept failing the leak check")


def render(model, description, out_path):
    prompt = ("Create exactly one painting as a single image (no panels, borders or frames). "
              "Draw only what the text says. Do not recreate or reference any existing "
              "artwork; invent every detail the text does not specify.\n\n"
              f"Text: {description}")
    msg = call(model, [{"type": "text", "text": prompt}], modalities=["image", "text"])
    url = msg["images"][0]["image_url"]["url"]
    out_path.write_bytes(base64.b64decode(url.split(",", 1)[1]))


def judge(original, generated, description, metrics=None, note=""):
    rubric = "\n".join(f'- "{k}": {v}' for k, v in (metrics or METRICS).items())
    prompt = (
        "Image 1 is an original painting. Image 2 is an attempt to recreate it from a short "
        "text description. Score how closely image 2 matches image 1 on each metric, 0-10 "
        "(10 = indistinguishable, 5 = same idea with clear differences, 0 = unrelated). "
        "Be strict and judge only what is visible.\n\nMetrics:\n" + rubric +
        "\n\nImage 2 was drawn from ONLY this text:\n" + repr(description) + note +
        "\n\nAlso check for copying from memory: list specific details in image 2 that "
        "match image 1 but are NOT stated or clearly implied by the text (an image model that "
        "recognized the artwork would add these). Generic filler that happens to match does "
        "not count.\n\nReply with JSON only: "
        '{"<metric>": {"score": <0-10>, "reason": "<one sentence>"}, ..., '
        '"memory_copy": {"details": ["..."], "score": <0-10, 0 = nothing copied, '
        '10 = clearly reproduced from memory>}}'
    )
    msg = call(JUDGE, [{"type": "text", "text": prompt}, image_part(original),
                       image_part(generated)])
    raw = msg["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def main():
    out = HERE / "out" / PAINTING["id"]
    out.mkdir(parents=True, exist_ok=True)

    log = out / "result.json"
    prev = json.loads(log.read_text()) if log.exists() else {}
    if "description" in prev:
        description = prev["description"]
        print(f"[describe] reusing saved description: {description!r}")
    else:
        print("[describe]")
        description = describe(PAINTING["file"])
        print(f"  {len(description.split())} words: {description!r}")

    results = prev.get("renders", {})
    for model in RENDERERS:
        print(f"[render] {model}")
        generated = out / f"render_{model.replace('/', '_')}.png"
        if not generated.exists():  # reuse a render from an interrupted run
            render(model, description, generated)
        print(f"[judge] {JUDGE}")
        scores = judge(PAINTING["file"], generated, description)
        for k in METRICS:
            print(f"  {k:19s} {scores[k]['score']:>2}/10  {scores[k]['reason']}")
        mean = sum(scores[k]["score"] for k in METRICS) / len(METRICS)
        mc = scores.get("memory_copy", {})
        print(f"  mean {mean:.1f}/10   memory_copy {mc.get('score')}/10: {mc.get('details')}")
        results[model] = {"image": generated.name, "scores": scores, "mean": mean}

    print(f"[done] total cost ${cost:.4f}")
    log.write_text(json.dumps({
        "describer": prev.get("describer", DESCRIBER), "judge": JUDGE, "word_cap": WORD_CAP,
        "description": description, "renders": results,
        "cost_usd": prev.get("cost_usd", 0) + cost,
    }, indent=2))


if __name__ == "__main__":
    main()
