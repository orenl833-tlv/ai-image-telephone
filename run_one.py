"""Single-painting test of the describe -> render -> guess pipeline via OpenRouter.

Usage: OPENROUTER_API_KEY=... python3 run_one.py
Stdlib only. Outputs go to painting_telephone/out/<painting>/.
"""
import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"
HERE = Path(__file__).parent

PAINTING = {
    "id": "milkmaid",
    "file": HERE / "paintings" / "milkmaid.jpg",
    "answer": "The Milkmaid - Johannes Vermeer",
    # Words the description may not contain (title, artist, museum, style labels).
    "banned": ["milkmaid", "vermeer", "rijksmuseum", "delft", "dutch golden age"],
}
GLOBAL_BANNED = ["painting by", "painted by", "baroque", "renaissance", "impressionis",
                 "famous", "masterpiece", "museum", "century"]

DESCRIBER = "anthropic/claude-sonnet-5"
RENDERER = "google/gemini-3.1-flash-image"
GUESSER = "openai/gpt-5.6-terra"
WORD_CAP = 15
RENDERS_PER_DESCRIPTION = 2


def call(model, content, modalities=None):
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    if modalities:
        body["modalities"] = modalities
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]


def data_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def describe(image_path):
    prompt = (
        f"Describe this image in at most {WORD_CAP} words so that an illustrator who has "
        "never seen it could recreate it. Focus on what is visible: subject, pose, "
        "composition, colors, lighting, objects. Do NOT name the artwork, artist, museum, "
        "art movement, era, or say that it is famous. Output only the description."
    )
    msg = call(DESCRIBER, [{"type": "text", "text": prompt},
                           {"type": "image_url", "image_url": {"url": data_url(image_path)}}])
    return msg["content"].strip()


def leak_check(text):
    """Return list of problems; empty means the description is allowed."""
    problems = []
    lower = text.lower()
    for w in PAINTING["banned"] + GLOBAL_BANNED:
        if w in lower:
            problems.append(f"banned term: {w!r}")
    words = len(re.findall(r"\S+", text))
    if words > WORD_CAP:
        problems.append(f"{words} words > cap {WORD_CAP}")
    return problems


def render(description, out_path):
    msg = call(RENDERER, [{"type": "text", "text": f"Create a painting of: {description}"}],
               modalities=["image", "text"])
    url = msg["images"][0]["image_url"]["url"]
    out_path.write_bytes(base64.b64decode(url.split(",", 1)[1]))


def guess(image_path):
    prompt = (
        "This image is an AI recreation of a famous painting. Which painting is it? "
        'Reply with JSON only: {"guesses": ["Title - Artist", ...]} with your top 5, '
        "most likely first."
    )
    msg = call(GUESSER, [{"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": data_url(image_path)}}])
    raw = msg["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])["guesses"]


def rank_score(guesses, answer_keys):
    """1 for a first-place hit, 1/2 for second, ... 0 if absent."""
    for i, g in enumerate(guesses):
        if any(k in g.lower() for k in answer_keys):
            return 1 / (i + 1), i + 1
    return 0.0, None


def main():
    out = HERE / "out" / PAINTING["id"]
    out.mkdir(parents=True, exist_ok=True)

    description = describe(PAINTING["file"])
    problems = leak_check(description)
    print(f"[describe] {DESCRIBER}: {description!r}")
    if problems:
        print("[leak-check] REJECTED:", "; ".join(problems))
        sys.exit(1)
    print("[leak-check] ok")

    results = []
    for n in range(RENDERS_PER_DESCRIPTION):
        img = out / f"render_{n}.png"
        render(description, img)
        guesses = guess(img)
        score, rank = rank_score(guesses, ["milkmaid", "kitchen maid"])
        print(f"[render {n}] {img.name} -> guesses: {guesses}")
        print(f"[render {n}] score={score:.2f} (rank {rank})")
        results.append({"image": img.name, "guesses": guesses, "score": score, "rank": rank})

    summary = {
        "painting": PAINTING["answer"], "describer": DESCRIBER, "renderer": RENDERER,
        "guesser": GUESSER, "word_cap": WORD_CAP, "description": description,
        "renders": results,
        "mean_score": sum(r["score"] for r in results) / len(results),
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] mean score {summary['mean_score']:.2f} -> {out / 'result.json'}")


if __name__ == "__main__":
    main()
