"""Chain test: describe -> render -> guess, then describe the generated image, and repeat.

Each hop describes the first image from the previous hop (hop 1 describes the
original painting), renders 2 images, and asks the guesser about both.
Stops when the top guess is wrong for both images, or at MAX_HOPS.

Usage: OPENROUTER_API_KEY=... python3 chain.py
"""
import json

from run_one import (HERE, PAINTING, RENDERS_PER_DESCRIPTION, describe, guess,
                     leak_check, rank_score, render)

MAX_HOPS = 12
ANSWER_KEYS = ["milkmaid", "kitchen maid"]


def describe_clean(image_path, attempts=5):
    feedback = ""
    for _ in range(attempts):
        text = describe(image_path, feedback)
        problems = leak_check(text)
        if not problems:
            return text
        print(f"  [leak-check] rejected {text!r}: {'; '.join(problems)}")
        feedback = (f"\n\nYour previous attempt was rejected ({'; '.join(problems)}): "
                    f"{text!r}. Fix that; count the words.")
    raise RuntimeError("describer kept leaking")


def main():
    out = HERE / "out" / PAINTING["id"] / "chain"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "chain.json"
    # Resume: keep completed hops (both renders guessed), redo the rest.
    hops = json.loads(log.read_text()) if log.exists() else []
    hops = [h for h in hops if len(h["renders"]) == RENDERS_PER_DESCRIPTION]
    source = out / f"hop{len(hops):02d}_render0.png" if hops else PAINTING["file"]

    for hop in range(len(hops) + 1, MAX_HOPS + 1):
        description = describe_clean(source)
        print(f"[hop {hop}] {source.name} -> {description!r}")
        renders = []
        for n in range(RENDERS_PER_DESCRIPTION):
            img = out / f"hop{hop:02d}_render{n}.png"
            render(description, img)
            guesses = guess(img)
            score, rank = rank_score(guesses, ANSWER_KEYS)
            print(f"  [render {n}] top guess: {guesses[0]!r}  (answer rank: {rank})")
            renders.append({"image": img.name, "guesses": guesses, "score": score, "rank": rank})
        hops.append({"hop": hop, "source": source.name, "description": description,
                     "renders": renders})
        log.write_text(json.dumps(hops, indent=2))

        if all(r["rank"] != 1 for r in renders):
            print(f"[stop] top guess wrong on both images at hop {hop}")
            break
        source = out / f"hop{hop:02d}_render0.png"
    else:
        print(f"[stop] still recognized after {MAX_HOPS} hops")


if __name__ == "__main__":
    main()
