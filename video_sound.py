"""Movie-clip benchmark with sound: clip -> frames + audio notes -> description ->
generated clip (with audio) -> grade on picture and sound.

Picture: every describer and the judge see the same labeled 6-frame grid (video_pilot).
Sound: an audio model turns each soundtrack (original and generated) into the same kind
of notes - timestamped transcript, music, sound effects - so describers and the judge,
who can't hear, all work from text. The judge writes its account of the original once
and grades every recreation against it.

Usage: OPENROUTER_API_KEY=... python3 video_sound.py <film.mp4> <start_s> <duration_s> <clip_id>
"""
import base64
import itertools
import json
import re
import sys
from pathlib import Path

ARGS = sys.argv[1:]
import video_pilot as vp  # noqa: E402  (frame grids, crop, ffmpeg helpers; imports eval_one)

e = vp.e
DESCRIBERS = ["x-ai/grok-4.7", "google/gemini-3.8-flash"]
VIDEO_MODELS = {  # model -> aspect ratios it accepts
    "x-ai/grok-imagine-video-1.5-lite": ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"],
    "bytedance/seedance-2.0-mini": ["1:1", "3:4", "9:16", "4:3", "16:9", "21:9", "9:21"],
    "alibaba/wan-3.0": ["16:9", "4:3", "1:1", "3:4", "9:16"],
}
# Not from any describer's company. Tested: Voxtral transcribed well but looped when asked
# for timestamps; Nemotron Omni said it could not hear; MiMo handled the full notes prompt.
AUDIO_MODEL = "xiaomi/mimo-v2.6-flash"
JUDGE = "anthropic/claude-opus-5.5"
WORD_CAP, SOUND_CAP, MIN_WORDS = 95, 35, 30  # 60 picture + 35 sound
SOUND_METRIC_WEIGHTS = {"speech": 1, "music": 1, "sound_effects": 1, "sync": 0.5}  # sync is rough
# Never allowed in a description: the film's title, cast and character names.
BANNED = ["six-gun rhythm", "six gun rhythm", "tex fletcher", "fletcher", "joan barclay",
          "ralph peters", "jim davis"]

METRICS = dict(vp.METRICS)
METRICS.update({
    "speech": "what is said (content), by what kind of voice (man/woman, tone)",
    "music": "presence, genre, tempo, mood and instruments of any music",
    "sound_effects": "non-speech sounds: hooves, wind, doors, gunshots...",
    "sync": "sounds and speech line up with the picture (lips move when someone talks, "
            "sounds match visible actions)",
})


def has_music(notes):
    """False when the audio notes say the original has no music. Then "no music" in a
    recreation is not an achievement, so the music metric is left out instead of
    handing every recreation a free 10."""
    m = re.search(r"music[^a-z]*?(none|no music|not present|absent)", notes, re.I)
    return m is None


def quoted_words(notes):
    """Words inside quotation marks in audio notes, i.e. the transcribed speech."""
    quotes = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', notes)
    return [w for q in quotes for w in re.findall(r"[a-z']+", q.lower())]


def speech_word_match(orig_notes, gen_notes):
    """Share of the original's spoken words that also occur in the recreation (0-1).
    Objective and repeatable; reported next to the judge's speech score."""
    orig, gen = quoted_words(orig_notes), set(quoted_words(gen_notes))
    return round(sum(w in gen for w in orig) / len(orig), 2) if orig else None


def audio_notes(video, cache):
    """Soundtrack -> timestamped text notes, via a model that can hear."""
    if cache.exists():
        return cache.read_text()
    wav = cache.with_suffix(".wav")
    vp.ff("-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav))
    data = base64.b64encode(wav.read_bytes()).decode()
    prompt = (
        "Listen to this short soundtrack and write notes for someone who cannot hear it. "
        "1) Speech: transcribe every line with its start time in seconds and the kind of "
        "voice (man/woman, approximate age, tone). Write 'none' if there is no speech. "
        "2) Music: genre, tempo, mood, instruments, singing; or 'none'. "
        "3) Sound effects: each sound with its time; or 'none'. "
        "4) If the track is silent, say 'silent'. Do not name or guess any film, song, "
        "performer or character. Plain text, at most 120 words."
    )
    text = e.call(AUDIO_MODEL, [{"type": "text", "text": prompt},
                                {"type": "input_audio",
                                 "input_audio": {"data": data, "format": "wav"}}],
                  max_tokens=1500)["content"].strip()  # cap guards against looping output
    wav.unlink()
    cache.write_text(text)
    return text


def check(text):
    words = len(text.split())
    lower = text.lower()
    problems = [f"banned term {b!r}" for b in BANNED if b in lower]
    if words > WORD_CAP:
        problems.append(f"{words} words > cap {WORD_CAP}")
    if words < MIN_WORDS:
        problems.append(f"only {words} words")
    if "sound:" not in lower:
        problems.append("missing the 'Sound:' part")
    else:
        sound = lower.split("sound:", 1)[1]
        if len(sound.split()) > SOUND_CAP:
            problems.append(f"Sound part has {len(sound.split())} words > {SOUND_CAP}")
    return problems


def describe(model, grid, notes, duration):
    prompt = (
        f"These are {vp.N_FRAMES} frames from a {duration:.0f}-second movie clip, in time order "
        "left to right, top row then bottom row; each is labeled with its number and time. "
        "Below the image are notes on the clip's soundtrack.\n\n"
        f"Soundtrack notes:\n{notes}\n\n"
        f"Describe the clip in at most {WORD_CAP} words so a video maker who never saw or "
        "heard it could recreate it as closely as possible. Write the picture first, then a "
        f"final part starting with 'Sound:' of at most {SOUND_CAP} words (dialogue, voices, "
        "music, effects). It will be compared with the original on: composition, color, "
        "lighting, figures, setting, style of the footage, the action, movement and camera "
        "motion, order of events, speech, music, sound effects and sync; sound counts for up "
        "to 40% depending on how much speech and music the clip has. State the "
        "orientation, and say where each person stands in the frame (left, right, center, "
        "foreground or background). Do NOT name the film, actors, characters, songs or performers. Output "
        "only the description."
    )
    feedback = ""
    for _ in range(8):
        text = e.call(model, [{"type": "text", "text": prompt + feedback},
                              e.image_part(grid)])["content"].strip()
        problems = check(text)
        if not problems:
            return text
        print(f"  [{model}] rejected: {'; '.join(problems)}")
        feedback = f"\n\nPrevious attempt rejected ({'; '.join(problems)}): {text!r}. Fix it."
    raise RuntimeError(f"{model} kept failing the checks")


def _generate(model, prompt, duration, aspect, out_mp4, extra):
    import time
    import urllib.request
    key = e.os.environ["OPENROUTER_API_KEY"]
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def req(url, body=None):
        r = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                   headers=hdr, method="POST" if body else "GET")
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"{model}: HTTP {err.code}: {err.read().decode()[:400]}") from None

    job = json.loads(req("https://openrouter.ai/api/v1/videos", {
        "model": model, "prompt": prompt, "duration": int(round(duration)),
        "resolution": "480p", "aspect_ratio": aspect, **extra}))
    while True:
        time.sleep(15)
        st = json.loads(req(job["polling_url"]))
        if st["status"] == "completed":
            break
        if st["status"] in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"{model} job {st['status']}: {st.get('error')}")
    out_mp4.write_bytes(req(f"https://openrouter.ai/api/v1/videos/{job['id']}/content?index=0"))
    c = (st.get("usage") or {}).get("cost") or 0
    with e._lock:
        e.cost += c
    return c


SOUND_WEIGHT_RULES = (
    "Decide how much the soundtrack should count in this clip's grade (sound_weight, from 0 "
    "to 0.40; the picture gets the rest). Rules: speech AND music -> 0.40; only speech or "
    "only music -> 0.25; silent (no speech, no music; hiss or room tone only) -> 0. If there "
    "is very little speech or music (a word or two, a few faint notes), or only sound "
    "effects, lower the weight below the matching rule by as much as you think is fair."
)


def original_account(out, grid, notes):
    """The judge's account of the original plus the clip's sound weight, decided once per
    clip from the original so every recreation of it is weighed the same way."""
    cache = out / "original_account.json"
    if cache.exists():
        return json.loads(cache.read_text())
    prompt = (
        f"These are {vp.N_FRAMES} frames of a movie clip (left to right, top row then bottom "
        "row, each labeled with number and time), followed by notes on its soundtrack.\n\n"
        f"Soundtrack notes:\n{notes}\n\n1) In 4-5 sentences write down what happens: "
        f"{vp.EVENTS_INSTRUCTIONS}; and what is heard and when.\n2) {SOUND_WEIGHT_RULES}\n\n"
        'Reply with JSON only: {"account": "...", "sound_weight": <0-0.40>, '
        '"sound_weight_reason": "<one sentence>"}'
    )
    raw = e.call(JUDGE, [{"type": "text", "text": prompt}, e.image_part(grid)])["content"]
    res = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    res["sound_weight"] = max(0.0, min(0.40, float(res["sound_weight"])))
    cache.write_text(json.dumps(res, indent=2))
    return res


def judge(orig_grid, account, gen_grid, gen_notes, metrics):
    rubric = "\n".join(f'- "{k}": {v}' for k, v in metrics.items())
    prompt = (
        f"Image 1 shows {vp.N_FRAMES} frames of an original movie clip; image 2 shows "
        f"{vp.N_FRAMES} frames, at the same relative times, of an attempt to recreate it from "
        "a short text description. Frames run left to right, top row then bottom row, each "
        "labeled with number and time.\n\n"
        f"What happens in the original, picture and sound (treat as correct): {account}\n\n"
        f"Notes on the recreation's soundtrack: {gen_notes}\n\n"
        f"First write down what happens in the recreation: {vp.EVENTS_INSTRUCTIONS}; and what "
        "is heard. Then score how closely the recreation matches the original on each "
        "metric, 0-10 (10 = indistinguishable, 5 = same idea with clear differences, 0 = "
        "unrelated or missing). Judge timing loosely but order of events strictly. For sync, "
        "use the frames and the timestamps in the notes. Be strict.\n\nMetrics:\n" + rubric +
        '\n\nReply with JSON only: {"recreation_events": "<2-3 sentences>", '
        '"<metric>": {"score": <0-10>, "reason": "<one sentence>"}, ...}'
    )
    raw = e.call(JUDGE, [{"type": "text", "text": prompt}, e.image_part(orig_grid),
                         e.image_part(gen_grid)])["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def weighted_scores(scores, metrics, sound_weight):
    """Picture = mean of picture metrics; sound = weighted mean (sync counts half);
    overall = (1 - sound_weight) * picture + sound_weight * sound."""
    pic = [k for k in vp.METRICS if k in metrics]
    snd = {k: w for k, w in SOUND_METRIC_WEIGHTS.items() if k in metrics}
    p = sum(scores[k]["score"] for k in pic) / len(pic)
    out = {"mean_picture": round(p, 2), "sound_weight": sound_weight}
    if snd and sound_weight > 0:
        so = sum(scores[k]["score"] * w for k, w in snd.items()) / sum(snd.values())
        out.update(mean_sound=round(so, 2), mean=round((1 - sound_weight) * p + sound_weight * so, 2))
    else:
        out.update(mean_sound=None, mean=round(p, 2))
    return out


def main():
    src, start, duration, clip_id = Path(ARGS[0]), float(ARGS[1]), float(ARGS[2]), ARGS[3]
    out = e.HERE / "out" / "video_sound" / clip_id
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "original.mp4"
    if not clip.exists():
        crop = vp.detect_crop(src, start, duration)
        vp.ff("-ss", str(start), "-t", str(duration), "-i", str(src),
              *(["-vf", f"crop={crop}"] if crop else []), "-c:v", "libx264",
              "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip))
    _, w, h = vp.probe(clip)
    grid = out / "original_grid.png"
    vp.frame_grid(clip, grid)
    notes = audio_notes(clip, out / "original_audio.txt")
    acc = original_account(out, grid, notes)
    account, sound_weight = acc["account"], acc["sound_weight"]
    if sound_weight == 0:  # silent clip: grade the picture only
        metrics = dict(vp.METRICS)
    elif has_music(notes):
        metrics = dict(METRICS)
    else:  # no free 10 for "no music", but added music still costs points
        metrics = {k: v for k, v in METRICS.items() if k != "music"}
        metrics["sound_effects"] = (METRICS["sound_effects"] + "; the original has NO music, so any "
                                    "music or extra sounds it lacks count against the recreation")
    print(f"original {w}x{h}\nAUDIO NOTES: {notes}\nJUDGE ACCOUNT: {account}\n"
          f"sound weight {sound_weight:.0%} ({acc.get('sound_weight_reason')}); "
          f"music metric: {'used' if 'music' in metrics else 'skipped'}\n")

    descriptions = {}
    for m in DESCRIBERS:
        f = out / f"desc_{m.replace('/', '_')}.txt"
        if not f.exists():
            t = e.cost
            f.write_text(describe(m, grid, notes, duration))
            (out / f"desc_{m.replace('/', '_')}.cost").write_text(str(e.cost - t))
        descriptions[m] = f.read_text()
        print(f"[{m}] {descriptions[m]}\n")

    results = []
    for m, vm in itertools.product(DESCRIBERS, VIDEO_MODELS):
        d = out / f"{m.replace('/', '_')}__{vm.replace('/', '_')}"
        d.mkdir(exist_ok=True)
        rp = d / "result.json"
        if rp.exists():
            results.append(json.loads(rp.read_text()))
            continue
        aspect = min(VIDEO_MODELS[vm], key=lambda a: abs(int(a.split(":")[0]) / int(a.split(":")[1]) - w / h))
        r = {"describer": m, "video_model": vm, "aspect": aspect, "cost_usd": {}}
        try:
            r["cost_usd"]["video"] = round(_generate(vm, descriptions[m], duration, aspect,
                                                     d / "video.mp4", {"generate_audio": True}), 4)
        except RuntimeError as err:
            print(f"  FAILED {vm}: {err}")
            r["error"] = str(err)
            rp.write_text(json.dumps(r, indent=2))
            results.append(r)
            continue
        vp.frame_grid(d / "video.mp4", d / "grid.png")
        t = e.cost
        r["audio_notes"] = audio_notes(d / "video.mp4", d / "audio.txt")
        r["scores"] = judge(grid, account, d / "grid.png", r["audio_notes"], metrics)
        r["cost_usd"]["notes_and_judge"] = round(e.cost - t, 4)
        r["speech_word_match"] = speech_word_match(notes, r["audio_notes"])
        r.update(weighted_scores(r["scores"], metrics, sound_weight))
        rp.write_text(json.dumps(r, indent=2))
        results.append(r)
        print(f"  {m} x {vm}: picture {r['mean_picture']} sound {r['mean_sound']} "
              f"overall {r['mean']}  speech words {r['speech_word_match']}  ${r['cost_usd']}")
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\ntotal cost this run ${e.cost:.3f}")


if __name__ == "__main__":
    main()
