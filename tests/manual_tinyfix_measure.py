"""One-off measurement harness for the tinyfix default-mode decision.

Not part of the pytest suite (manual_ prefix). Downloads the pinned
model files to ~/.offair/models on first run, then prints, for
both modes: every fix made to the fixture script text, the
planted-error battery results, and timing. Verdicts on the
fixture script are hand-checked after the run and recorded in
docs/notes/tinyfix-2026-11-21.md — this script only reports what
changed, it does not judge.
"""
from __future__ import annotations

import time
from pathlib import Path

from app.transcription.tinyfix import build_checker

FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "meeting-script.txt"

PLANTED = [
    # (sentence with errors, expected fixed sentence)
    ("I went form the store to there house.",
     "I went from the store to their house."),
    ("Their going to bring there books over hear.",
     "They're going to bring their books over here."),
    ("Your going to love this, its grate.",
     "You're going to love this, it's great."),
    ("We need to seperate the wheat form the chaff and recieve it.",
     "We need to separate the wheat from the chaff and receive it."),
    ("Its to late to turn back now, we passed the point of no return.",
     "It's too late to turn back now, we passed the point of no return."),
    ("The affect was immediate, much more then we expected.",
     "The effect was immediate, much more than we expected."),
    ("I definately think we should adress this untill its resolved.",
     "I definitely think we should address this until it's resolved."),
    ("Thier proposal occured to me while I was reading the minutes.",
     "Their proposal occurred to me while I was reading the minutes."),
    ("Please confirm weather you can attend the meeting on Tuesday.",
     "Please confirm whether you can attend the meeting on Tuesday."),
    ("The team wich worked on this did a grate job overall.",
     "The team which worked on this did a great job overall."),
    ("We need to accomodate everyones schedule for the pilot.",
     "We need to accommodate everyone's schedule for the pilot."),
    ("Its neccessary to check the hardware list before Monday.",
     "It's necessary to check the hardware list before Monday."),
    ("The past week past quickly and we past the deadline.",
     "The past week passed quickly and we passed the deadline."),
    ("There going to loose the contract if they dont act soon.",
     "They're going to lose the contract if they don't act soon."),
    ("I cant believe its already Friday and the notes arent done.",
     "I can't believe it's already Friday and the notes aren't done."),
]

CORRECT_CONTROLS = [
    "The meeting is on Tuesday and the notes are in the shared folder.",
    "We discussed the budget and the deadline is Friday.",
    "OffAir records Teams calls locally with no cloud upload.",
    "The sherpa diarization model runs on the CPU with onnxruntime.",
    "Priya and Marcus joined the call from the hardware lab.",
    "The pilot budget is twelve thousand dollars.",
    "Please send the updated timeline by Monday morning.",
    "The books are on the shelf next to the hardware list.",
    "Their books and their notes were both in the shared folder.",
    "Qwen and Ollama are used for the optional local summary.",
]


def _run(checker, texts):
    started = time.monotonic()
    results = []
    for text in texts:
        fixed, corrections = checker.fix_text(text)
        results.append((text, fixed, corrections))
    return results, time.monotonic() - started


def main() -> None:
    script_text = FIXTURE_SCRIPT.read_text(encoding="utf-8")
    script_lines = [line.strip() for line in script_text.splitlines()
                    if line.strip()]
    for mode in ("balanced", "careful"):
        checker = build_checker(mode=mode)
        print(f"=== mode: {mode} ===")
        results, elapsed = _run(checker, script_lines)
        changed = [(s, f, c) for s, f, c in results if f != s]
        print(f"fixture script: {len(changed)}/{len(script_lines)} "
              f"lines changed in {elapsed:.2f}s")
        for source, fixed, corrections in changed:
            print(f"  - {source}\n  + {fixed}\n"
                  f"    {[(c.original, c.replacement) for c in corrections]}")
        planted, elapsed = _run(
            checker, [source for source, _ in PLANTED])
        fixed_count = 0
        exact = 0
        for (source, expected), (_, fixed, _) in zip(PLANTED, planted):
            if fixed == expected:
                exact += 1
            if fixed != source:
                fixed_count += 1
        print(f"planted battery: {exact}/{len(PLANTED)} exact, "
              f"{fixed_count}/{len(PLANTED)} changed in {elapsed:.2f}s")
        for (source, expected), (_, fixed, _) in zip(PLANTED, planted):
            if fixed != expected:
                print(f"  MISS: {source}\n    got: {fixed}\n"
                      f"    want: {expected}")
        controls, elapsed = _run(checker, CORRECT_CONTROLS)
        wrong = [(s, f) for s, f, _ in controls if f != s]
        print(f"correct controls: {len(wrong)}/{len(CORRECT_CONTROLS)} "
              f"wrongly changed in {elapsed:.2f}s")
        for source, fixed in wrong:
            print(f"  WRONG: {source}\n    -> {fixed}")


if __name__ == "__main__":
    main()
