# Diarization — who said what

Diarization labels each transcript segment with a speaker. Pick
the mode on the Session view (Diarization group) before pressing
"Diarize"; the result is written to `diarization.json` in the
session folder, and the Session view's transcript shows the labels.

## The three modes

- **Simple — You vs Remote (no token, fast).** Labels segments by
  mic vs system track energy: you (the microphone) vs everyone else
  (the call audio). No model, no download, instant. The right
  choice when there are only two sides to the call.
- **Sherpa — multi-speaker, no token (the default).** Multi-speaker
  diarization via sherpa-onnx (pyannote segmentation ONNX +
  ERes2Net). No Hugging Face account or token; the models (~33 MB)
  download once on first use from GitHub releases. Use this when
  several people speak on the call.
- **Full — pyannote (needs HF token).** Multi-speaker diarization
  via pyannote.audio on CPU. Needs a Hugging Face token and accepted
  model licences (paste the token into the Hugging Face token field
  on the Session view, in the Diarization group, or set `HF_TOKEN`
  in the environment — the token is never stored; the app shows a
  guided setup). Only needed if sherpa's results disappoint you.

## Speaker hints

The min/max speaker count boxes under the mode picker are hints,
not a hard count: they tell the diarizer roughly how many voices
to look for. Leave them at the defaults unless a session is
systematically over- or under-split.

## What happens next

Diarized segments get labels like "Speaker A", "Speaker B" (or
"You" / "Remote" in Simple mode). The Session view then tries to
match those voices to people OffAir already knows — see
[Speakers & renaming](SPEAKERS.md) — and renaming a speaker there
updates the labels in every past session.

## Labels on disk always match the labels returned

The speaker labels written to `diarization.json` and
`transcript.json` are exactly the labels diarization returned —
computed from the same turn boundaries that are stored in the
file, with no re-derivation in between. If the written files
ever disagreed with the returned labels, diarization fails
loudly instead of leaving wrong labels for speaker learning to
read. (A sub-millisecond boundary case once let the two
disagree; it is covered by a regression test.)

## Thresholds

The sherpa clustering threshold (0.5) and the ECAPA match
threshold (0.55) were swept on the 3-speaker fixture on
21 November 2026 — see
[notes/threshold-sweep-2026-11-21.md](notes/threshold-sweep-2026-11-21.md).
Neither default moved: 0.5 and 0.6 produced identical results,
0.4 over-segmented, and the match thresholds were
indistinguishable on clean clusters. Real-meeting validation is
still the open gate — the fixture is clean synthetic speech.
