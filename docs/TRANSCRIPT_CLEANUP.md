# Transcript cleanup + names dictionary

OffAir keeps the **verbatim transcript as the record** and derives
everything else from it. Cleanup never edits
`transcript.json` / `transcript.txt` / `transcript.srt` — those files
stay byte-identical after any clean pass (a test proves it:
`tests/test_transcript_cleanup.py`). If a cleanup ever looks wrong,
delete the derived files and re-clean.

## The rule layer (automatic, deterministic, no LLM)

After every transcription — manual or auto-processed — OffAir runs
the rule cleanup and writes two derived artifacts next to the
verbatim files:

| File | Contents |
| --- | --- |
| `transcript-clean.json` | Cleaned segments, same shape as `transcript.json`, plus a `"cleaning"` block with the per-rule counts and the source file |
| `transcript-clean.txt` | The cleaned segments as `[MM:SS] Speaker: text` lines, speaker display names applied like the verbatim export |

The chain re-cleans after diarization and after speaker
identification, so the cleaned artifact always carries the final
speaker names. Rules, in order:

1. **Names-dictionary corrections** (see below).
2. **Filler removal** — um, uh, hmm, er, erm, mmm, ah (whole
   words only — "umbrella" is untouched). A filler at the start
   of a segment capitalises the new first word ("Um, let's go" →
   "Let's go").
3. **Punctuation / spacing tidy** — collapse whitespace, drop the
   space before `.,!?;:)`, collapse doubled punctuation, ensure a
   space after `,;:!?` when a letter follows.

There is deliberately **no repeat-removal rule**. An earlier
version had false-start repair and repeated-word dedup; both
were removed Nov 22, 2026, because on real disfluent speech
every removal they made was wrong — against the published
transcript of a 9-minute interview they deleted 6 words the
speaker actually said, including a deliberate "technique,
technique, technique" chant. No text-only signal distinguishes
a stutter repeat from intentional emphasis. Repeats are kept
verbatim; see `docs/notes/cleanup-repeat-rule-2026-11-22.md`.

Segments that clean down to nothing (a segment that was only
"Um.") are dropped from the cleaned artifact and counted as
`dropped_segments`.

Measured on the 61.3 s 3-speaker fixture (12 segments): the
deterministic rule pass takes **~2.5 ms** — effectively instant
next to the ~9 s Whisper-tiny transcription of the same audio
(model load included). It is pure Python regex over the segment
texts; no model, no network, negligible memory. The tinyfix stage
below adds about 0.1 s per sentence plus a ~1 s model load on
first use in a session.

## The tinyfix layer (opt-in, off by default, English only)

After the rules and the names dictionary, a small ONNX model —
tinyfix (paperweight/tinyfix, MIT, 1.9M parameters, 2.1 MB) —
fixes real words in the wrong place that no rule can see:
"form" → "from", "their" → "there", "your" → "you're". It scores
each word against the 16 words before it and 4 after, and swaps
only when a nearby real word fits clearly better. The stage runs
with cross-segment context (the window never resets at a segment
boundary) and never touches timestamps. It is **off by default**
since Nov 22, 2026 — enable it in Settings to opt in. On
disfluent real speech it changed 2 correct words ("lace" →
"place", "hate" → "have", both at 0.9998 confidence) while
fixing none of Whisper's 62 substitutions, which is why the
default flipped; see
`docs/notes/cleanup-repeat-rule-2026-11-22.md`.

- **Names win.** Every word of every names-dictionary entry is on
  tinyfix's ignore list, so a known name or product term is never
  "fixed". The dictionary runs first, as above.
- **Modes.** `careful` (the default) makes fewer wrong changes;
  `balanced` fixes a few more wrong words. Measured Nov 21 2026
  on the fixture script text and a planted-error battery
  (docs/notes/tinyfix-2026-11-21.md): careful made zero false
  changes on the clean fixture script where balanced made one
  ("clock" → "click"), while keeping the core mix-up fixes —
  hence careful as the shipped default mode. Both modes miss some
  errors the model does not score highly enough (`its` → `it's`,
  `affect` → `effect`). Measured Nov 22 2026 on disfluent real
  speech (see the note above), careful still changed 2 correct
  words — the fixture "0 false changes" does not generalize,
  which is why the stage is opt-in rather than default-on.
- **English only.** The stage is skipped when the transcript's
  language is known non-English, and runs when the language is
  absent (older transcripts) or English. The skip is recorded in
  the `"cleaning"` block as `tinyfix_skipped`.
- **First-use download.** The model files (~3.5 MB total) are
  downloaded once from public Hugging Face at a pinned revision,
  verified by sha256, and cached under
  `~/.offair/models/tinyfix/<revision>/`. The Settings view shows
  download status while it runs. If the download fails (no
  network) or the files fail their hash check, cleanup degrades
  to the deterministic rules alone and records
  `tinyfix_skipped: "unavailable: …"` — transcription and the
  session are never affected.
- **Settings.** A toggle turns the stage on (off by default)
  and a mode choice picks balanced/careful (Settings → "Pre-roll
  & enterprise cleanup" section). Leaving it off is recorded as
  `tinyfix_skipped: "disabled"`, and the output is byte-identical
  to the deterministic-only cleanup. An explicit opt-in is
  remembered across upgrades (see the note above).

The Session view's transcript has an **Original / Cleaned** toggle to switch
the segment list between the verbatim and the cleaned transcript,
and a **Re-clean transcript** button (next to the toggle) that
regenerates the derived artifacts — press it after renaming a
speaker or editing the names dictionary.

## The LLM polish layer (optional, OFF by default, local Ollama only)

**Polish with local LLM** (Session view) rewrites the *cleaned*
transcript into a separate artifact, `transcript-polished.md`,
whose header states plainly that it is an AI rewrite of the
verbatim transcript — never a replacement for it. It reuses the
existing Ollama client (`app/exports/local_llm.py`): localhost
only, never a cloud LLM. If Ollama is not running it fails with a
clear message ("Ollama is not running on localhost:11434…") and
leaves the rule-cleaned version in place.

Settings has a checkbox, **"AI-polish transcripts with the local
LLM after auto-processing"**, which runs the polish automatically
at the end of the auto-process chain. It is OFF by default; a
failed auto-polish is a status-line note, not a modal. The manual
Polish button works regardless of that checkbox — clicking it is
an explicit request.

## The enterprise-cleanup variant (optional, OFF by default, text only)

**Enterprise cleanup** (Session view) runs the headless Claude
Code CLI — `claude -p`, subprocess only, never ACP/A2A — over
the verbatim transcript and writes a third, distinct artifact,
`transcript-enterprise-clean.txt` (text only: the CLI returns
unstructured text, so no `transcript-enterprise-clean.json`
segment file exists). The Original / Cleaned toggle gains this
third variant, explicitly labeled "Enterprise-cleaned". The
verbatim transcript and the rule-cleaned artifacts are never
modified, and the three variants are never merged. The variant
registry lives in `app/transcription/cleanup.py`
(`TRANSCRIPT_VARIANTS = ("original", "cleaned", "enterprise")`,
`TRANSCRIPT_VARIANT_FILES`, `transcript_variant_path()`); the
runner lives in `app/transcription/enterprise_cleanup.py`.

What leaves the machine, stated plainly wherever the button
appears: **transcript text only — no audio ever leaves the
machine.** The verbatim transcript text is fed to the CLI on
stdin; the cleaned text comes back on stdout. The artifact's
header states the same.

The fixed prompt instructs the CLI to clean the transcript:
fix grammar, punctuation and obvious transcription errors;
remove fillers, repair false starts, remove repeated words, tidy
punctuation/spacing; keep every speaker label and timestamp
exactly; add no facts, names, dates, or commitments; remove no
spoken content except fillers and false starts; return only the
rewritten transcript in the same `[MM:SS] Speaker: text` format.
(The prompt is a contextual-judgment mechanism — an LLM reading
the whole transcript — not the removed regex rules above; it
still asks for false-start/repeat handling because the model can
judge intent from context in a way the regexes could not.)

Behaviour:

| Situation | Result |
| --- | --- |
| Success | `transcript-enterprise-clean.txt` written (header + cleaned text), atomically |
| `claude` CLI missing | Guided error naming the install (Claude Code, `claude` on PATH) — no artifact |
| Timeout (default 120 s, configurable) | Process (and its process group) killed, clear timeout error — no artifact |
| Cancel | Process killed, clear cancel error — no artifact |
| Non-zero exit / empty output | Clear error naming the failure — no artifact |

The runner is worker-thread-safe (no shared state; cancel kills
the running process) and must run in a worker thread, never the
UI thread. Settings (in `app/config.py`, enterprise-cleanup
block only): `enterprise_cleanup_enabled` (default `false`),
`enterprise_cleanup_timeout_seconds` (default `120.0`),
`enterprise_cleanup_claude_command` (default `"claude"` — the
CLI name on PATH, or an absolute-path override). The Ollama
defaults are untouched by this feature. Dictation sessions use
the deterministic cleaned note (`transcript-clean.txt`) as
their primary readable artifact; enterprise cleanup is a
separate, explicit action, never automatic.

## The names dictionary

`~/.offair/names_dictionary.json` — plain JSON, user-editable in
Settings (one entry per line; a bare line is a name,
`wrong -> Correct` is a correction):

```json
{"version": 1,
 "names": ["John", "Priya", "Marcus"],
 "corrections": {"Meat Trace": "MeetTrace"}}
```

It corrects two ways, both reversible because the original is kept:

1. **Decode-time biasing.** The names (and the correct spellings
   from the corrections map) become faster-whisper's
   `initial_prompt` at decode time
   (`TranscriptionEngine.transcribe_file(..., initial_prompt=...)`
   — verified against faster-whisper 1.2.1, which accepts the
   parameter). Whisper then hears "MeetTrace" instead of
   "Meat Trace" more often. No dictionary → no prompt is passed,
   so transcription behaves exactly as before.
2. **Post-process correction.** Known mishearings are replaced in
   the cleaned transcript with whole-word, case-insensitive
   matching — "Marcus" is never rewritten inside "Marcuson".
   The seed correction is real: Whisper tiny mishears the product
   name "MeetTrace" as "Meat Trace" in this project's demo data
   (the fixture test asserts the mapping; on the current tiny run
   the fixture happens to transcribe it correctly, so the fixture
   clean pass reports 0 corrections — the seed is still the
   correction to have when tiny does mishear it).

Renaming a speaker adds their name to the dictionary
automatically — the identity rename path
(`app/speakers/session.py::rename_identity`), the per-session
label rename, and correcting a suggestion all call
`add_name()`. A dictionary failure is logged, never raised, so it
cannot break a rename.

A missing or corrupt dictionary file falls back to the seed
defaults (the "Meat Trace" → "MeetTrace" correction); the file is
only written by an explicit save, so deleting an entry is
respected.

## Files

- `app/transcription/cleanup.py` — `clean_text` / `clean_segment` /
  `clean_session` (rule layer), `polish_session` (LLM layer),
  the three-variant registry (`TRANSCRIPT_VARIANTS`,
  `TRANSCRIPT_VARIANT_FILES`, `transcript_variant_path`)
- `app/transcription/enterprise_cleanup.py` —
  `EnterpriseCleanupRunner` / `enterprise_clean_text` /
  `enterprise_clean_session` (headless `claude -p`, text only)
- `app/transcription/names_dictionary.py` — load/save, `add_name`,
  `build_initial_prompt`, `apply_corrections`, editor format
  parse/format
- `app/transcription/engine.py` — `initial_prompt` passthrough
- `app/config.py` — `transcript_polish_enabled` setting
- `app/ui/main_window.py` — Original/Cleaned toggle, Re-clean and
  Polish buttons, cleanup hooks in the auto-process chain
  (after transcribe, after diarize, at the end of suggest)
- `app/ui/settings_dialog.py` — names dictionary editor + polish
  checkbox
- `app/speakers/session.py` — `rename_identity` adds the new name
  to the dictionary

## Tests

- `tests/test_transcript_cleanup.py` — one unit test per rule,
  dictionary prompt construction, whole-word replacement
  (including the substring negative test), the "Meat Trace" →
  "MeetTrace" seed case, rename-adds-to-dictionary, engine
  `initial_prompt` passthrough, polish success + Ollama-down
  failure, and the fixture clean pass (real Whisper tiny on
  `tests/fixtures/meeting-3speakers.mp3`: 12 segments, 0 fillers /
  0 corrections on the current run, verbatim byte-identical
  afterwards. (The `false_starts_repaired` / `repeated_words_removed`
  counts are retained in the cleaning record as always-zero fields
  since the repeat rules were removed — see the decision record
  below.)
- `tests/test_ui_cleanup.py` (Qt, gui-shots venv) — the
  Original/Cleaned toggle + Re-clean path, the Polish button's
  Ollama-down error, the Settings dictionary editor + polish
  checkbox, and the auto-process chain producing the cleaned
  artifact for a new session.
