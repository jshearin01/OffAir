# Transcript cleanup: the repeat rules are removed (Nov 22, 2026)

## Decision

The false-start-repair and repeat-dedup rules in
`app/transcription/cleanup.py` were **removed outright** on Nov 22,
2026. The deterministic layer now has three rules only:

1. names-dictionary corrections,
2. filler removal (whole-word `um`, `uh`, `hmm`, `er`, `erm`,
   `mmm`, `ah`, anywhere in the text),
3. punctuation tidy (collapse repeated punctuation, spacing).

Repeats in speech are kept verbatim. The `CleanStats` fields
`false_starts_repaired` / `repeated_words_removed` still exist and
still appear in the cleaning block of `transcript-clean.json`, but
they are always 0 — kept so the artifact shape does not change for
existing readers.

The Ollama polish prompt and the enterprise-cleanup prompt still
instruct an LLM to handle false starts. That is a contextual-judgment
mechanism, not a regex rule, and is out of scope for this change.

## Evidence

Two real-speech measurements on saved YouTube sessions, scored
against published transcripts with two agreeing WER implementations
(measurement reports are dev-machine artifacts, not committed to
this repo; the numbers below are the decision record):

- **Test 1** (Matt Cutts TED talk, 207 s clean speech, 494 words):
  verbatim WER 1.42%; cleanup changed nothing, in either mode.
- **Test 2** (Lex Fridman / Tom Brands interview excerpt, 542 s
  disfluent speech, 1,697 words): verbatim WER 8.72%.
  Deterministic cleanup made it **worse**: 9.07%. Every removal
  was wrong — the published transcript keeps all six words:

  | Segment | Rule | Removed | Ground truth keeps |
  |---|---|---|---|---|
  | 0 | false-start (dash) | "feeding" from "feeding — feeding that —" | the repeat |
  | 19 | repeat dedup | second "maybe" | "maybe maybe" |
  | 35 | repeat dedup | second "that" | "that that" |
  | 51 | repeat dedup | one "technique" | the chant "technique, technique, technique" |
  | 66 | false-start (bare) | "in the" from "in the in the 10,000 reps" | the restart |

  The fixture corpus that originally justified the rules contains
  only synthetic hand-written sentences — no real-speech evidence
  that the rules ever help. Test 1 shows 0 firings on clean speech;
  test 2 shows 6/6 removals wrong on disfluent speech. A rule that
  only ever harms on real speech is worse than no rule.

## Rule formulations considered and rejected

Per the decision procedure (a formulation had to distinguish the
test-2 kept repeats from genuine stutter, with no new false
positives on the fixture corpus):

- **Function-word-only dedup** — fails: "that that" and "in the
  in the" are function words, and ground truth keeps them.
- **Triple-only dedup** — fails: the chant is a triple, and ground
  truth keeps all three.
- **Dash + fragment repair** — fails: Whisper emits whole words,
  not fragments, and the dash case ("feeding — feeding") was kept
  by ground truth. No real-speech case proves dash repair correct.
- **Timing-based** (short gap = stutter) — impossible: there are
  no word-level timestamps in the transcript segments.

No formulation survives. Hence removal, not repair.

## Re-measurement (Nov 22, 2026, after the change)

Re-ran `clean_session()` on copies of the saved verbatim
`transcript.json` files only — no re-recording, no re-transcription
(harness kept with the dev-machine measurement artifacts, not
committed to this repo; scoring uses the original harnesses' normalization
`[^a-z0-9]+` → space with `wer.align`, cross-checked by the
rolling-row Levenshtein from `verify2.py`; the two implementations
agree exactly on every number below). The verbatim
`transcript.json` was verified byte-identical after every run.

| Session | Verbatim | New default (tinyfix OFF) | Explicit OFF | Explicit ON, careful |
|---|---|---|---|---|
| Test 1 (494 words) | 1.42% (7 edits) | 1.42% (7) | 1.42% (7) | 1.42% (7) |
| Test 2 (1,697 words) | 8.72% (148) | **8.72% (148)** | 8.72% (148) | 8.84% (150) |

Test 2 deterministic-only returns to the verbatim 8.72% (was 9.07%):
the 6 wrong removals are gone. The only remaining deterministic
change is the punctuation tidy ("to..." → "to."), which normalizes
away in scoring. Test 1 stays at 0 changes in every variant.

Explicit-ON careful is 8.84% (150 = 148 + the same 2 tinyfix
corruptions as before the change: "lace"→"place", "hate"→"have").
The tinyfix stage itself is unchanged — only its default flipped
(see the companion section below). The original 9.19% (156) was
148 + 6 removals + 2 corruptions; the arithmetic closes.

## Companion change: tinyfix default OFF (Nov 22, 2026)

Test 2 also showed the tinyfix stage changing 2 correct words at
0.9998 confidence, with none of Whisper's 62 substitutions falling
in any tinyfix confusion class. The stage is now **opt-in**: off by
default on fresh installs. The mode default ("careful") is
unchanged. Details:

- `TINYFIX_DEFAULT_ENABLED` in `app/config.py` is `False`;
  `clean_session(tinyfix_enabled=None)` resolves to OFF.
- A stored explicit opt-in survives: settings carry a
  `tinyfix_enabled_explicit` flag (mirroring the Nov 20, 2026
  `whisper_model_explicit` pattern). It is set only by a
  user-driven click on the Settings checkbox (`clicked` signal —
  never programmatic `setChecked`), and is written on save.
- Upgrade migration: a pre-flag settings file with stored
  `tinyfix_enabled=True` cannot be told apart from the old ON
  default (every settings save wrote the checkbox state, checked
  by default), so it migrates to OFF, non-explicit. Stored False
  or absent stays OFF. Honest caveat: an explicit opt-in saved
  under 1.0.5 is indistinguishable from the old default and
  resets to OFF once on upgrade; re-enabling in Settings sets
  the flag and persists.
- The Settings checkbox tooltip and the docs now say off by
  default / opt-in. The mode-combo wording ("Careful is the
  default") is unchanged.

## Do not re-add a repeat rule without real-speech evidence

Any future repeat-removal proposal must be measured against real,
disfluent speech with a published ground truth — not fixtures —
and must distinguish the cases in the table above from genuine
stutter. Until such evidence exists, repeats stay verbatim.
