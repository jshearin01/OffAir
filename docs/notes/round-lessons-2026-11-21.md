# Lessons from the 2026-11-21 feature round (ce-compound)

Corrected beliefs captured for the next round. Each was learned
the hard way in this round — by a failing test, a screenshot, or
a worker's red-first proof — not assumed.

1. **Labels must be computed from the persisted values, not the
   raw engine output.** `_label_segments_from_turns` labelled from
   full-precision turns while `to_dict()` rounded boundaries to
   3 decimals for `diarization.json`; the returned payload and the
   file disagreed on sub-millisecond boundaries, and learning read
   the file. Fix pattern: derive labels from the rounded turns,
   serialize exactly the returned list, then re-read both files and
   raise loudly on any mismatch. Any future producer/consumer pair
   that rounds on write must share the rounded source.

2. **Manual learn paths had no mixed-cluster guard at all.**
   `auto_identify_session` guarded; `learn_from_session`, Confirm,
   and Correct trusted full-cluster embeddings. A shared guard
   predicate used by all four paths is the fix — and a guard that
   lives in only one path is a bug waiting for the other three.

3. **Ollama's client shared one 10 s timeout for every request,
   including `/api/generate`.** A real CPU summary could never
   return. Generate needs its own long timeout (600 s here); probe
   and listing keep short ones. When a client wraps a long-running
   operation, check the timeout per endpoint, not per client.

4. **Killing only the direct child does not end a CLI run.** A
   grandchild holding the stdout pipe kept a 0.3 s-timeout run alive
   for 5.0 s; process-group kill took 0.37 s. Any subprocess that
   can spawn children needs its own process group + group kill on
   cancel/timeout — on both POSIX (`start_new_session`) and
   Windows (`CREATE_NEW_PROCESS_GROUP`).

5. **The 1280x800 geometry audit is blind to compression.** It
   catches widgets overflowing the window, not buttons squeezed
   below their size hints ("Paus", "ookmar"). A second assertion
   class is needed: rendered width >= sizeHint width for buttons
   in fixed-width cards. Screenshots remain the backstop — view
   them; the audit passing is not proof.

6. **`/tmp` is a 512 MB tmpfs that eats test runs and demo data.**
   Run suites with `TMPDIR=~/workspace/.pytest-tmp`, clear stale
   `/tmp/pytest-of-*` before Qt runs, keep scratch copies under
   `~/workspace`, and expect `/tmp/meettrace-demo` to need
   re-seeding (seed + prep + rename_identity + suggestions
   regenerate) before any screenshot refresh.

7. **A feeder that nothing calls is inert, not done.** The pre-roll
   buffer shipped with no `feed` caller; the follow-up unit had to
   add the feeder, and the lifecycle contract (stop feeder, drain
   buffer, start capture) had to be written down verbatim for the
   UI unit to consume. A core with no caller needs its wiring
   named in the same unit, or an explicit follow-up before the
   round closes.

8. **Qt `&` in a group-box/label title is a mnemonic, not text.**
   "Pre-roll & enterprise cleanup" renders as "Pre-roll
   _enterprise cleanup". Escape user-facing `&` as `&&` in Qt
   widget titles.

9. **Stale counts propagate.** Adding the 8th/9th MCP tool left
   "seven tools" in a test expectation, two docstrings, and the
   README. When a count changes, grep the whole tree for the old
   number — tests that assert the count will fail, but prose
   won't.

## Merge-time lessons (Nov 21–22 2026, found verifying the round)

10. **A modal dialog in the record path hangs the whole suite.**
    The low-disk warning was a modal `QMessageBox` fired from
    `on_record`. Pytest's tmp dir lives on the 512 MB tmpfs, always
    below the 1 GiB threshold, so the dialog could never be
    dismissed offscreen and the Qt suite wedged at the first
    auto-record test (faulthandler stack: `QMessageBox.warning`
    inside `_check_disk_while_recording`). Warnings fired while a
    recording runs must be non-modal (kept alive on `self`), and
    the regression test asserts `isModal() is False`.

11. **A page taller than the viewport cannot be fixed by
    minimum heights.** The Settings page (devices + advanced +
    form) is ~1128 px tall in a 641 px viewport; Qt violates
    minimum sizes when the total minimum exceeds the space, so
    wrapped hint labels clipped to one line no matter what their
    minimum reported. Fix: one scroll area for the whole page,
    with the dialog form in a holder whose only child the rebuild
    replaces — devices/advanced survive rebuilds, nothing is ever
    squeezed. (A scroll area settles its content on the second
    event-loop turn; the geometry audit pumps events twice
    before auditing.)

12. **A missing optional dependency masks a real bug.** The Qt
    test venv lacked `onnxruntime`, so hybrid search silently
    degraded to text everywhere and four UI call sites that called
    `search_sessions(...)` without `mode=` — defaulting to hybrid
    — behaved as text search. Installing `onnxruntime` (a real
    dependency, already bundled) exposed it: "Text" mode returned
    semantic hits. Fix: pass `mode="text"` explicitly at every
    text-intended call site (browser + Library, incl. fallbacks);
    MCP already did. Keep the Qt venv's dependencies at parity
    with requirements, and never let a test assume the semantic
    model is unavailable — pin the mode explicitly instead.

13. **Tinyfix's segment re-splitting must use per-segment length
    deltas.** Length-changing fixes applied to the joined text ate
    characters off later segments (`Their`→`They're` turned
    `books` into `book`). Record each fix's span, then re-split
    with per-segment deltas. Port word-list tables verbatim
    including set construction (`flatMap` plurals), and read the
    orchestration loop (`index.ts`), not just the engine files —
    it hides behaviours the engine files don't show.

