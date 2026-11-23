# Threshold sweep — 21 November 2026

Sweep of the sherpa clustering threshold {0.4, 0.5, 0.6} × the ECAPA
match threshold {0.45, 0.55, 0.65} on the 3-speaker fixture
(`tests/fixtures/meeting-3speakers-16k.wav`, 61 s, 9 turns, 3 TTS
voices in John / Priya / Marcus order, repeated 3×). Run on this VM
with the real sherpa pipeline and the real default ECAPA backend
(`onnx-ecapa-v1`); the sherpa models were already present locally.
Synthetic fixture only — this is not real-meeting audio, and no
real-meeting claim is made here.

## How the numbers were measured

- **Turn accuracy** — each diarized turn's identity was checked
  with the Sherpa ERes2Net extractor against embeddings of the
  three isolated speaker recordings
  (`speaker-{john,priya,marcus}-16k.wav`; cross-speaker cosines
  −0.02 to 0.01, same-speaker 1.0 by construction). A turn counts
  correct when its predicted identity equals the script order for
  that position (John, Priya, Marcus, repeated). Turns shorter
  than 0.5 s cannot be embedded and were left unscored; with 13
  turns at threshold 0.4 the positional alignment shifts after
  the first split, so that row's turn accuracy is indicative, not
  exact. Turn accuracy is scored over the first 9 diarized turns.
- **Speaker accuracy (match step)** — for each diarized cluster,
  one ONNX ECAPA embedding over all of the cluster's turns was
  matched against a temporary database holding one reference
  sample per speaker, at each match threshold. "Expected" is the
  cluster's majority ERes2Net identity. Clusters made only of
  unscored short turns have no majority and are excluded from
  this step (1 cluster at threshold 0.4).
- **Cluster count** — distinct labels in the diarized turns, vs 3.
- **Cluster purity** — fraction of scored turns whose ERes2Net
  identity equals their own cluster's majority identity. A pure
  cluster holds one voice; a mixed cluster holds more than one.
- ECAPA reference separation on the isolated recordings:
  John–Priya 0.120, John–Marcus 0.064, Priya–Marcus 0.046
  (same-speaker 1.0). The match thresholds sit far below the
  clean-cluster scores (0.92–0.96, margins 0.75–0.84), which is
  why they do not separate on this fixture.

## Results

| Sherpa threshold | Turns | Clusters (vs 3) | Turn accuracy | Cluster purity | Unscored short turns |
|---|---|---|---|---|---|
| 0.4 | 13 | 6 | 0.50 (4/8 scored) | 0.91 (10/11) | 2 |
| 0.5 | 9 | 3 | 1.00 (9/9) | 1.00 (9/9) | 0 |
| 0.6 | 9 | 3 | 1.00 (9/9) | 1.00 (9/9) | 0 |

0.5 and 0.6 produced byte-identical turn lists. At 0.4 the
pipeline over-segments: Marcus's third turn is split into three
clusters (including two sub-0.5 s fragments and one turn that
overlaps its neighbour), and a John/Marcus fragment cluster
appears that no threshold can rescue — see below.

Match step, per sherpa threshold (clusters correct / unknown / wrong):

| Sherpa threshold | Match 0.45 | Match 0.55 | Match 0.65 |
|---|---|---|---|
| 0.4 | 4 / 0 / 1 (of 5 scored) | 4 / 0 / 1 | 4 / 0 / 1 |
| 0.5 | 3 / 0 / 0 (of 3) | 3 / 0 / 0 | 3 / 0 / 0 |
| 0.6 | 3 / 0 / 0 (of 3) | 3 / 0 / 0 | 3 / 0 / 0 |

At every sherpa threshold the three match thresholds gave
identical verdicts. The one wrong cluster at sherpa 0.4
(`SPEAKER_04`, mostly Marcus audio with a John fragment) matched
John at 0.856 with margin 0.65 — a mixed cluster, not a threshold
problem: the mixed-cluster guard, not a lower threshold, is what
keeps such a cluster out of a profile. The clean clusters scored
0.923–0.956 with margins 0.745–0.839, so no match threshold in
{0.45, 0.55, 0.65} came near flipping any of them.

Wall time per sherpa run was 48–53 s for the 61 s fixture,
including pipeline/model loading each run; it is not a
real-time-factor measurement.

## Decision

**No default changed.** The current defaults — sherpa clustering
threshold 0.5 (`SHERPA_DEFAULT_CLUSTERING_THRESHOLD`) and ECAPA
match threshold 0.55 (`SPEAKER_MATCH_THRESHOLDS["onnx-ecapa-v1"]`)
— stay. No setting won clearly on all three metrics: 0.5 and 0.6
tie exactly on every metric, 0.4 is worse on all three, and the
match thresholds are indistinguishable on clean clusters. Moving
a default on this fixture alone would be fitting noise.

## Open gate

Real-meeting validation is still outstanding. This fixture is
clean TTS with long turns and wide voice separation; noisy rooms,
short turns, overlapping speech and similar voices will narrow
every margin above. Re-run this sweep on real meeting audio
before trusting either default blindly — in particular, check
whether 0.6 holds up when turns are short, and whether the match
threshold still separates voices that sound alike. The original
YouTube sessions that motivated the label fix are gone, so that
validation needs a fresh real recording.
