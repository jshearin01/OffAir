# Security Policy

OffAir records, transcribes, and stores meeting audio and transcripts
locally. A vulnerability in that handling can expose private
conversations, so please report security issues privately — do not open
a public issue with exploit details.

Use GitHub's **private vulnerability reporting** for this repository
(Security tab → "Report a vulnerability"). Describe what you found, how
to reproduce it, and what data you believe it exposes. If private
vulnerability reporting is not enabled on the repository, open a minimal
issue that says only "security report pending" with no details, and the
maintainer will follow up.

The project will acknowledge reports and fix confirmed issues before
disclosing them. There is no bug-bounty program.

## Scope notes

- OffAir never uploads audio, transcripts, or embeddings. Any network
  traffic from the core recording/transcription/diarization paths is a
  bug — report it as above.
- The optional enterprise-LLM cleanup sends transcript *text* (never
  audio) to a CLI the user configured. That is intended behavior when
  the user turns it on, not a vulnerability.
- The Hugging Face token used for the optional pyannote path is read
  from the environment or entered in the Hugging Face token field
  on the Session view (Diarization group) and is never written to
  disk by OffAir. Do not include a real token in any report.
