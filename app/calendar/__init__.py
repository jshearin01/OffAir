"""Calendar linking for OffAir sessions.

This package links a recorded session to the Outlook calendar event
it belongs to, so the Recording Browser / main window can say which
meeting a recording was. It is Windows-only (Outlook COM) and
**read-only by design** — see :mod:`app.calendar.outlook`.
"""
from __future__ import annotations
