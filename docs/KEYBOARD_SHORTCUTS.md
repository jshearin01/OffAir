# Keyboard shortcuts

Shortcuts work inside OffAir only — they are not global hotkeys, so
they do nothing while another app is in front. No new dependencies
are involved; they are plain Qt shortcuts on the main window.

| Shortcut | What it does |
|---|---|
| Ctrl+B | Bookmark. While recording, marks the current moment in the recording. With a session loaded and stopped, marks the loaded session at the moment now playing, the selected segment, or the start of the session. Otherwise it does nothing. |
| Ctrl+P | Pause / Resume. Pauses while recording, resumes while paused. Otherwise it does nothing. |
| Ctrl+K | Jump to search. Opens the Library view and puts the cursor in its search box, with the current text selected. |
| Delete | Delete the selected Library session. A confirmation names the session and lists what is removed; Cancel is the default. Does nothing when the Library list has no selection, or when the search box has the focus. |

The same actions are always available as buttons: Bookmark sits on
its own row below Pause / Resume / Stop in the Record view
while recording, and "Add bookmark at current moment" sits under
the bookmark list on the Session view. Nothing is shortcut-only.

Related: [Overview](OVERVIEW.md) (bookmarks, dictation, pins, and
the waveform), [Retention](RETENTION.md) (bookmarks whose audio was
deleted render disabled, with the reason).
