"""Design system for the OffAir PyQt6 UI — light default, dark too.

Consumer-grade visual design (not a reflow): one considered palette
per theme, a real type scale, generous spacing, rounded cards and
controls, and a single accent colour used sparingly. Light is the
default — clean and airy — and dark is an equally finished
alternative, not an afterthought.

Palettes
--------
Light: warm off-white background (#F7F6F3), white cards, warm-gray
borders, near-black text, indigo accent (#4F46E5).
Dark: layered near-black grays (#17171B background), same indigo
accent family (#6366F1 primary, #818CF8 text accents).

The stylesheets are plain text so they can be unit-tested without
Qt; :func:`apply_theme` needs a Qt widget (or QApplication) and
raises a clear RuntimeError when PyQt6 is absent — importing this
module never requires Qt.

Design tokens (Nov 10 2026 — the "ugly UI" fix, John's report:
mixed rounded-vs-square cards, buttons overlapping, text hard
against box edges). One radius per family, everywhere:

* ``CARD_RADIUS = 12`` — cards (``QFrame#card``) and card-like
  containers (``QGroupBox``). No per-widget exceptions.
* ``CONTROL_RADIUS = 8`` — buttons, inputs, combos, lists and
  list rows, tooltips. The Record button is a control too.
* ``PILL_RADIUS = 10`` — the one deliberate exception: the tiny
  status-pill / callout family (status pill, capture-mode callout,
  AUTO badge). They are pill-shaped labels, not controls, and
  share this radius among themselves.
* Thin chrome is fully rounded by construction — radius = half
  its thickness (progress bar 6 px tall → 3 px; scrollbar handle
  8 px wide → 4 px). That is the same rule, not another exception.

Padding scale (applied in the layouts, QSS cannot set it):
outer window margin 16, card padding 16, tab content 16,
12 between sections, 6 inside the Record card only (measured —
the card is full at 800 px height; see main_window.py).
"""
from __future__ import annotations

# One accent, used sparingly: primary actions, active states, focus.
ACCENT_LIGHT = "#4F46E5"
ACCENT_DARK = "#6366F1"

# Corner radii — see the module docstring. One value per family.
CARD_RADIUS = 12
CONTROL_RADIUS = 8
PILL_RADIUS = 10

# Token dicts — the QSS template below is shared so both themes stay
# structurally identical (same selectors, same spacing, same radii);
# only the colours change.
_LIGHT = {
    "bg": "#F7F6F3",
    "surface": "#FFFFFF",
    "surface2": "#F1EFEA",
    "border": "#E6E2DA",
    "border_strong": "#D5D0C6",
    "text": "#1C1917",
    "text2": "#57534E",
    "text3": "#A8A29E",
    "accent": "#4F46E5",
    "accent_hover": "#4338CA",
    "accent_pressed": "#3730A3",
    "accent_soft": "#EEF2FF",
    "accent_soft_border": "#C7D2FE",
    "accent_soft_text": "#3730A3",
    "red": "#DC2626",
    "red_hover": "#B91C1C",
    "red_pressed": "#991B1B",
    "red_soft": "#FEE2E2",
    "red_soft_hover": "#FECACA",
    "red_soft_border": "#FECACA",
    "red_text": "#B91C1C",
    "amber_soft": "#FEF3C7",
    "amber_soft_border": "#FDE68A",
    "amber_text": "#92400E",
    "green": "#16A34A",
    "amber": "#D97706",
    "scroll_handle": "#D6D3D1",
    "scroll_handle_hover": "#A8A29E",
    "tooltip_bg": "#292524",
    "tooltip_text": "#FAFAF9",
}

_DARK = {
    "bg": "#17171B",
    "surface": "#202027",
    "surface2": "#2A2A32",
    "border": "#2E2E38",
    "border_strong": "#3F3F4C",
    "text": "#F4F4F5",
    "text2": "#A1A1AA",
    "text3": "#71717A",
    "accent": "#818CF8",
    "accent_hover": "#A5B4FC",
    "accent_pressed": "#6366F1",
    "accent_soft": "#2B2B40",
    "accent_soft_border": "#4338CA",
    "accent_soft_text": "#C7D2FE",
    "red": "#EF4444",
    "red_hover": "#F87171",
    "red_pressed": "#DC2626",
    "red_soft": "#3A1D1D",
    "red_soft_hover": "#4A2424",
    "red_soft_border": "#7F1D1D",
    "red_text": "#FCA5A5",
    "amber_soft": "#3A2E12",
    "amber_soft_border": "#713F12",
    "amber_text": "#FCD34D",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "scroll_handle": "#3F3F4C",
    "scroll_handle_hover": "#52525B",
    "tooltip_bg": "#26262C",
    "tooltip_text": "#F4F4F5",
}

# Primary-button background: indigo in both themes (light: #4F46E5,
# dark: #6366F1 so white text stays legible on the dark surface).
_PRIMARY_BG = {"light": "#4F46E5", "dark": "#6366F1"}
_PRIMARY_HOVER = {"light": "#4338CA", "dark": "#818CF8"}

_QSS_TEMPLATE = """
/* Base — one type scale, generous rhythm everywhere. */
QWidget {{ background-color: {bg}; color: {text}; font-size: 13px; }}
QMainWindow {{ background-color: {bg}; }}
QDialog {{ background-color: {bg}; }}

/* Typography: title 22 semibold, page title 18 semibold, section
   labels 12 semibold in secondary colour, timer 32 in monospace.
   Labels are transparent by default — the base QWidget rule used
   to paint every label with the page background, which showed as
   grey strips on white cards. */
QLabel {{ background-color: transparent; color: {text}; }}
QLabel#appTitle {{ font-size: 22px; font-weight: 600; color: {text}; }}
QLabel#appSubtitle {{ color: {text2}; font-size: 12px; }}
QLabel#pageTitle {{ font-size: 18px; font-weight: 600; color: {text}; }}
QLabel#timerLabel {{ font-size: 32px; font-weight: 600; color: {text};
    font-family: monospace; }}
QLabel#timerLabel[recording="true"] {{ color: {red}; }}
QLabel#sectionTitle {{ font-size: 12px; font-weight: 600; color: {text2}; }}
QLabel#fieldLabel {{ font-size: 12px; font-weight: 600; color: {text2}; }}
QLabel#hint {{ color: {text3}; font-size: 12px; }}

/* Loaded-session line on the Session view — which session the
   view is showing. Semibold secondary text, no box: it is a
   label, not a card. */
QLabel#loadedSessionLabel {{ color: {text2}; font-size: 12px;
    font-weight: 600; padding: 2px 4px; }}

/* Capture-mode callout on the Record card — a proper soft box
   (the old grey strip was just the page background showing
   through the label). */
QLabel#modeLabel {{ background-color: {surface2}; color: {text2};
    border: 1px solid {border}; border-radius: {pill_radius}px;
    padding: 6px 12px; font-size: 12px; }}

/* Status pill — quiet by default; red while recording, amber paused. */
QLabel#statusPill {{ background-color: {surface2}; color: {text2};
    border: 1px solid {border}; border-radius: {pill_radius}px;
    padding: 6px 12px; font-size: 12px; }}
QLabel#statusPillRecording {{ background-color: {red_soft};
    color: {red_text}; border: 1px solid {red_soft_border};
    border-radius: {pill_radius}px; padding: 6px 12px; font-size: 12px;
    font-weight: 600; }}
QLabel#statusPillPaused {{ background-color: {amber_soft};
    color: {amber_text}; border: 1px solid {amber_soft_border};
    border-radius: {pill_radius}px; padding: 6px 12px; font-size: 12px;
    font-weight: 600; }}

/* AUTO badge — shown on the Record card while an auto-started
   recording runs. Same red as the recording pill: unmissable. */
QLabel#autoBadge {{ background-color: {red}; color: #ffffff;
    border-radius: {pill_radius}px; padding: 2px 8px; font-size: 11px;
    font-weight: 700; }}

/* Buttons — quiet by default; one accent, used sparingly. */
QPushButton {{ background-color: {surface}; color: {text};
    border: 1px solid {border_strong}; border-radius: {control_radius}px;
    padding: 9px 16px; font-weight: 500; }}
QPushButton:hover {{ background-color: {surface2}; }}
QPushButton:pressed {{ background-color: {surface2}; }}
QPushButton:disabled {{ color: {text3}; background-color: {surface};
    border-color: {border}; }}
QPushButton#primaryButton {{ background-color: {primary_bg};
    color: #ffffff; border: none; font-weight: 600; }}
QPushButton#primaryButton:hover {{ background-color: {primary_hover}; }}
QPushButton#primaryButton:pressed {{
    background-color: {accent_pressed}; }}
QPushButton#primaryButton:disabled {{ background-color: {surface2};
    color: {text3}; }}
QPushButton#recordButton {{ background-color: {red}; color: #ffffff;
    border: none; font-weight: 600; font-size: 15px;
    padding: 12px 14px; border-radius: {control_radius}px; }}
QPushButton#recordButton:hover {{ background-color: {red_hover}; }}
QPushButton#recordButton:pressed {{
    background-color: {red_pressed}; }}
QPushButton#recordButton:disabled {{ background-color: {surface2};
    color: {text3}; }}
QPushButton#dangerButton {{ background-color: {red_soft};
    color: {red_text}; border: 1px solid {red_soft_border};
    font-weight: 600; }}
QPushButton#dangerButton:hover {{ background-color: {red_soft_hover}; }}
QPushButton#dangerButton:pressed {{
    background-color: {red_soft_hover}; }}

/* Inputs — white/surface fields, accent border on focus. */
QLineEdit, QComboBox, QListWidget, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {surface}; color: {text};
    border: 1px solid {border_strong}; border-radius: {control_radius}px;
    padding: 8px 10px;
    selection-background-color: {accent};
    selection-color: #ffffff; }}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus {{ border: 1px solid {accent}; }}
QLineEdit:disabled, QComboBox:disabled, QTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {text3}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background-color: {surface};
    color: {text}; border: 1px solid {border};
    selection-background-color: {accent};
    selection-color: #ffffff; }}

/* Lists — roomy rows, accent selection, soft hover. */
QListWidget::item {{ padding: 8px 12px; border-radius: {control_radius}px; }}
QListWidget::item:selected {{ background-color: {accent};
    color: #ffffff; }}
QListWidget::item:hover {{ background-color: {surface2}; }}
QListWidget::item:selected:hover {{ background-color: {accent}; }}
QTextEdit {{ line-height: 1.4; }}

/* Progress — slim accent bar, no percentage text. 3 px radius is
   half the 6 px height: fully rounded by construction (see the
   module docstring — thin chrome, not a fourth radius family). */
QProgressBar {{ background-color: {surface2}; border: none;
    border-radius: 3px; height: 6px; text-align: center;
    color: {text}; }}
QProgressBar::chunk {{ background-color: {accent}; border-radius: 3px; }}

QStatusBar {{ background-color: {bg}; color: {text2}; font-size: 12px;
    border-top: 1px solid {border}; }}
QStatusBar::item {{ border: none; }}
QCheckBox {{ color: {text}; spacing: 8px; }}

/* Cards and structure — rounded, 1px borders, no heavy shadows. */
QFrame#card {{ background-color: {surface}; border: 1px solid {border};
    border-radius: {card_radius}px; }}

/* Calendar badges — one recording per badge on the month grid.
   Soft accent fill (same family as the selected list row, not the
   loud primary button): the grid can hold many of them. Radius is
   the control token (8 px), like every other small control. */
QFrame#calendarBadge {{ background-color: {accent_soft};
    border: 1px solid {accent_soft_border};
    border-radius: {control_radius}px; }}
QFrame#calendarBadge:hover {{ border-color: {accent}; }}
QLabel#calendarBadgeText {{ color: {accent_soft_text}; font-size: 12px; }}
QFrame#headerBar {{ background-color: transparent; border: none;
    border-bottom: 1px solid {border}; }}
QGroupBox {{ border: 1px solid {border}; border-radius: {card_radius}px;
    margin-top: 10px; padding: 12px; padding-top: 18px;
    font-weight: 600; background-color: {surface}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px;
    padding: 0 6px; color: {text2}; font-size: 12px;
    background-color: {surface}; }}

/* Tabs — underline indicator, no boxed pane. (Kept for the
   Recording Browser / any future tabbed window; the main window
   uses left navigation since the Nov 19 2026 redesign.) */
QTabWidget::pane {{ border: none; background-color: transparent; }}

/* Left navigation — the main window's view switcher (Nov 19 2026).
   Plain rows, the selected one in the soft accent fill, radius
   the control token (8 px) like every other small control. */
QListWidget#navList {{ background-color: transparent; border: none;
    outline: none; font-size: 14px; }}
QListWidget#navList::item {{ color: {text2}; border-radius:
    {control_radius}px; padding: 9px 12px; margin-bottom: 2px; }}
QListWidget#navList::item:selected {{ color: {text}; font-weight: 600;
    background-color: {accent_soft}; }}
QListWidget#navList::item:hover {{ color: {text}; }}
QLabel#navSection {{ color: {text2}; font-size: 11px; font-weight: 600;
    padding: 10px 12px 4px 12px; }}
QTabBar::tab {{ background-color: transparent; color: {text2};
    border: none; border-bottom: 2px solid transparent;
    padding: 12px 18px; margin-right: 4px; font-weight: 500; }}
QTabBar::tab:selected {{ color: {text}; font-weight: 600;
    border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}

QSplitter::handle {{ background-color: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}

/* Scrollbars — slim, rounded, quiet. 4 px radius is half the 8 px
   width: fully rounded by construction (thin chrome, see above). */
QScrollArea {{ border: none; background-color: transparent; }}
QScrollBar:vertical {{ background-color: transparent; width: 8px;
    border: none; margin: 2px 0; }}
QScrollBar::handle:vertical {{ background-color: {scroll_handle};
    border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{
    background-color: {scroll_handle_hover}; }}
QScrollBar:horizontal {{ background-color: transparent; height: 8px;
    border: none; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background-color: {scroll_handle};
    border-radius: 4px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{
    background-color: {scroll_handle_hover}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0px; width: 0px; }}

QToolTip {{ background-color: {tooltip_bg}; color: {tooltip_text};
    border: 1px solid {border_strong}; border-radius: {control_radius}px;
    padding: 6px 8px; }}
"""


def _build(theme: str) -> str:
    tokens = dict(_LIGHT if theme == "light" else _DARK)
    tokens["primary_bg"] = _PRIMARY_BG[theme]
    tokens["primary_hover"] = _PRIMARY_HOVER[theme]
    tokens["card_radius"] = CARD_RADIUS
    tokens["control_radius"] = CONTROL_RADIUS
    tokens["pill_radius"] = PILL_RADIUS
    return _QSS_TEMPLATE.format(**tokens)


LIGHT_QSS = _build("light")
DARK_QSS = _build("dark")

THEMES: dict[str, str] = {"dark": DARK_QSS, "light": LIGHT_QSS}
# Light is the default — the clean, airy consumer look. (Must match
# DEFAULT_THEME in app/config.py; tests assert they agree.)
DEFAULT_THEME = "light"


def available_themes() -> list[str]:
    return sorted(THEMES)


def stylesheet_for(theme: str) -> str:
    """The QSS text for ``theme``. Raises ValueError for unknown themes."""
    if theme not in THEMES:
        raise ValueError(
            f"Unknown theme {theme!r}; supported: {sorted(THEMES)}")
    return THEMES[theme]


def apply_theme(target, theme: str) -> str:
    """Apply ``theme`` to a Qt widget/app (anything with setStyleSheet).

    Returns the stylesheet applied. Raises ValueError for an unknown
    theme and RuntimeError when the target has no setStyleSheet
    (e.g. PyQt6 is not installed).
    """
    qss = stylesheet_for(theme)
    setter = getattr(target, "setStyleSheet", None)
    if setter is None:
        raise RuntimeError(
            "Cannot apply a theme: this object has no setStyleSheet "
            "(PyQt6 is not installed?)")
    setter(qss)
    return qss


def apply_theme_to_app(app, theme: str) -> str:
    """Apply ``theme`` at the QApplication level (Phase 6 fix).

    A stylesheet set on the QApplication cascades to EVERY window
    the app opens — the main window, the Recording Browser and the
    Settings dialog — which fixes the Phase 5 bug where the dark
    theme was applied to the main window only and the other
    windows rendered in the default light style. ``main.py`` calls
    this once at startup; ``apply_theme`` on a single window is
    still available for standalone use (screenshots, tests).
    """
    return apply_theme(app, theme)
