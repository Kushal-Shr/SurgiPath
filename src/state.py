"""
App state machine: PRE_OP, INTRA_OP, POST_OP.

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

This module is the only place that reads or writes the app "mode" and
intra-op "phase" in Streamlit session state. The rest of the app should
call get_mode(), set_mode(), get_phase(), set_phase() instead of touching
st.session_state["mode"] or st.session_state["phase"] directly.

  MODE:
  - PRE_OP: user is in the checklist; can click "Start Surgery" when checklist
    is stable.
  - INTRA_OP: surgery in progress; phase dropdown and rules run; "End Surgery"
    switches to POST_OP.
  - POST_OP: viewing analytics from the event log.

  PHASE:
  - One of: incision, suturing, irrigation, closing. Used by the rule engine
    (src.rules) to decide which rules apply and is logged in events.

  FUNCTIONS:
  - init_state(): call once at app startup; sets default mode and phase if
    not already in session_state.
  - set_mode(mode), get_mode(): write/read current mode.
  - set_phase(phase), get_phase(): write/read current phase.

  IMPLEMENTATION NOTE: Each function uses try/except and imports streamlit
  inside the function so that code that imports state.py does not break if
  Streamlit is not installed (e.g. in tests). The keys MODE_KEY and PHASE_KEY
  are the actual strings stored in session_state.
"""
from typing import Literal

AppMode = Literal["PRE_OP", "INTRA_OP", "POST_OP"]

MODE_KEY = "mode"
PHASE_KEY = "phase"
INITIAL_MODE: AppMode = "PRE_OP"
DEFAULT_PHASE = "incision"


def init_state() -> None:
    """Set default mode and phase in session state if not already set."""
    try:
        import streamlit as st
        if MODE_KEY not in st.session_state:
            st.session_state[MODE_KEY] = INITIAL_MODE
        if PHASE_KEY not in st.session_state:
            st.session_state[PHASE_KEY] = DEFAULT_PHASE
    except Exception:
        pass


def set_mode(mode: AppMode) -> None:
    """Set current app mode (PRE_OP, INTRA_OP, or POST_OP)."""
    try:
        import streamlit as st
        st.session_state[MODE_KEY] = mode
    except Exception:
        pass


def get_mode() -> AppMode:
    """Return current app mode."""
    try:
        import streamlit as st
        return st.session_state.get(MODE_KEY, INITIAL_MODE)
    except Exception:
        return INITIAL_MODE


def set_phase(phase: str) -> None:
    """Set current intra-op phase (e.g. incision, suturing)."""
    try:
        import streamlit as st
        st.session_state[PHASE_KEY] = phase
    except Exception:
        pass


def get_phase() -> str:
    """Return current intra-op phase."""
    try:
        import streamlit as st
        return st.session_state.get(PHASE_KEY, DEFAULT_PHASE)
    except Exception:
        return DEFAULT_PHASE
