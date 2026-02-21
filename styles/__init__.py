"""Styles package for Project Panacea surgical dashboard."""

from pathlib import Path

import streamlit as st

_STYLES_DIR = Path(__file__).parent


def load_css(filename: str = "surgical_theme.css") -> None:
    """Read a CSS file from the styles directory and inject it into the Streamlit page."""
    css_path = _STYLES_DIR / filename
    css_text = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
