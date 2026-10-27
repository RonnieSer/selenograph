"""Start the lunar mapping app."""
import streamlit as st

from app import core

st.set_page_config(page_title="Selenograph", layout="wide")

with core.catalog_snapshot():
    st.navigation([
        st.Page("app/app_pages/paint.py", title="Paint", icon=":material/brush:", default=True),
        st.Page("app/app_pages/gallery.py", title="Gallery", icon=":material/photo_library:"),
        st.Page("app/app_pages/progress.py", title="Progress", icon=":material/trending_up:"),
    ], position="top").run()
