"""EARTHSCOPE - entry point and page router.  Run with:  streamlit run app.py"""
import streamlit as st

from earthscope import config
from ui import components, state

st.set_page_config(page_title="EARTHSCOPE", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
state.init()

pages = [
    st.Page("views/overview.py", title="Earth Overview", icon="🌍", default=True),
    st.Page("views/investigation.py", title="Environmental Investigation", icon="🔎"),
    st.Page("views/fingerprint.py", title="Environmental Fingerprint", icon="🧬"),
    st.Page("views/comparison.py", title="Regional Comparison", icon="🗺️"),
    st.Page("views/anomalies.py", title="Anomalies and Timeline", icon="📈"),
    st.Page("views/report.py", title="Investigation Report", icon="📄"),
]
nav = st.navigation(pages)
components.sidebar_footer()
nav.run()
