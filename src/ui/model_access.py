"""Model access (D-025): the notebook is the single source of truth for model code and
calibration data. Loaded once per process; everything here is immutable after load.
"""
from pathlib import Path

import streamlit as st

import notebook_loader

NB = str(Path(__file__).resolve().parent.parent / "model_notebook.ipynb")


@st.cache_resource
def get_model():
    """Load all model functions from the notebook once per session."""
    return notebook_loader.load_model(NB)


m = get_model()

P0 = m.Params()  # notebook defaults — the single source of truth for every control's initial value
TDEF = m.target_defaults()      # exact forward images of the notebook defaults (round-trip exact)
_PARAM_TO_TARGET = {v: k for k, v in m.TARGET_PARAM.items()}
_PARAM_TO_TARGET.update({"delta_dev": "t_lag_mo", "delta_rel": "t_lag_mo"})
