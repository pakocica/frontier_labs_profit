"""Cached simulation wrappers (st.cache_data over the notebook functions).
"""
import numpy as np
import streamlit as st

from .model_access import m


def _items(d: dict):
    return tuple(sorted(d.items()))


@st.cache_data(show_spinner=False)
def sim_cached(params_items):
    p = m.Params(**dict(params_items))
    return m.simulate(p)


@st.cache_data(show_spinner=False)
def delay_cached(params_items, taus):
    p = m.Params(**dict(params_items))
    return m.delay_comparison(p, taus=taus)


@st.cache_data(show_spinner=False)
def sim_window_cached(params_items, t0, width, tau_mo):
    """Simulate a time-varying delay: tau(t) = tau_mo months only inside [t0, t0+width)."""
    p = m.Params(**dict(params_items))
    tau = tau_mo / 12.0

    def tau_fn(t):
        t = np.asarray(t, float)
        return np.where((t >= t0) & (t < t0 + width), tau, 0.0)

    return m.simulate(p, tau_fn=tau_fn)
