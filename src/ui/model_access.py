"""Model access (D-025): `model.py` is the single source of truth for model code and
calibration data. Everything here is immutable after import.

D-085: the model is a plain module now — the notebook-parsing loader is gone, so this is an
ordinary import and Python's module cache does the "once per process" work that
`st.cache_resource` used to.
"""
import model as m

P0 = m.Params()  # model defaults — the single source of truth for every control's initial value
TDEF = m.target_defaults()      # exact forward images of the model defaults (round-trip exact)
_PARAM_TO_TARGET = {v: k for k, v in m.TARGET_PARAM.items()}
_PARAM_TO_TARGET.update({"delta_dev": "t_lag_mo", "delta_rel": "t_lag_mo"})
