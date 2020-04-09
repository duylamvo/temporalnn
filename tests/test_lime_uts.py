"""Test module for explanation ai implemented with LIME"""
import pytest
import json
import numpy as np
import pandas as pd

from temporalnn.explain.lime import LIMETimeSeries
from keras.models import load_model

DATA_DIR = "data"


def preload_uts(data_dir=DATA_DIR):
    df = pd.read_csv(f"{data_dir}/climate_small.csv")
    with open(f"{data_dir}/climate_small_schema.json") as f:
        schema = json.load(f)
        df = df.astype(schema)

    df = df.set_index("measure_date")
    tmk = df.query("stations_id == 2074")["tmk"]

    input_steps = 32
    return tmk[-input_steps:].to_numpy()


# Todo - use preload of pytest instead
ts_original = preload_uts()
model = load_model(f"{DATA_DIR}/uts_tmk_32_1.h5")


def predict_uts_tmk(x, *args):
    _shape = x.shape
    if len(_shape) == 2:
        _shape = (1, _shape[0], _shape[1])
    elif len(_shape) == 1:
        _shape = (1, _shape[0], 1)
    x = x.reshape(_shape)

    y_hat = model.predict(x, *args)

    # flatten to one value
    return y_hat.ravel().item()


def test_segments_and_convert_to_features():
    # Get a last time series with matching with model,
    #   here example use uts_32_1 with input 32 steps, and output 1 step.

    xai_uts = LIMETimeSeries(x=ts_original)
    ts_segmented, features = xai_uts.to_features()

    assert ts_segmented.ndim == 2 and ts_segmented.ravel().shape == ts_original.shape


def test_convert_to_original_form():
    xai_uts = LIMETimeSeries(x=ts_original)
    ts_segmented, features = xai_uts.to_features()
    xai_uts.x_segmented = ts_segmented
    xai_uts.x_features = features

    # In regression, 0 will make data become 0, nan is better option
    z_comma = np.random.choice(a=[1, np.nan], size=len(features))
    z_original = xai_uts.to_original_form(z_comma)

    assert z_original.shape == ts_original.shape


def test_distance_and_pi_function():
    pass


def test_sampling_from_features():
    xai_uts = LIMETimeSeries(x=ts_original)
    ts_segmented, features = xai_uts.to_features()
    xai_uts.x_segmented = ts_segmented
    xai_uts.x_features = features

    z_comma = xai_uts.sampling_features(features, n_features_on=4, on_offs=[1, np.nan])
    assert len(z_comma[1]) == len(features)

    t = z_comma[1]
    z_original = xai_uts.to_original_form(t)
    assert z_original.shape == ts_original.shape


def test_generate_sample_set():
    xai_uts = LIMETimeSeries(x=ts_original, predict_fn=predict_uts_tmk)
    samples_set = xai_uts.neighbors(on_offs=[1, 0])

    t = samples_set[0]
    assert len(t) == 2

    z_comma, y_hat = t
    assert isinstance(z_comma, np.ndarray) and isinstance(y_hat, (float, int))


def test_explain_model():
    xai_uts = LIMETimeSeries(x=ts_original, predict_fn=predict_uts_tmk)
    xai_model = xai_uts.explain(on_offs=[1, 0])
    assert all([xai_model.coef_ is not None,
                len(xai_model.coef_) == len(xai_uts.x_features)]
               )


# @pytest.mark.skip("Interaction, hence skip.")
def test_viz_of_features():
    xai_uts = LIMETimeSeries(x=ts_original, predict_fn=predict_uts_tmk)
    xai_model = xai_uts.explain(on_offs=[1, 0])

    demo_dict = xai_uts.create_demo_explanation(xai_model, top_k=3)
    xai_uts.viz_explain(demo_dict)
