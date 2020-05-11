"""Test module for explanation ai implemented with LIME"""
import os
import pytest

from skimage.io import imread
from skimage.transform import resize as imresize

from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions

from temporalnn.explain.lime import LIMEImage

inet_model = inc_net.InceptionV3()

DATA_DIR = "tests/data" if os.path.isdir("tests") else "data"


def predict_inception_v3(x):
    shape = None
    if x.ndim == 3:
        shape = (1, x.shape[0], x.shape[1], x.shape[2])
    if x.ndim == 2:
        shape = (1, x.shape[0], x.shape[1], 1)
    if not shape:
        raise ValueError(f"{x.shape}")
    _x = x.reshape(shape)
    prediction = inet_model.predict(_x)
    _, label, prob = decode_predictions(prediction)[0][0]
    return prob


def test_explains_segmentation_and_features():
    # img = imread("tests/data/cat.jpg")
    img = imread(f"{DATA_DIR}/cat.jpg")
    img_original = imresize(img, (299, 299))

    xai_img = LIMEImage(x=img_original, predict_fn=predict_inception_v3)
    img_segmented, features = xai_img.to_features()
    w, h, _ = img_original.shape
    assert img_segmented.shape == (w, h)


def test_explains_z_comma_set():
    img = imresize(imread(f"{DATA_DIR}/cat.jpg"), (299, 299))
    img_original = imresize(img, (299, 299))

    xai_img = LIMEImage(x=img_original)
    img_segmented, features = xai_img.to_features()
    z_comma = xai_img.sampling_features(features, n_features_on=4)
    assert len(z_comma[1]) == len(features)


def test_convert_from_z_comma_to_z_original_form():
    img = imresize(imread(f"{DATA_DIR}/cat.jpg"), (299, 299))

    xai_img = LIMEImage(x=img)
    img_segmented, features = xai_img.to_features()
    xai_img.x_segmented = img_segmented
    xai_img.x_features = features

    z_comma = xai_img.sampling_features(features, n_features_on=4)
    assert len(z_comma[1]) == len(features)

    t = z_comma[1].astype(int)
    z_original = xai_img.to_original_form(t)
    assert z_original.shape == img.shape


def test_generate_sample_set():
    img = imread(f"{DATA_DIR}/cat.jpg")
    img = imresize(img, (299, 299))

    xai_img = LIMEImage(x=img, predict_fn=predict_inception_v3)
    samples_set = xai_img.neighbors()

    t = samples_set[0]
    assert len(t) == 2


def test_explain_model():
    img = imread(f"{DATA_DIR}/cat.jpg")
    img = imresize(img, (299, 299))

    xai_img = LIMEImage(x=img, predict_fn=predict_inception_v3)
    xai_model = xai_img.explain()
    assert xai_model.coef_ is not None


@pytest.mark.skip("Interactive, not automatic possible.")
def test_viz_result():
    img = imread(f"{DATA_DIR}/cat.jpg")
    img = imresize(img, (299, 299))

    xai_img = LIMEImage(x=img, predict_fn=predict_inception_v3, n_features=10)
    xai_model = xai_img.explain()

    xai_img.viz_coefficient(xai_model)
    xai_img_dict = xai_img.create_demo_explanation(xai_model, top_k=3)
    xai_img.viz_explain(xai_img_dict, )
