"""Learn lime to explain images.

This module is to implement of LIME method from scratch.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

from abc import ABC
from skimage import segmentation
from sklearn.linear_model import Ridge
from skimage.measure import compare_ssim


class LIMEAbstract(ABC):
    """Abstract module of LIME which include all methods needs to implemented."""

    def __init__(self, x,
                 predict_fn=None,
                 n_features=20,
                 sample_size=100,
                 ):
        self.x = x
        self.predict_fn = predict_fn
        self.n_features = n_features
        self.sample_size = sample_size
        self.logger = logging.getLogger(self.__class__.__name__)

        self.x_segmented = None
        self.x_features = None

    def pi(self, z):
        pass

    def to_features(self):
        pass

    def sampling_features(self, *args, **kwargs):
        pass

    def neighbors(self, **kwargs):
        pass

    def explain(self, *args, **kwargs):
        pass

    def viz_explain(self, *args, **kwargs):
        pass


class LIMEImage(LIMEAbstract):

    def pi(self, z):
        """Return weight (distance or similarity) of instance z to x.
        :param z: generated from z' which has same presentation to x.

        return: Pi of z to x ~ weight of z to x"""
        # because the function of distance is already return similarity of z, hence

        x = self.x
        assert x.shape == z.shape

        multichannel = (len(x.shape) == 3)
        similarity = compare_ssim(x, z, multichannel=multichannel)
        return similarity

    def to_features(self):
        """Segment instance x.

        :return: a vector of unique segments (or said, features)
        """
        x = self.x

        # A 2d matrix which pixel belongs to which segment
        multichannel = (x.ndim == 3)
        x_segmented = segmentation.slic(x, 10, multichannel=multichannel)

        # extract the segment number out of the matrix -> a vector of segment categories (number)
        #   this is used later to turn on or off which segments and for masking purpose.
        features = np.unique(x_segmented)
        return x_segmented, features

    def to_original_form(self, z_comma):
        """Convert back to original format with mask."""
        x = self.x
        x_segmented = self.x_segmented
        features = self.x_features

        # Which feature currently available for sample z'
        features_on = features[z_comma]

        # mask2d of z' ~ corresponding to pixels of instance x
        # remember, mask2d is for segments, mask3d is for RGB image
        mask = np.isin(x_segmented, features_on).astype(int)
        _shape = self.to_dim(x, x_segmented)
        mask = mask.reshape(_shape)

        # Get z as same original presentation of instance x
        z_original = x * mask
        return z_original

    @staticmethod
    def to_dim(x, y):
        """Convert from dimension of y to x.

        It does not necessary to have same shape, but to enable broadcasting capability.
        """
        shape = x.ndim
        if y.ndim != x.ndim:
            # convert shape of x_segmented -> original dimension
            # which dimension does not match e.g (True, True, False)
            #   -> converted from (299, 299) to (299, 299, 1). Original form has (299, 299, 3)
            #   then now, it could be broadcasted.
            not_matched_shape = np.isin(x.shape, y.shape)
            shape = not_matched_shape * x.shape + np.bitwise_not(not_matched_shape)
        return shape

    def sampling_features(self, features, n_features_on=None):
        """Draw z_comma from feature vector.

        A z_comma is a feature vector which is represented as 0s and 1s (on/off features).
        :param features: unique features which could be label or number of unique features/categories.
        :param n_features_on: how many features want to be drawn from vectors,
            if None then random from 0 to len of features

        """
        sample_size = self.sample_size

        # number of vectors in feature vectors will be on/off ~ present/absent
        n_features = min(self.n_features, len(features))
        n_features_on = n_features_on or np.random.randint(0, n_features)

        # probability of on and off for 1 feature
        prob_features_on = n_features_on / n_features
        prob_features_off = 1 - prob_features_on

        # random draw a z', an on/off vector of 0s and 1s from feature vector
        z_comma = np.random.choice(a=[1, 0],
                                   size=(sample_size, n_features),
                                   p=[prob_features_on, prob_features_off])
        return z_comma

    def neighbors(self, n_features_on=None, sample_size=None, *args):

        # predict_fn = self.predict
        predict_fn = self.predict_fn
        sample_size = sample_size or self.sample_size

        x_segmented, features = self.to_features()
        self.x_segmented = x_segmented
        self.x_features = features
        self.n_features = min(self.n_features, len(features))

        z_comma = self.sampling_features(self.x_features, n_features_on)

        # Todo: Convert to use numpy apply instead of loop for
        samples_set = []
        # Create data set neighbors of X based on X'
        for i in range(sample_size):
            # Convert z_comma to z which has original form and shape with x
            z_original = self.to_original_form(z_comma[i])
            y_hat = predict_fn(z_original, *args)

            # We record feature vector and result of predicted function
            sample = (z_comma[i], y_hat)

            # list of tuples will be return (size * tuples)
            samples_set.append(sample)
        return samples_set

    def explain(self, x=None):
        if x is not None:
            self.x = x

        # get a sample is tuple of (z_comma[i], y_hat)
        #   z_comma is input, y_hat as output/target
        samples_set = self.neighbors()
        z_comma = [i[0] for i in samples_set]
        target = [i[1] for i in samples_set]

        # Define a local explain mode g(z') = w * z'
        linear_xai = Ridge()
        linear_xai.fit(z_comma, target)

        self.logger.info(linear_xai.coef_)
        return linear_xai

    @staticmethod
    def viz_coefficient(linear_xai):
        _coef = linear_xai.coef_
        plt.plot(_coef, alpha=0.7, linestyle='none', marker='*', markersize=5)
        plt.show()

    @staticmethod
    def viz_explain(image_dict, idx=0, title=None):
        fig = plt.figure()
        fig.suptitle(title)
        idx = 330
        for k, v in image_dict.items():
            idx += 1
            ax = plt.subplot(idx)
            ax.set_title(k)
            ax.imshow(v)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.show()

    def create_demo_explanation(self, linear_xai, top_k=None):
        """Create demo for explain image."""
        x_original = self.x
        x_segmented = self.x_segmented
        features = self.x_features
        _shape = self.to_dim(x_original, x_segmented)

        sorted_idx = np.argsort(linear_xai.coef_)
        top_k = top_k or len(sorted_idx)
        top_k_idx = sorted_idx[-top_k:]

        # Get mask for presenting only top-k features
        k_features = features[top_k_idx]
        k_features_mask = np.isin(x_segmented, k_features).astype(int)

        # x presented as top features which have highest contributions to model
        xai_segmented = x_segmented * k_features_mask
        xai_original = x_original * k_features_mask.reshape(_shape)

        # visualize how lime convert from x -> x'-> z'-> z -> explain
        # Make small sample set only to visualize
        n_features_on = int(self.n_features / 2)
        sample_size = top_k * 10
        samples_set = self.neighbors(n_features_on=n_features_on, sample_size=sample_size)

        # Get one sample to visualize z_comma and z_original
        rand_sample = np.random.randint(0, len(samples_set))
        sample = samples_set[rand_sample]
        z_comma, _ = sample

        # get image of z in original, segments, and superpixel
        z_original = self.to_original_form(z_comma)

        viz_images_dict = {
            "Original": x_original,
            "Features": x_segmented,
            "A sample z drawn from features": z_original,
            f"Top {top_k} features": xai_segmented,
            f"Top {top_k} features in original form": xai_original,

        }
        return viz_images_dict
