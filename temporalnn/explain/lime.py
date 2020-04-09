"""Learn lime to explain images.

This module is to implement of LIME method from scratch.
"""
import logging
import numpy as np
import seglearn as sl
import matplotlib.pyplot as plt

from abc import ABC
from skimage import segmentation
from skimage.measure import compare_ssim
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances


def viz_lime_theory():
    from graphviz import Digraph

    dot = Digraph(comment='LIME')
    dot.node(name='X', label='Instance X')
    dot.node(name='X_segment', label='X - Segmentation')
    dot.node(name='X_comma', label="X'")
    dot.node(name='Z_comma', label="Sampling Neighbors Z'")
    dot.node(name='Z', label="Z")
    dot.node(name='F', label="f(Z)")
    dot.node(name='P', label="π(x,z)")
    dot.node(name='G', label="g(z') = W*Z'", color="turquoise", style="filled")
    dot.node(name='L', label="Loss of f(z) ~ g(z') with weight π")
    dot.node(name='O', label="Optimizer: argmin { loss }")
    dot.node(name='R', label="W^ ~ Weights of features", color="turquoise", style="filled")

    dot.edge('X', 'X_segment')
    dot.edge('X_segment', 'X_comma')
    dot.edge('X_comma', 'Z_comma')
    dot.edge('Z_comma', 'Z')
    dot.edge('Z', 'F')
    dot.edge("Z", "P", "Similarity of x to z")
    dot.edge('Z_comma', 'G')
    dot.edges(["FL", "GL", "PL", "LO", "OR"])

    c = Digraph(name="sampling_z", node_attr={"shape": "box", "style": "dashed"})
    c.node(name='S', label="Sampling Neighbors Z'")
    c.node(name='A', label="Z'")
    c.node(name='B', label="Z'")
    c.node(name='C', label="Z'")

    c.edges(["SB", "SC", "SA"])
    dot.subgraph(c)

    return dot.render("test-ouput/round-table.gv", view=True)


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

        convert shape of x_segmented -> original dimension
            which dimension does not match e.g (True, True, False)
        E.g -> converted from (299, 299) to (299, 299, 1). Original form has (299, 299, 3)
            then now, it could be broad-casted.
        """
        # Todo improve this
        shape = x.ndim
        if y.ndim < x.ndim:
            not_matched_shape = np.isin(x.shape, y.shape)
            shape = not_matched_shape * x.shape + np.bitwise_not(not_matched_shape)
        elif y.ndim > x.ndim:
            not_matched_shape = np.isin(y.shape, x.shape)
            shape = not_matched_shape * y.shape + np.bitwise_not(not_matched_shape)

        return shape

    def sampling_features(self, features, n_features_on=None, on_offs=None):
        """Draw z_comma from feature vector.

        A z_comma is a feature vector which is represented as 0s and 1s (on/off features).
        :param features: unique features which could be label or number of unique features/categories.
        :param n_features_on: how many features want to be drawn from vectors,
            if None then random from 0 to len of features
        :param on_offs: a vector of values corresponding to on and off bit.
            in image segmentation, it could be 0, 1s for turning off or on a feature.
            in regression it could be nan instead of 0 (due to 0 will have different meaning)

        """
        sample_size = self.sample_size

        # number of vectors in feature vectors will be on/off ~ present/absent
        n_features = min(self.n_features, len(features))
        n_features_on = n_features_on or np.random.randint(0, n_features)

        # probability of on and off for 1 feature
        prob_features_on = n_features_on / n_features
        prob_features_off = 1 - prob_features_on

        # random draw a z', an on/off vector of 0s and 1s from feature vector
        on_offs = on_offs or [1, 0]
        z_comma = np.random.choice(a=on_offs,
                                   size=(sample_size, n_features),
                                   p=[prob_features_on, prob_features_off])
        return z_comma

    def neighbors(self, n_features_on=None, sample_size=None, *args, **kwargs):

        # predict_fn = self.predict
        predict_fn = self.predict_fn
        sample_size = sample_size or self.sample_size

        x_segmented, features = self.to_features()
        self.x_segmented = x_segmented
        self.x_features = features
        self.n_features = min(self.n_features, len(features))

        z_comma = self.sampling_features(self.x_features, n_features_on, **kwargs)

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

    def explain(self, x=None, **kwargs):
        if x is not None:
            self.x = x

        # get a sample is tuple of (z_comma[i], y_hat)
        #   z_comma is input, y_hat as output/target
        samples_set = self.neighbors(**kwargs)
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


class LIMETimeSeries(LIMEImage):
    """Implementation of LIME in explain Temporal Model."""

    def __init__(self, x_steps=32, steps_per_segment=4, **kwargs):
        super().__init__(**kwargs)

        self.x_steps = x_steps
        self.steps_per_segment = steps_per_segment
        if self.steps_per_segment is None and self.n_features is not None:
            self.steps_per_segment = int(len(self.x) / self.n_features)

    def pi(self, z, gamma=0.001):
        """Apply gaussian radial basis function as kernel ~ RBF kernel

        :param z: z_original drawn and converted to original from features.
            it should have the same format with original
        :param gamma: exp(-(||x-y||² / (2*sigma²)) ~ exp(-gamma*||x-y||²))
            gamma = 1/2.sigma²
        """

        x = self.x
        assert x.shape == z.shape
        # RBF kernel (here, use (x-y)²
        d = pairwise_distances(x.reshape(1, -1), z.reshape(1, -1))
        pi = np.exp(-gamma * d)

        return pi.flatten().item()

    def to_features(self, overlap_rate=0.):
        x = self.x
        steps_per_segment = self.steps_per_segment

        assert steps_per_segment < len(x), "Steps per segment should smaller than len(x)"
        # Given an TS x as array
        seg = sl.transform.Segment(steps_per_segment, overlap=overlap_rate)
        seg.fit([x])

        # Overlap_rate r, then n_segments = [len(series) / (width * (1-r))] - 1
        #  if r = 0 ~ n_segments = len(series) / width
        # a x_segmented for uts will be an 2d-array
        x_segmented, _, _ = seg.transform([x])

        # Features now, is an array of index of the 2d-array
        assert x_segmented.ndim == 2
        n_features, steps_per_feature = x_segmented.shape
        features = np.array(range(0, n_features))
        # create a vector of index, which represents indices of a ts

        return x_segmented, features

    def to_original_form(self, z_comma):
        """Convert back to original format with mask."""
        x_segmented = self.x_segmented
        features = self.x_features

        _shape = (len(features), 1)
        mask = z_comma.reshape(_shape)

        # Get z as same original presentation of instance x
        z_original = x_segmented * mask

        return z_original.ravel()

    def create_demo_explanation(self, linear_xai, top_k=None):
        """Create demo for explain image."""
        x_original = self.x
        x_segmented = self.x_segmented
        features = self.x_features
        n_features = len(features)

        sorted_idx = np.argsort(linear_xai.coef_)
        top_k = top_k or len(sorted_idx)
        top_k_idx = sorted_idx[-top_k:]

        # Get mask for presenting only top-k features
        k_features = np.empty(n_features)
        k_features.fill(np.nan)
        k_features[top_k_idx] = 1

        # x presented as top features which have highest contributions to model
        xai_original = self.to_original_form(k_features)
        xai_segmented = x_segmented * k_features.reshape(n_features, 1)

        # visualize how lime convert from x -> x'-> z'-> z -> explain
        # Make small sample set only to visualize
        n_features_on = int(n_features / 2)
        sample_size = top_k * 10
        samples_set = self.neighbors(n_features_on=n_features_on, sample_size=sample_size, on_offs=[1, np.nan])

        # Get one sample to visualize z_comma and z_original
        rand_sample = np.random.randint(0, len(samples_set))
        sample = samples_set[rand_sample]
        z_comma, _ = sample

        # get image of z in original, segments, and superpixel
        z_original = self.to_original_form(z_comma)
        viz_dict = {
            "Original": x_original,
            "Features": x_segmented,
            "A sample z drawn from features": z_original,
            f"Top {top_k} features": xai_segmented,
            f"Top {top_k} features in original form": xai_original,

        }
        return viz_dict

    @staticmethod
    def viz_segmented_ts(segmented_ts):
        assert segmented_ts.ndim == 2, "segmented of uts should be 2 dimension [[], ...,[]]"
        n_features, width = segmented_ts.shape
        _len = 0
        for i in range(0, n_features):
            s = np.empty(_len + width)
            s.fill(np.nan)
            _len += width
            s[-width:] = segmented_ts[i]
            # Ugly fix when first value is nan, plot will ignore this value
            #   -> the ax will change and not match -> fix by add 0 value
            s[0] = 0 if s[0] == np.nan else s[0]
            s[-1] = 0 if s[-1] == np.nan else s[-1]
            plt.plot(s)

    @staticmethod
    def viz_explain(demo_dict, idx=0, title=None):
        fig = plt.figure()
        fig.suptitle(title)
        idx = 330
        for k, v in demo_dict.items():
            idx += 1
            ax = plt.subplot(idx)
            ax.set_title(k)
            if v.ndim == 1:
                v[0] = 0 if v[0] == np.nan else v[0]
                v[-1] = 0 if v[-1] == np.nan else v[-1]
                ax.plot(v)
            elif v.ndim == 2:
                LIMETimeSeries.viz_segmented_ts(v)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.show()
