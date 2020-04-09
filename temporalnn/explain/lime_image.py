"""Learn lime to explain images.

This module is to demo implement of LIME method from scratch.
"""
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractclassmethod

from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions

from skimage import segmentation, color
from skimage.io import imread
from skimage.transform import resize as imresize
from sklearn.linear_model import Ridge
from skimage.measure import compare_ssim


# Todo: Convert it into markdown
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
