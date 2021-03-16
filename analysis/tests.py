import zipfile
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from radiomics import featureextractor
from sklearn.feature_selection import VarianceThreshold
from plots import Plots
from train import Train
from extract_feature import ExtractFeature
from cse163_utils import assert_equals
import os.path
from os import path


def test_extract_zip():
    assert_equals(True, path.exists('../data'))
    assert_equals(True, path.exists('../data/brain_tumor_dataset/labels.npy'))
    assert_equals(True, path.exists('../data/brain_tumor_dataset/images.npy'))
    assert_equals(True, path.exists('../data/brain_tumor_dataset/masks.npy'))

def main():
    test_extract_zip()

if __name__ == '__main__':
    main()