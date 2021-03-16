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


def test_extract_feature():
    test_EF = ExtractFeature(Nan)
    test_EF._images = test_EF._images[:30]
    test_EF._masks = test_EF._masks[:30]
    test_EF._labels = test_EF._labels[:30]
    features = data.extract_feature(image, masks)
    features['labels'] = labels
    assert_equals(30, len(features.index))
    selected_features = data.feature_selection(features)
    assert_equals(30, len(selected_features.index))


def test_brain_tumor_features_csv():
    features = pd.read_csv('brain_tumor_features.csv')
    assert_equals(3064, len(features.index))
    assert_equals(3, features['labels'].unique())


def main():
    test_extract_zip()
    test_feature_selection()
    test_brain_tumor_features_csv()

if __name__ == '__main__':
    main()