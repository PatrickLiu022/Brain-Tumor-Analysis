"""
CSE163 Final Project
Tommy Chung, Patrick Liu, Yi Jin

This file implements and runs functions to load data from
brain_tumor.zip, extract radiomic feactures, and plot graphs.
"""
import zipfile
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
from sklearn.feature_selection import VarianceThreshold
# import os.path
from os import path

class ExtractFeature:

  def __init__(self, zip_file):
    IMG = '../data/brain_tumor_dataset/images.npy'
    LABELS = '../data/brain_tumor_dataset/labels.npy'
    MASKS = '../data/brain_tumor_dataset/masks.npy'
    if not(path.exists(IMG) or
       path.exists(LABLES) or
       path.exists(MASKS)):
      self._extract_zip(zip_file)
    self._images = np.load(IMG, allow_pickle=True)
    self._labels = np.load(LABELS, allow_pickle=True)
    self._masks = np.load(MASKS, allow_pickle=True)


  def get_npy(self):
    return self._images, self._labels, self._masks

  # extracting file, stays in this class
  def _extract_zip(self, file):
    """
    Takes in a file with the zip extension and unzips it
    in the relative path
    """
    with zipfile.ZipFile(file, 'r') as zip_ref:
      zip_ref.extractall('../data')
    zip_ref.close()

  # extract
  def extract_feature(self, images, masks):
    """
    Takes in three numpy files containing the brain scans, tumor masks.
    Returns a dataframe of all the radiomic feactures
    extracted from images and masks.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    list_features = []
    for i in range(len(images)):
      image = images[i]
      mask = masks[i]
      features = extractor.execute(sitk.GetImageFromArray(image,
                                sitk.sitkInt8),
                    sitk.GetImageFromArray(mask.astype(int)))
      list_features.append(features)
    df = pd.DataFrame(list_features, columns=features.keys())
    return df

  # extracts, stays here
  def feature_selection(self, df):
    """
    Takes in a numpy dataframe. Returns the selected features in df
    with high variance.
    """
    # change non-numeric values to Nan
    df = df.apply(pd.to_numeric, errors='coerce')
    # drop all columns with Nan value
    df = df.dropna(axis='columns')
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(df, y=df['labels'])
    df = df[df.columns[sel.get_support(indices=True)]]
    return df