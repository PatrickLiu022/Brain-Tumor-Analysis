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
from os import path

class ExtractFeature:

  """
  This class performs zip file extraction, numpy data loading,
  and radiomic feature extraction.
  """

  def __init__(self, zip_file):
    """
    Takes in a zip file and unzips to extract the
    numpy files
    """
    IMG = '../data/brain_tumor_dataset/images.npy'
    LABELS = '../data/brain_tumor_dataset/labels.npy'
    MASKS = '../data/brain_tumor_dataset/masks.npy'
    if not(path.exists(IMG) or
       path.exists(LABELS) or
       path.exists(MASKS)):
      self._extract_zip(zip_file)
    self._images = np.load(IMG, allow_pickle=True)
    self._labels = np.load(LABELS, allow_pickle=True)
    self._masks = np.load(MASKS, allow_pickle=True)


  def get_npy(self):
    """
    Returns the numpy image, labels, and masks files.
    """
    return self._images, self._labels, self._masks

  def _extract_zip(self, file):
    """
    Takes in a file with the zip extension and unzips it
    in the relative path
    """
    with zipfile.ZipFile(file, 'r') as zip_ref:
      zip_ref.extractall('../data')
    zip_ref.close()

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

  def feature_selection(self, df):
    """
    Takes in a numpy dataframe. Returns the selected features in df
    with high variance.
    """
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis='columns')
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(df, y=df['labels'])
    df = df[df.columns[sel.get_support(indices=True)]]
    return df