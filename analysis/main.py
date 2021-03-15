from plots import Plots
from train import Train
from extract_feature import ExtractFeature
import zipfile
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from radiomics import featureextractor
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def main():
    DATA_PATH = '../data/brain-tumor.zip'

    # load data
    data = ExtractFeature(DATA_PATH)
    image, labels, masks = data.get_npy()
    features = data.extract_feature(image, masks) # <= this takes a REALLY long time, approx 3 minutes and 30 seconds
    features['labels'] = labels
    df = data.feature_selection(features)
    high_variance_features = df.to_csv('brain_tumor_features.csv')

    # plots for extract
    plots = Plots(image, labels, masks)
    plots.count_each_class()
    plots.compare_images_example(image, masks, labels, 0)
    plots.compare_images_example(image, masks, labels, 150)
    plots.compare_images_example(image, masks, labels, 1000)

    # machine learning

    train = Train(high_variange_features)
    plots.draw_correlation(train.get_df)

    bt_features = train.get_features()
    bt_labels = train.get_labels()
    bt_train_x, bt_test_x, bt_train_y, bt_test_y = train.get_train(bt_features, bt_labels)

    # evaluate accuracy
    plots.plot_accuracies(train.find_max_depth(bt_train_x, bt_train_y, bt_test_x, bt_test_y),
                    'max depth')
    plots.plot_accuracies(train.find_max_features(bt_train_x, bt_train_y, bt_test_x, bt_test_y),
                    'max features')
    plots.plot_accuracies(train.find_max_leaf_nodes(bt_train_x, bt_train_y, bt_test_x, bt_test_y),
                    'max leaf nodes')
    acc_model = train.train_model(bt_train_x, bt_train_y, bt_test_x, bt_test_y)

    plots.plot_confusion(acc_model, bt_test_x, bt_test_y)
    train.plot_feature_importance(acc_model, bt_features)
    plots.plot_mean_feature(train.get_df, 'original_glcm_ClusterShade')

    # neural networks training
    img_features, img_labels, mlp = train.identify_tumor(image, labels, masks)
    plots.plot_tumor_identification(image, labels, masks, img_features, img_labels, mlp)
    



if __name__ == '__main__':
    main()