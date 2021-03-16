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
    run = True

    while run:

        print('\nLoading data and extracting features ...')
        print('\nIf you ran this section before on your local device, you can skip this part.')
        load_data_input = input('Continue? Press any key to continue, n to stop: ')

        if load_data_input != 'n':
            # load data
            data = ExtractFeature(DATA_PATH)
            image, labels, masks = data.get_npy()
            features = data.extract_feature(image, masks) # <= this takes a REALLY long time, approx 3 minutes and 30 seconds
            features['labels'] = labels
            df = data.feature_selection(features)
            df.to_csv('brain_tumor_features.csv')

        
        print('\nPlotting visualizations and training models ...')
        user_input= input('Continue? Press any key to contninue, n to stop: ')

        if user_input != 'n':

            data = ExtractFeature(DATA_PATH)
            image, labels, masks = data.get_npy()
            high_variance_features = pd.read_csv('brain_tumor_features.csv')

            # plots for extract
            plots = Plots(image, labels, masks)
            print('Generating Tumor Counts plot ...')
            plots.count_each_class()
            print('Generating image and mask example 1...')
            plots.compare_images_example(image, masks, labels, 0)
            print('Generating image and mask example 2 ...')
            plots.compare_images_example(image, masks, labels, 150)
            print('Generating image and mask example 3 ...')
            plots.compare_images_example(image, masks, labels, 1000)

            # machine learning
            train = Train(high_variance_features)
            plots.draw_correlation(train.get_df())

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
            plots.plot_mean_feature(train.get_df(), 'original_glcm_ClusterShade')

        print('\nTraining neural networks ...')
        user_input= input('Continue? Press any key to continue, n to stop: ')

        if user_input != 'n':
            # neural networks training
            index = input('Pick any of the 3064 images for the model to predict on by typing a number from 1 - 3064: ')
            index = int(index) - 1
            image_scan, img_features, img_labels, mlp = train.identify_tumor(image, labels, masks, index)
            print('Generating prediction of mask in image and label of true mask')
            masks_int = masks[index]
            masks_int = masks_int + 1
            plots.plot_tumor_identification(image_scan, labels, masks_int, img_features, img_labels, mlp)

        user_input = input('Would you like to run the model again? (y/n): ')
        if user_input == 'n':
            run = False

        



if __name__ == '__main__':
    main()