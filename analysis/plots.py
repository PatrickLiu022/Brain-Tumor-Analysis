"""
CSE163 Final Project
Tommy Chung, Patrick Liu, Yi Jin

This file plots the visualizations for the radiomic
analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import numpy as np


class Plots:

    """
    This class plots the relationship between radiomic features
    and machine learning accuracies.
    """

    def __init__(self, image, labels, masks):
        """
        Stores the image, labels, and masks files.
        """
        self._image = image
        self._labels = labels
        self._masks = masks

    def draw_correlation(self, df):
        """
        Takes in a pandas dataframe and draws the correlation of features
        in the dataframe. Saves the plot as correlation.png.
        """
        corr = df.corr()
        plt.plot(figsize=(15, 10))
        sns.heatmap(corr, vmin=-0.5, vmax=0.5, square=True)
        plt.title('Correlation between Features')
        plt.savefig('../plots/correlation.png', bbox_inches='tight')
        plt.close()

    def plot_accuracies(self, accuracies, name):
        """
        Takes in an accuracy number given by a model prediction and a
        name label. Plots the accuracies of the machine learning model
        prediction with given name save it in the graph
        Accuracy_vs_name.png.
        """
        sns.relplot(kind='line', x=name, y='test accuracy', data=accuracies)
        plt.title(f'Accuracy as {name} Changes')
        plt.xlabel(name)
        plt.ylabel('Accuracy')
        plt.ylim(0.6, 1)
        plt.savefig(f'../plots/Accuracy_vs_{name}.png', bbox_inches='tight')
        plt.close()

    def plot_confusion(self, model, test_x, test_y):
        """
        Takes in a machine learning model and two test sets. Plots the
        confusion graph of the given model with the testing sets test_x
        and test_y. Save it in the confusion_matrix.png.
        """
        matrix = plot_confusion_matrix(model, test_x, test_y,
                                       display_labels=['Meningioma',
                                                       'Glioma',
                                                       'Pituitary'],
                                       normalize='true',
                                       cmap='Blues')
        matrix.ax_.set_title('Normalized Confusion Matrix')
        plt.savefig('../plots/confusion_matrix.png', bbox_inches='tight')
        plt.close()

    def plot_mean_feature(self, df, feature_name):
        """
        Given a dataframe and feature name of existing row, it plots a boxplot
        showing distribution of the feature between the labels, which is
        saved in 'Distribution of {feature_name} between Brain Tumors'.
        """
        df = df.loc[:, [feature_name, 'labels']]
        sns.boxplot(x='labels', y=feature_name, data=df, showfliers=False)
        plt.xticks([0, 1, 2], ['Meningioma',
                               'Glioma',
                               'Pituitary'])
        plt.title(f'Distribution of {feature_name} between Brain Tumors')
        plt.savefig(
          f'../plots/{feature_name}_boxplot.png', bbox_inches='tight'
          )
        plt.close()

    def count_each_class(self):
        """
        Takes the labels in the from the `labels` files and plots
        a bar chart showing how many types of brain tumors are in the
        file and shows their relative size ot each other. Saves in
        "counts_bar.png"
        """
        each_label, counts = np.unique(self._labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(each_label, counts, tick_label=['Meningioma',
                                                'Glioma',
                                                'Pituitary'])
        plt.title('Number of samples for each label')
        plt.savefig('../plots/counts_bar.png', bbox_inches='tight')
        plt.close()

    def compare_images_example(self, images, masks, labels, starting_index):
        """
        Takes in three numpy files containing the brain scans, tumor masks,
        labels, and index. Finds three examples from each category of the
        three brain tumors with given starting index and plot it in the
        file compare_tumors_index.png.
        """
        example = {}
        index = starting_index
        while len(example) < 6:
            index += 1
            if labels[index] == 1:
                example[0] = images[index]
                example[1] = masks[index]
            elif labels[index] == 2:
                example[2] = images[index]
                example[3] = masks[index]
            elif labels[index] == 3:
                example[4] = images[index]
                example[5] = masks[index]

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
        axs[0, 0].imshow(example[0], cmap="gray")
        axs[0, 0].set_title('Meningioma Tumor Image')

        axs[0, 1].imshow(example[1])
        axs[0, 1].set_title('Meningioma Tumor Mask')

        axs[1, 0].imshow(example[2], cmap="gray")
        axs[1, 0].set_title('Glioma Tumor Image')

        axs[1, 1].imshow(example[3])
        axs[1, 1].set_title('Glioma Tumor Mask')

        axs[2, 0].imshow(example[4], cmap="gray")
        axs[2, 0].set_title('Pituitary Tumor Mask')

        axs[2, 1].imshow(example[5])
        axs[2, 1].set_title('Pituitary Tumor Mask')

        plt.savefig(
          f'../plots/compare_tumors_{starting_index}.png', bbox_inches='tight'
          )
        plt.close()

    def plot_tumor_identification(
      self, image, labels, masks, img_features, img_labels, mlp_model):
        """
        Takes in one image, the corresponding labels, mask, features,
        and label, and a neural network model. Identifies the tumor region
        and highlights in white and predicts the given image's tumor label.
        Plots the image with the title of the true label to the predicted one.
        """
        i_index = 0
        j_index = 0
        for i in image:
            for j in i:
                if masks[i_index][j_index] != 1:
                    image[i_index][j_index] = 5000
                j_index = 0 if (j_index == 511) else j_index + 1
            i_index += 1
        new_input_img = img_features.loc[1][0]
        new_input_label = img_labels.loc[1]
        new_input_img = new_input_img.reshape(-1, 1)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        mask_map = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}
        mlp_prediction = list(mlp_model.predict(new_input_img))
        actual_title = mask_map[new_input_label]
        prediction = mask_map[mlp_prediction[0]]

        plt.title(
          f'Image with {actual_title}. Predicted title was: {prediction}')
        plt.savefig('../plots/image_with_mlp_prediction')
        plt.close()
