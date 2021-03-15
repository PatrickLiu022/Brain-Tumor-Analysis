import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import numpy as np

class Plots:

    def __init__(self, image, labels, masks):
        self._image = image
        self._labels = labels
        self._masks = masks
        

    def draw_correlation(df):
        """
        Takes in a pandas dataframe and draws the correlation of features
        in the dataframe. Saves the plot as correlation.png.
        """
        corr = df.corr()
        plt.plot(figsize=(15, 10))
        sns.heatmap(corr, vmin=-0.5, vmax=0.5, square=True)
        plt.title('Correlation between Features')
        plt.savefig('correlation.png', bbox_inches='tight')
        plt.close()

    # plotting
    def plot_accuracies(accuracies, name):
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
        plt.savefig(f'Accuracy_vs_{name}.png', bbox_inches='tight')
        plt.close()

    # plot
    def plot_confusion(model, test_x, test_y):
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
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        plt.close()

    # plot
    def plot_mean_feature(df, feature_name):
        """
        Given a dataframe and feature name of existing row, it plots a boxplot
        showing distribution of the feature between the labels, which is
        saved in 'Distribution of {feature_name} between Brain Tumors'.
        """
        sns.boxplot(x='label', y=feature_name, data=df, showfliers = False)
        plt.xticks([0, 1, 2], ['Meningioma',
                            'Glioma',
                            'Pituitary'])
        plt.title(f'Distribution of {feature_name} between Brain Tumors')
        plt.savefig(f'{feature_name}_boxplot_png', bbox_inches='tight')
        plt.close()


    # from extract_feature
    def count_each_class(self):
        """
        Takes the labels in the from the `labels` files and plots
        a bar chart showing how many types of brain tumors are in the
        file and shows their relative size ot each other. Saves in
        "counts_bar.png"
        """
        each_label, counts = np.unique(self._labels, return_counts=True)
        plt.figure(figsize=(10,6))
        plt.bar(each_label, counts, tick_label=['Meningioma',
                            'Glioma',
                            'Pituitary'])
        plt.title('Number of samples for each label')
        plt.savefig('counts_bar.png', bbox_inches='tight')
        plt.close()

    
    # from extract feature
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

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,15))
        image_menin = sitk.GetImageFromArray(example[0], sitk.sitkInt8)
        axs[0,0].imshow(sitk.GetArrayFromImage(image_menin), cmap="gray")
        axs[0,0].set_title('Meningioma Tumor Image')
        mask_menin = sitk.GetImageFromArray(example[1].astype(int))
        axs[0,1].imshow(sitk.GetArrayFromImage(mask_menin))
        axs[0,1].set_title('Meningioma Tumor Mask')
        image_glio = sitk.GetImageFromArray(example[2], sitk.sitkInt8)
        axs[1,0].imshow(sitk.GetArrayFromImage(image_glio), cmap="gray")
        axs[1,0].set_title('Glioma Tumor Image')
        mask_glio = sitk.GetImageFromArray(example[3].astype(int))
        axs[1,1].imshow(sitk.GetArrayFromImage(mask_glio))
        axs[1,1].set_title('Glioma Tumor Mask')
        image_pitu = sitk.GetImageFromArray(example[4], sitk.sitkInt8)
        axs[2,0].imshow(sitk.GetArrayFromImage(image_pitu), cmap="gray")
        axs[2,0].set_title('Pituitary Tumor Mask')
        mask_pitu = sitk.GetImageFromArray(example[5].astype(int))
        axs[2,1].imshow(sitk.GetArrayFromImage(mask_pitu))
        axs[2,1].set_title('Pituitary Tumor Mask')

        plt.savefig(f'compare_tumors_{starting_index}.png', bbox_inches='tight')
        plt.close()


    def plot_tumor_identification(self, image, labels, masks, img_feautres, img_labels, model):
        masks_npy = masks + 1
        image_scan = image
        for i in range(len(image_scan_1[2])):
            for j in range(len(image_scan_1[2][i])):
                val = masks_npy_1[i][j]
                for k in range(len(val)):
                    if val[k] != 1:
                        image_scan_1[i][j][k] = 10000
        new_input_img = img_features.loc[1][0]
        new_input_label = img_labels.loc[1]
        new_input_img = new_input_img.reshape(-1, 1)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        plt.imshow(image_scan_1[1], cmap='gray')
        mask_map = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}
        mlp_prediction = list(mlp.predict(new_input_img))
        actual_title = mask_map[new_input_label]
        prediction = mask_map[mlp_prediction[0]]

        plt.title(f'Image with {actual_title}. Predicted title was: {prediction}')