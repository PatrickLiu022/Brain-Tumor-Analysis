"""
CSE163 Final Project
Tommy Chung, Patrick Liu, Yi Jin
This file implements and runs functions that are useful to make and
evaluate Decision Tree Classifier model in classification of three
types of brain tumor with a csv file of features.
"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix

class Train:

  def __init__(self, csv):
    self._df = csv.loc[:, csv.columns != 'Unnamed: 0']  


  def get_df(self):
    return self._df


  def get_features(self):
    return self._df.loc[:, self._df.columns != 'labels']


  def get_labels(self):
    return self._df['labels']


  def get_train(self, features, labels):
    train_x,test_x,train_y,test_y = train_test_split(features, labels,
                                                      test_size = 0.3,
                                                      random_state=25)
    return train_x, test_x, train_y, test_y

  # train.py
  def find_max_depth(self, train_x, train_y, test_x, test_y):
    """
    Takes in two training sets and two testing sets. Returns a list
    of accuracies of the model trained by two parameters train_x, and
    train_y and tested by the other two parameters test_x, test_y with
    max depth ranging from 1 to 30.
    """
    accuracies = []
    for i in range(1, 30):
      model = DecisionTreeClassifier(max_depth=i, random_state=1)
      model.fit(train_x, train_y)
      pred_test = model.predict(test_x)
      test_acc = accuracy_score(test_y, pred_test)
      accuracies.append({'max depth': i, 'test accuracy': test_acc})
    return pd.DataFrame(accuracies)

  # train
  def find_max_features(self, train_x, train_y, test_x, test_y):
    """
    Takes in two training sets and two testing sets and returns a
    list of accuracies of the model trained by train_x, and train_y
    and tested by test_x, test_y with max features from 5 to 66.
    """
    accuracies = []
    for i in range(5, 66):
      model = DecisionTreeClassifier(max_depth=9, max_features=i,
                      random_state=1)
      model.fit(train_x, train_y)
      pred_test = model.predict(test_x)
      test_acc = accuracy_score(test_y, pred_test)
      accuracies.append({'max features': i, 'test accuracy': test_acc})
    return pd.DataFrame(accuracies)

  # train
  def find_max_leaf_nodes(self, train_x, train_y, test_x, test_y):
    """
    Takes in two training sets and two testing sets and returns a
    list of accuracies of the model trained by train_x, and train_y
    and tested by test_x, test_y with max leaf node from 5 to 200.
    """
    accuracies = []
    for i in range(1, 40):
      model = DecisionTreeClassifier(max_depth=9, max_features=64,
                      max_leaf_nodes=(i * 5),
                      random_state=1)
      model.fit(train_x, train_y)
      pred_test = model.predict(test_x)
      test_acc = accuracy_score(test_y, pred_test)
      accuracies.append({'max leaf nodes': i * 5,
                'test accuracy': test_acc})
    return pd.DataFrame(accuracies)


  # train
  def train_model(self, train_x, train_y, test_x, test_y):
    """
    Takes in two training sets and two testing sets and returns a
    machine learning model trained by train_x, train_y. Test the model
    with test_x and test_y to get and prints the accuracy score.
    """
    model = DecisionTreeClassifier(max_depth= 9, max_features=64,
                    max_leaf_nodes=95, random_state=1)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    print('Accuracy Score: ' + str(accuracy_score(test_y, predictions)))
    return model


  # train
  def plot_feature_importance(self, model, features):
    """
    Takes in a machine learning model and the features from the
    pandas data set. Plots the feature importance graph of the given
    model. Feature above an importance level of 0.05 will be graphed.
    Graph is saved in feacture_ importance.png.
    """
    importance = model.feature_importances_
    f_importance = pd.DataFrame(list(zip(features.columns, importance)),
                  columns=['Name', 'Importance'])
    f_importance = f_importance.sort_values(by=['Importance'],
                        ascending=False)
    f_importance = f_importance[f_importance['Importance'] >= 0.05]
    sns.barplot(x=f_importance['Importance'] , y=f_importance['Name'])
    plt.title('Feature Importances')
    plt.savefig('feature_importance.png', bbox_inches='tight')


  # train
  def identify_tumor(self, image_npy, label_npy, masks_npy):
    pd_image_scan = pd.DataFrame(image_npy)
    pd_labels = pd.DataFrame(label_npy)
    pd_masks = pd.DataFrame(masks_npy)

    pd_image_scan['image'] = pd_image_scan[0]
    pd_image = pd.DataFrame(pd_image_scan['image'])
    arr = []
    for i in range(len(pd_image)):
      result = 0
      for j in pd_image.loc[i]:
        if j.sum() != 0:
          result += (3064/j.sum() * 100)
        else:
          result += 0
        arr.append(result)

    features = pd.DataFrame(arr)
    features['labels'] = pd.DataFrame(label_npy)
    img_features = features.loc[:, features.columns != 'labels']
    img_labels = features['labels']

    x_train, x_test, y_train, y_test = train_test_split(img_features, img_labels, test_size=0.3)
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), random_state=1)

    with warnings.catch_warnings():
      warnings.filterwarnings('error')
      try:
        mlp.fit(x_train, y_train)
      except Warning:
        print('Max iterations (200) reached')
    print('Training score:', mlp.score(x_train, y_train))
    print('Testing score:', mlp.score(x_test, y_test))

    return img_features, img_labels, mlp
