import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer


def readData():
    folderPath = os.getcwd()
    filePath = folderPath + '/dataset/' + 'forestfires.csv'
    dataset = pd.read_csv(filePath)
    return dataset


def getFeatures(dataset):
    X = dataset.loc[:, dataset.columns != 'area']
    Y = dataset[['area']]
    Y.is_copy = False
    Y.loc[Y['area'] > 0, 'area'] = '1'
    Y.loc[Y['area'] == 0, 'area'] = '0'
    return X, Y


def DecisionTree(descriptiveFeatures, targetFeature):
    le = LabelEncoder()
    targetFeature = targetFeature.apply(le.fit_transform)

    scoring = {'acc': 'accuracy',
               'prec_macro': 'precision_macro',
               'rec_micro': 'recall_macro'}

    clf = DecisionTreeClassifier()
    scores = cross_validate(clf, descriptiveFeatures, targetFeature, scoring=scoring,
                            cv=5, return_train_score=False)
    print(scores)


def NeuralNetworkClassifier(descriptiveFeatures, targetFeature):
    le = LabelEncoder()

    targetFeature = targetFeature.apply(le.fit_transform)

    scoring = {'acc': 'accuracy',
               'prec_macro': 'precision_macro',
               'rec_micro': 'recall_macro'}
    mlp = MLPClassifier(hidden_layer_sizes=(29, 20, 1), max_iter=1000)
    scores = cross_validate(mlp, descriptiveFeatures, targetFeature, scoring=scoring,
                            cv=5, return_train_score=False)
    print(scores)


def main():
    rawDataset = readData()
    dataset = pd.get_dummies(rawDataset)

    # print(dataset)
    descriptiveFeatures, targetFeature = getFeatures(dataset)

    print("NEURAL NETWORK : ", end="\n\n")
    NeuralNetworkClassifier(descriptiveFeatures, targetFeature)
    print("DECISION TREE : ", end="\n\n")
    DecisionTree(descriptiveFeatures, targetFeature)


if __name__ == '__main__':
    main()
