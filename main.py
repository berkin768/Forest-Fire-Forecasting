import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer


def dataPreperation(data):
    return data


def readData():
    folderPath = os.getcwd()
    filePath = folderPath + '/dataset/' + 'forestfires.csv'
    dataset = pd.read_csv(filePath)
    return dataset


def getFeatures(dataset):
    X = dataset.loc[:, dataset.columns != 'area']
    Y = dataset[['area']]
    # Y.ix[Y.area > 0, 'area'] = 1
    # Y.ix[Y.area < 0, 'area'] = 0
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
                            cv=5, return_train_score=True)
    print(scores)


def NeuralNetworkClassifier(descriptiveFeatures, targetFeature):
    le = LabelEncoder()

    targetFeature = targetFeature.apply(le.fit_transform)

    X_train, X_test, y_train, y_test = train_test_split(
        descriptiveFeatures, targetFeature, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(29, 20, 1), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    prediction = mlp.predict(X_test)
    """cv = KFold(n_splits=3)
    prediction = cross_val_predict(mlp, X_train, y_train, cv=cv)"""
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))


def main():
    rawDataset = readData()
    dataset = pd.get_dummies(rawDataset)

    # print(dataset)
    descriptiveFeatures, targetFeature = getFeatures(dataset)
    """with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
        print(targetFeature)"""

    # NeuralNetworkClassifier(descriptiveFeatures, targetFeature)
    DecisionTree(descriptiveFeatures, targetFeature)


if __name__ == '__main__':
    main()
