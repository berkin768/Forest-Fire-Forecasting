import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
from statistics import mean
import warnings
warnings.filterwarnings('ignore')

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

    names = list()
    values = list()

    for key,val in scores.items():
        names.append(key)
        values.append(mean(val))
        print(key, end=' : ')
        print(mean(val))
    
    plt.figure(figsize=(10,7))
    plt.bar(range(len(names)),values,tick_label=names)
    plt.title('Decision Tree')
    plt.show()

def NeuralNetworkClassifier(descriptiveFeatures, targetFeature, hiddenLayerSize, hiddenLayerNumber, iterationNumber):
    le = LabelEncoder()

    targetFeature = targetFeature.apply(le.fit_transform)

    scoring = {'acc': 'accuracy',
               'prec_macro': 'precision_macro',
               'rec_micro': 'recall_macro'}

    if(hiddenLayerNumber == 1):
        mlp = MLPClassifier(hidden_layer_sizes=(hiddenLayerSize), max_iter=iterationNumber)
    elif(hiddenLayerNumber == 2):
        mlp = MLPClassifier(hidden_layer_sizes=(hiddenLayerSize,hiddenLayerSize*2), max_iter=iterationNumber)
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(hiddenLayerSize,hiddenLayerSize*2,hiddenLayerSize), max_iter=iterationNumber)

    scores = cross_validate(mlp, descriptiveFeatures, targetFeature, scoring=scoring,
                            cv=5, return_train_score=False)

    names = list()
    values = list()

    for key,val in scores.items():
        names.append(key)
        values.append(mean(val))
        print(key, end=' : ')
        print(mean(val))
    
    plt.figure(figsize=(10,7))
    plt.bar(range(len(names)),values,tick_label=names)
    plt.title('Neural Network')
    plt.show()
       
    
def main():
    rawDataset = readData()
    dataset = pd.get_dummies(rawDataset)

    descriptiveFeatures, targetFeature = getFeatures(dataset)

    print("NEURAL NETWORK (15) : ", end="\n")
    NeuralNetworkClassifier(descriptiveFeatures, targetFeature, 15,1,1000) #NN with 15 hidden layer
    print(end="\n")
    print("NEURAL NETWORK (30,60) : ", end="\n")
    NeuralNetworkClassifier(descriptiveFeatures, targetFeature, 30,2,2000) #NN with 15 hidden layer
    print(end="\n")
    print("NEURAL NETWORK (35,70,35) : ", end="\n")
    NeuralNetworkClassifier(descriptiveFeatures, targetFeature, 35,3,500) #NN with 15 hidden layer
    print(end="\n")
    print("DECISION TREE : ", end="\n")
    DecisionTree(descriptiveFeatures, targetFeature)

if __name__ == '__main__':
    main()