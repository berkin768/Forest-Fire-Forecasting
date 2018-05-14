import os
import pandas as pd

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
    return X,Y

def main():
    rawDataset = readData()
    dataset = pd.get_dummies(rawDataset)
   
    #print(dataset)
    descriptiveFeatures, targetFeature = getFeatures(dataset)
    """with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
        print(testY)"""


if __name__ == '__main__':
    main()
