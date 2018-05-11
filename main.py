import os
import pandas as pd


def dataPreperation(data):
    return data


def readData():
    folderPath = os.getcwd()
    filePath = folderPath + '/dataset/' + 'forestfires.csv'
    dataset = pd.read_csv(filePath)
    return dataset


def main():
    dataset = readData()
    test = pd.get_dummies(dataset)
    test
    print(test)


if __name__ == '__main__':
    main()
