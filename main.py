import os
import numpy as np
from numpy import array
from numpy import genfromtxt  # to parse csv file into np array
import pandas as pd


def dataPreperation(data):
    return data


def readData():
    folderPath = os.getcwd()
    filePath = folderPath + '/dataset/' + 'forestfires.csv'

    return filePath


def main():
    readData()


if __name__ == '__main__':
    main()
