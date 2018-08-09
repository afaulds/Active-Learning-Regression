import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json


# Load the diabetes dataset
def main():
    convert("bike")
    convert("concrete")
    convert("cps")
    convert("forestfires")
    convert("housing")
    convert("pm10")
    convert("redwine")
    convert("whitewine")


def convert(name):
    """
    Go through each file and change it to an object
    containing data, target, feature_names, and target_names.
    
    Args:
        name - Name of file before extension.
    Return:
        None
    """
    print()
    print("Normalizing {}...".format(name))
    # Get meta data
    with open("data/{}.meta".format(name), "r") as infile:
        column = int(infile.read())
    # Read header
    with open("data/{}.txt".format(name), "r") as infile:
        for line in infile:
            header = line.strip("\n").split("\t")
            break
    x = np.genfromtxt("data/{}.txt".format(name),
        delimiter="\t",
        skip_header = 1)

    header = np.array(header)
    m = list(range(x.shape[1]))
    del m[column]
    a = {
        'data': normalize(x[:, m]),
        'target': x[:, column],
        'feature_names': header[m],
        'target_names': header[column],
    }
    with open("data/{}.dat".format(name), "wb") as outfile:
        outfile.write(pickle.dumps(a))


def normalize(data):
    """
    Take feature set and normalize each column based
    on min and max value (scale variable from 0 to 1).
    
    Args:
        data - np matrix containing all features.
    Return:
        Data set such that all values are between 0 and 1.
    """
    for i in range(data.shape[1]):
        x = data[:, i]
        x_min = np.min(x)
        x_max = np.max(x)
        if x_min != x_max:
            x = (x - x_min) / (x_max - x_min)
            data[:, i] = x
    # Check for NAN values.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                print(data[i, :])
                exit()
    return data


if __name__ == "__main__":
    main()
