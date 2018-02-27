import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json


# Load the diabetes dataset
def main():
    #convert("bike")
    convert("concrete")
    #convert("cps")
    #convert("forestfires")
    #convert("housing")
    #convert("pm10")
    #convert("winequality-red")
    #convert("winequality-white")


def convert(name):
    # Get meta data
    with open("data/{}.meta".format(name), "r") as infile:
        column = int(json.loads(infile.read()))
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
    print(m)
    a = {
        'data': x[:, m],
        'target': x[:, column],
        'feature_names': header[m],
        'target_names': header[column],
        'DESCR': ''
    }
    print(a)
    with open("data/{}.dat".format(name), "wb") as outfile:
        outfile.write(pickle.dumps(a))


def converters(name):
    if name == "bike":
        return None
    else:
        return None

if __name__ == "__main__":
    main()
