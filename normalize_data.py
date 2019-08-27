import numpy as np
import pickle
import json


def main():
    names = ["forestfires", "concrete", "cps", "pm10", "housing", "redwine", "whitewine", "bike"]
    for name in names:
        obj = Normalize(name)
        obj.process()


class Normalize:

    def __init__(self, name):
        self.name = name

    def process(self):
        """
        Go through each file and change it to an object
        containing data, target, feature_names, and target_names.
        """
        print()
        print("Normalizing {}...".format(self.name))
        self.__read_meta()
        self.__read_header()
        self.__read_data()

        target = self.data[:, [self.meta["target_pos"]]]
        target_names = self.header[self.meta["target_pos"]]

        self.__hot_encoder()
        self.__normalize()
        self.__filter_columns()
        self.data_obj = {
            'data': self.data,
            'target': target,
            'feature_names': self.header,
            'target_names': target_names,
        }
        self.__print_summary()
        self.__write_data()

    def __read_meta(self):
        with open("data/{}.meta".format(self.name), "r") as infile:
            self.meta = json.loads(infile.read())

    def __read_header(self):
        with open("data/{}.txt".format(self.name), "r") as infile:
            for line in infile:
                self.header = np.array(line.strip("\n").split("\t"))
                return

    def __read_data(self):
        self.data = np.genfromtxt("data/{}.txt".format(self.name),
            delimiter="\t",
            skip_header = 1)

    def __write_data(self):
        with open("data/{}.dat".format(self.name), "wb") as outfile:
            outfile.write(pickle.dumps(self.data_obj))

    def __filter_columns(self):
        m = list(range(self.data.shape[1]))
        del m[self.meta["target_pos"]]
        for pos in self.meta["omit_list"]:
            m.remove(pos)
        self.data = self.data[:, m]
        self.header = self.header[m]

    def __hot_encoder(self):
        """
        """
        for pos in self.meta["categorical"]:
            categories = np.unique(self.data[:, pos])
            if len(categories) > 2:
                self.meta["omit_list"].append(pos)
                for category in categories:
                    self.header = np.append(self.header, "{}_{}".format(self.header[pos], category))
                    self.data = np.append(self.data, (self.data[:, pos] == category).astype(float)[..., None], 1)

    def __normalize(self):
        """
        Take feature set and normalize each column based
        on min and max value (scale variable from 0 to 1).

        Args:
            data - np matrix containing all features.
        Return:
            Data set such that all values are between 0 and 1.
        """
        for i in range(self.data.shape[1]):
            x = self.data[:, i]
            x_min = np.min(x)
            x_max = np.max(x)
            if x_min != x_max:
                x = (x - x_min) / (x_max - x_min)
                self.data[:, i] = x

        # Check for NAN values.
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if np.isnan(self.data[i, j]):
                    print(self.data[i, :])
                    exit()

    def __print_summary(self):
        print("Target Name: {}".format(self.data_obj["target_names"]))
        print("Feature Count: {}".format(self.data_obj["feature_names"].size))
        print("Data Size: {}".format(self.data_obj["data"].shape[0]))


if __name__ == "__main__":
    main()
