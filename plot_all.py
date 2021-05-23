import matplotlib.pyplot as plt
import numpy as np
import os
from ssbase import SemiSupervisedBase
from utils import Cache, Config


data = {}


def main():
    # Write output.
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("results/img"):
        os.mkdir("results/img")
    for name in Config.get()["names"]:
        for method in Config.get()["methods"]:
            data[(name, method)] = average_runs(name, method)
        plot_error(name)


def average_runs(name, method):
    num_runs = Config.get()["num_runs"]
    new_percent_list = np.arange(
        0.1, # self.label_percent,
        0.4, # self.max_percent,
        0.015 # self.batch_percent / 2.0
    )
    rmse_avg_list = np.zeros((len(new_percent_list)))
    mae_avg_list = np.zeros((len(new_percent_list)))
    for run_id in range(num_runs):
        key = "{}_{}_{}".format(name, method, run_id)
        percent_list, rmse_list, mae_list = Cache.get(key)
        rmse_avg_list += np.interp(new_percent_list, percent_list, rmse_list)
        mae_avg_list += np.interp(new_percent_list, percent_list, mae_list)
    rmse_avg_list /= num_runs
    mae_avg_list /= num_runs
    return new_percent_list, rmse_avg_list, mae_avg_list


def plot_error(name):
    # Plot RMSE
    fig, ax = plt.subplots()
    for method in Config.get()["methods"]:
        ax.plot(
            data[(name, method)][0],
            data[(name, method)][1],
            label=method
        )
    ax.legend(loc="upper right")
    ax.set_title("RMSE for {}".format(name))
    plt.savefig("results/img/{}_rmse.png".format(name))
    plt.close()

    # Plot MAE
    fig, ax = plt.subplots()
    for method in Config.get()["methods"]:
        ax.plot(
            data[(name, method)][0],
            data[(name, method)][2],
            label=method
        )
    ax.legend(loc="upper right")
    ax.set_title("MAE for {}".format(name))
    plt.savefig("results/img/{}_mae.png".format(name))
    plt.close()


if __name__ == "__main__":
    main()
