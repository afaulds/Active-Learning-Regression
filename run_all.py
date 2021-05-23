from ssanalyze import SemiSupervisedAnalyze
from utils import Config


def main():
    for name in Config.get()["names"]:
        for method in Config.get()["methods"]:
            s = SemiSupervisedAnalyze(name, method)
            s.get_average()


if __name__ == "__main__":
    main()
