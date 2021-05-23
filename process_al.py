from ssbase import SemiSupervisedBase
from utils import Config


def main():
    for name in Config.get()["names"]:
        for method in Config.get()["methods"]:
            s = SemiSupervisedBase(name, method)
            s.get_average()


if __name__ == "__main__":
    main()
