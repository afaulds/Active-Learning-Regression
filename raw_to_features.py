from utils import FeatureStandardizer
from utils import Config


def main():
    for name in Config.get()["names"]:
        fs = FeatureStandardizer(name)
        fs.process()


if __name__ == "__main__":
    main()
