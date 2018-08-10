from ssbase import SemiSupervisedBase


def main():
    #s = SemiSupervisedBase("concrete", "greedy")
    #s.get_average()
    names = ["forestfires", "concrete", "cps", "pm10", "housing", "redwine", "whitewine"]
    for name in names:
        s = SemiSupervisedBase(name, "random")
        s.get_average()
    for name in names:
        s = SemiSupervisedBase(name, "qbc")
        s.get_average()


if __name__ == "__main__":
    main()
