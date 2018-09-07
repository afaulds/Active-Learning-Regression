from ssbase import SemiSupervisedBase


def main():
    #s = SemiSupervisedBase("concrete", "greedy")
    #s.get_average()
    names = ["forestfires", "concrete", "cps", "pm10", "housing", "redwine", "whitewine", "bike"]
    names = ["pm10"]
    methods = ["random", "greedy", "qbc"]
    for name in names:
        for method in methods:
            s = SemiSupervisedBase(name, method)
            s.get_average()


if __name__ == "__main__":
    main()
