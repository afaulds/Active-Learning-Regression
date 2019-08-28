from ssbase import SemiSupervisedBase


def main():
    names = ["forestfires", "concrete", "cps", "pm10", "housing", "redwine", "whitewine", "bike"]
    names = ["concrete", "cps", "pm10", "housing"]
    methods = ["random", "bemcm", "qbc", "greedy", "qbc2"]
    methods = ["random", "bemcm", "qbc", "greedy"]
    methods = ["random", "qbc"]
    # methods = ["random"]
    for name in names:
        for method in methods:
            s = SemiSupervisedBase(name, method)
            s.get_average()


if __name__ == "__main__":
    main()
