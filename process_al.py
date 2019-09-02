from ssbase import SemiSupervisedBase


def main():
    names = ["forestfires", "concrete", "cps", "pm10", "housing", "redwine", "whitewine", "bike"]
    names = ["concrete", "cps", "pm10", "housing"]
    names = ["cps"]
    methods = ["random", "bemcm", "qbc", "greedy", "qbc2"]
    methods = ["random", "bemcm", "qbc", "greedy"]
    methods = ["random", "greedy"]
    methods = ["random", "greedy", "qbc"]
    for name in names:
        for method in methods:
            s = SemiSupervisedBase(name, method)
            s.get_average()


if __name__ == "__main__":
    main()
