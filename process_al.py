from ssbase import SemiSupervisedBase


def main():
    names = ["concrete", "cps", "forestfires", "pm10", "housing", "redwine", "whitewine"]
    #names = ["concrete", "cps", "forestfires", "pm10", "housing"]
    #names = ["forestfires", "concrete", "housing"]
    #names = ["concrete"]
    methods = ["random", "bemcm", "qbc", "greedy", "qbc2"]
    methods = ["random", "bemcm", "qbc", "greedy"]
    methods = ["random", "greedy"]
    methods = ["random", "random2", "random3", "greedy", "greedy2", "greedy3", "qbc", "qbc2", "qbc3"]
    methods = ["random", "bemcm", "abemcm_linear+", "abemcm_linear-", "abemcm_max", "abemcm_rel", "abemcm_eva", "qbc", "greedy"]

    for name in names:
        for method in methods:
            s = SemiSupervisedBase(name, method)
            s.get_average()


if __name__ == "__main__":
    main()
