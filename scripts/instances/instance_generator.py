#!/usr/bin/python3

import numpy as np
from random import randint, seed
from math import sqrt, ceil


class Instance:
    pass


class KnapsackInstance(Instance):
    H = 100

    def __init__(self, n_items, R, h, s=None):
        seed(s)
        self.n_items = n_items
        self.R = R
        self.h = h
        self.weights, self.profits = self.generate()
        self.capacity = int((self.h * sum(self.weights) / (self.H + 1)))
        self.filename = f"{self.__class__.__name__}_{self.n_items}_R_{self.R}_{self.h}"

    def generate(self):
        raise NotImplementedError

    def __repr__(self):
        info = f"{self.n_items}\n{self.capacity}\n\n"
        for i in range(self.n_items):
            info += f"{self.weights[i]} {self.profits[i]}\n"
        return info

    def to_file(self, filename=None):
        if filename is not None:
            self.filename = filename
        with open(f"{self.filename}.kp", "w") as file:
            file.write(self.__str__())


class Uncorrelated(KnapsackInstance):
    def generate(self):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [randint(1, self.R) for _ in range(self.n_items)]
        return w, p


class WeaklyCorrelated(KnapsackInstance):
    def generate(self, corr=10):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [randint(max(1, w_j - self.R / corr),
                     w_j + self.R / corr) for w_j in w]
        return w, p


class StronglyCorrelated(KnapsackInstance):
    def generate(self, corr=10):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [w_j + self.R / corr for w_j in w]
        return w, p


class InverseCorrelated(KnapsackInstance):
    def generate(self, corr=10):
        p = [randint(1, self.R) for _ in range(self.n_items)]
        w = [p_j + self.R / corr for p_j in p]
        return w, p


class AlmostStronglyCorrelated(KnapsackInstance):
    def generate(self, corr=10):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [randint(w_j + self.R / corr - self.R / 500,
                     w_j + self.R / corr + self.R / 500) for w_j in w]
        return w, p


class SubsetSum(KnapsackInstance):
    def generate(self):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [w_j for w_j in w]
        return w, p


class ProfitCeiling(KnapsackInstance):
    def generate(self, d=3):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [d * ceil(w_j / d) for w_j in w]
        return w, p


class Circle(KnapsackInstance):
    def generate(self, d=2):
        w = [randint(1, self.R) for _ in range(self.n_items)]
        p = [int(d * sqrt(w_j * (4 * self.R - w_j))) for w_j in w]
        return w, p


types = [
    Uncorrelated,
    WeaklyCorrelated,
    StronglyCorrelated,
    InverseCorrelated,
    AlmostStronglyCorrelated,
    SubsetSum,
    ProfitCeiling,
    Circle
]

if __name__ == "__main__":
    s = 0
    n = [50, 100, 200, 500, 1000, 10000]
    R = [1000, 10000]
    H = range(50, 51)
    max_instances = 1000
    count = 0
    for t_instance in types:
        print(
            f"Generating {max_instances * len(n) * len(R)} {t_instance.__name__} instances")
        for rep in range(max_instances):
            for size in n:
                for r in R:
                    for h in H:
                        instance = t_instance(size, r, h, s)
                        name = instance.filename + f"_{rep}"

                        instance.to_file(name)
                        s += 1
                        count += 1
    print(f"Generated {count} instances.")
