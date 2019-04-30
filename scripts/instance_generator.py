#!/usr/bin/python3

import numpy as np
import random


class Instance:
    pass


class KnapsackInstance(Instance):
    _generator = None

    def __init__(self, n_items, volume_perc, r, capacity, generator):
        self._generator = generator
        self.n_items = n_items
        self.volume_perc = volume_perc
        self.r = r
        self.capacity = capacity
        self.weights = np.zeros(n_items)
        self.profits = np.zeros(n_items)
        self.filename = f"{self._generator.__name__}_{self.n_items}"

    def generate(self):
        self._generator.generate(self)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if np.count_nonzero(self.weights) is not self.n_items:
            # Before printing the instance we must build it.
            self._generator.generate(self)
        info = f"{self.n_items}\n{self.capacity}\n\n"
        for i in range(self.n_items):
            info += f"{self.weights[i]} {self.profits[i]}\n"
        return info

    def to_file(self, filename=None):
        filename = filename or self.filename
        with open(f"{filename}.kp", "w") as file:
            file.write(self.__str__())


class GenerationStrategy:
    correlation = 10

    def __init__(self):
        pass

    def generate(knapsack):
        return


class StronglyCorrelated(GenerationStrategy):
    def generate(knapsack):
        for i in range(knapsack.n_items):
            knapsack.weights[i] = random.randint(1, knapsack.capacity)
            knapsack.profits[i] = knapsack.weights[i] + \
                knapsack.capacity / GenerationStrategy.correlation


class Uncorrelated(GenerationStrategy):
    def generate(knapsack):
        for i in range(knapsack.n_items):
            knapsack.weights[i] = random.randint(1, knapsack.capacity)
            knapsack.profits[i] = random.randint(1, knapsack.capacity)


class WeaklyCorrelated(GenerationStrategy):
    def generate(knapsack):
        for i in range(knapsack.n_items):
            knapsack.weights[i] = random.randint(1, knapsack.capacity)
            ranges = [max([1, knapsack.weights[i] - knapsack.capacity / GenerationStrategy.correlation]),
                      knapsack.weights[i] + knapsack.capacity / GenerationStrategy.correlation]
            knapsack.profits[i] = random.randint(ranges[0], ranges[1])


class InverseCorrelated(GenerationStrategy):
    def generate(knapsack):
        for i in range(knapsack.n_items):
            knapsack.profits[i] = random.randint(1, knapsack.capacity)
            weight = min([knapsack.capacity, knapsack.profits[i] +
                          knapsack.capacity / GenerationStrategy.correlation])
            knapsack.weights[i] = weight


class SubsetSum(GenerationStrategy):
    def generate(knapsack):
        for i in range(knapsack.n_items):
            knapsack.weights[i] = random.randint(1, knapsack.capacity)
            knapsack.profits[i] = knapsack.weights[i]


if __name__ == "__main__":
    reps = 10
    strategies = [Uncorrelated, StronglyCorrelated,
                  SubsetSum, InverseCorrelated, WeaklyCorrelated]
    for stg in strategies:
        for rep in range(reps):
            knapsack = KnapsackInstance(50, 100, 10, 10000, stg)
            knapsack.to_file(filename=f"{knapsack.filename}_{rep}")
