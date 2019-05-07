#!/usr/bin/python3

from generate_dataset import *
from ortools.algorithms import pywrapknapsack_solver
from os import listdir
from os.path import isfile, join

outpath = "/home/marrero/Universidad/TesisDoctoral/resources/kplib/results/"


def parse_instance(instance):
    content = []
    with open(instance) as file:
        content = file.readlines()
    dim = float(content[1])
    capacity = float(content[2])
    weights = []
    profits = []
    for line in content[4:]:
        row = line.rstrip().split()
        weights.append(float(row[0]))
        profits.append(float(row[1]))

    #print(f"Resume\n-Dim: {dim}\n-Q: {capacity}\n-W: {weights}\n-P: {profits}")
    return dim, capacity, weights, profits


if __name__ == "__main__":
    paths = [uncorrelated,
             weakly_correlated,
             # strongly_correlated,
             # inverse_strongly
             ]  # subset_sum]
    paths = [j for i in paths for j in i]
    # Creamos el solver
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        'test')
    # Ejecutamos cada instancia
    for path in paths:
        print(f"Working with instances in {path}")
        instances = [join(path, f)
                     for f in listdir(path) if isfile(join(path, f))]
        for i in range(len(instances)):
            print(f"\t- {i + 1}/{len(instances)}")
            dim, capacity, weights, profits = parse_instance(instances[i])
            solver.Init(profits, [weights], [capacity])
            computed_value = solver.Solve()

            out_filename = instances[i][:-2] + "result"
            packed_items = [x for x in range(len(weights))
                            if solver.BestSolutionContains(x)]
            packed_weights = [weights[i] for i in packed_items]
            total_weight = sum(packed_weights)

            with open(out_filename, "w") as ofile:
                ofile.write(
                    "Solved with OR-Tools KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER\n")
                ofile.write(f"Total weight: {total_weight}\n")
                ofile.write(f"Total profit: {computed_value}\n")
        print(f"{path} done!")
