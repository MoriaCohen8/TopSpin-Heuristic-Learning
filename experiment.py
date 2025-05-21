import time
import csv
import random
from topspin import TopSpinState
from training import *
from BWAS import BWAS


def run_experiment(w, b, heuristic_name, heuristic, states):
    runtimes = []
    path_lengths = []
    expansions = []
    i=0
    for state in states:
        print(heuristic_name+":   "+"w: "+str(w)+"  b: "+str(b)+"     state "+str(i)+"/1000")
        i=i+1
        start_time = time.time()
        path, num_expansions = BWAS(state, w, b, heuristic.get_h_values, 1000000)
        end_time = time.time()
        if path is not None:
            runtimes.append(end_time - start_time)
            path_lengths.append(len(path) - 1)
            expansions.append(num_expansions)

    avg_runtime = round(sum(runtimes) / len(runtimes) if runtimes else float('inf'), 2)
    avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else float('inf')
    avg_expansions = round(sum(expansions) / len(expansions) if expansions else float('inf'), 2)

    return avg_runtime, avg_path_length, avg_expansions


def main():
    w_values = [2, 5]
    b_values = [1,100]
    Bootstrapping = BootstrappingHeuristic(11, 4)
    Bellman = BellmanUpdateHeuristic(11, 4)
    # model.get_h_values()
    heuristics = [
        ("learned-Bellman", Bellman),
        ("basic", BaseHeuristic(11, 4)),
        ("learned-bootstrap", Bootstrapping)
    ]

    # Generate a fixed set of random states to use for all experiments
    test_states = generate_random_states(11, 4, num_states=1000)
    results = []
    for w in w_values:
        for b in b_values:
            for heuristic_name, heuristic in heuristics:
                avg_runtime, avg_path_length, avg_expansions = run_experiment(w, b, heuristic_name, heuristic,
                                                                              test_states)
                results.append([w, b, heuristic_name, avg_runtime, avg_path_length, avg_expansions])

    # Define the file path
    file_path = "heuristic_comparison_results.csv"

    # Write the results to a CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["W", "B", "Heuristic", "Avg. Runtime", "Avg. Path Length", "Avg. # Expansions"])
        writer.writerows(results)

    print(f"CSV file saved to {file_path}")


if __name__ == "__main__":
    main()
