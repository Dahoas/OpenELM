import argparse
import os
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat_folder", type=str)
    args = parser.parse_args()

    with open(os.path.join(args.stat_folder, "stats.jsonl"), "r") as f:
        stats = f.readlines()
        stats = [json.loads(sample) for sample in stats]

    with open(os.path.join(args.stat_folder, "database.jsonl"), "r") as f:
        database = f.readlines()
        database = [json.loads(sample) for sample in database]

    for step, stat_sample, database_sample in zip(range(len(stats)), stats, database):
        stat_sample["step"] = step
        database_sample["step"] = step

    plt.clf()
    steps = [i for i in range(len(database))]
    fitnesses = [sample["fitness"] for sample in database]
    plt.plot(steps, fitnesses, "-o")
    plt.title("Fitnesses vs. steps")
    plt.savefig("figs/steps_fitness.png")

    # Filter out -100
    database = [sample for sample in database if sample["fitness"] != -100]
    plt.clf()
    steps = [sample["step"] for sample in database]
    fitnesses = [sample["fitness"]  for sample in database]
    plt.plot(steps, fitnesses, "-o")
    plt.title("Fitnesses vs. steps")
    plt.savefig("figs/filtered_steps_fitness.png")
