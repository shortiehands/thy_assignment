import os
import argparse

import numpy.random as rnd
from operators import destroy_1, destroy_by_most_expensive, repair_1, repair_best_improvement
from psp import PSP, Parser
from src.alns import ALNS
from src.alns.criteria import *
from src.helper import save_output
from src.settings import DATA_PATH
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    json_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    parsed = Parser(json_file)
    psp = PSP(parsed.name, parsed.workers, parsed.tasks, parsed.Alpha)

    # construct random initialized solution
    psp.random_initialize2(seed)

    print("Initial solution objective is {}.".format(psp.objective()))

    # Generate output file
    save_output("THY_ALNS", psp, "initial")  # // Modify with your name

    # ALNS
    random_state = rnd.RandomState(seed)

    alns = ALNS(random_state)

    # -----------------------------------------------------------------
    # // Implement Code Here
    # You should add all your destroy and repair operators here
    # add destroy operators
    alns.add_destroy_operator(destroy_1)
    # alns.add_destroy_operator(destroy_block_based)
    # alns.add_destroy_operator(destroy_cost_based)
    # alns.add_destroy_operator(destory_for_unassigned)
    # alns.add_destroy_operator(destroy_by_most_expensive)


    # // add repair operators
    # alns.add_repair_operator(repair_1)
    alns.add_repair_operator(repair_best_improvement)
    # -----------------------------------------------------------------

    # run ALNS & Select Criterion
    criterion = HillClimbing()  # // Modify with your criterion

    ## Base Omega and Lambda ##
    omegas = [40,25,10,0]  # // Select the weights adjustment strategy
    lambda_ = 0.9  # // Select the decay parameter

    result = alns.iterate(
        psp, omegas, lambda_, criterion, iterations=10000, collect_stats=True
    )  # // Modify number of ALNS iterations as you see fit

    # result
    solution = result.best_state
    objective = solution.objective()


    print("Best heuristic objective is {}.".format(objective))
    
    # print("Best heuristic objective is {}.".format(best_solution.objective()))

    # visualize final solution and generate output file
    save_output("THY_ALNS", solution, "solution")  # // Modify with your name

### ======================================================== ###
    ## Trying adaptive omega and lambda ##

    def adaptive_results():
        omegas = [20,10,2,0]  # // Select the weights adjustment strategy
        current_omega = omegas.copy()
        current_lambda = 0.6  # // Select the decay parameter

        lambda_min = 0.3
        lambda_max = 0.9
        adaptation_interval = 100
        low_threshold = 0.01
        high_threshold = 0.1
        target_improvement = 0.05

        window_improvement = []
        max_iter = 5000

        current_solution = psp.copy()
        best_solution = current_solution.copy()

        for iter in tqdm(range(max_iter), desc="ALNS Iterations"):
            result = alns.iterate(current_solution, current_omega, current_lambda, criterion, iterations=1, collect_stats=True)
            candidate_solution = result.best_state

            improvement = current_solution.objective() - candidate_solution.objective()
            window_improvement.append(improvement)

            if candidate_solution.objective() < current_solution.objective():
                current_solution = candidate_solution.copy()
                if current_solution.objective() < best_solution.objective():
                    best_solution = current_solution.copy()

            if (iter + 1) % adaptation_interval == 0:
                avg_improve = sum(window_improvement)/len(window_improvement)

                if avg_improve < low_threshold:
                    current_lambda = max(lambda_min, current_lambda *0.9)

                elif avg_improve > high_threshold:
                    current_lambda = min(lambda_max, current_lambda * 0.1)

                scaling = avg_improve / target_improvement if target_improvement > 0 else 0.1
                current_omega = [max(1, omega * scaling) for omega in omegas]

                window_improvement = []

        return best_solution

    # solution = adaptive_results()
    # objective = solution.objective()