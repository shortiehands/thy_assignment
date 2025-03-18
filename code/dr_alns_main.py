import argparse
from src.helper import save_output
from stable_baselines3 import PPO
import statistics

from psp_AlnsEnv import pspAlnsEnv

# model_path = ""  # // Include trained model path here (e.g. "src/dr_alns/trained_model/pspAlnsEnv/<IDX>_PPO_ActorCriticPolicy_<N_STEPS>_<N_WORKERS>_<DATE>/model.zip")
model_path = "/Users/panda/Desktop/MITB/10_AIPlanning/thy_assignment/code/src/dr_alns/trained_models/pspAlnsEnv/6_PPO_ActorCriticPolicy_2000000_10_03-16_21-12/intermediate_models/intermediate_model_2000000_steps.zip"

if __name__ == "__main__":
    iterations = 1000  # // Modify number of ALNS iterations as you see fit

    model = PPO.load(model_path)
    
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=lambda s: [int(item) for item in s.split(',')], help='seed')
    args = parser.parse_args()
    
    json_file = args.data
    
    objs = []
    
    for seed in args.seed:
        print(seed)
        seed = int(seed)
        
        parameters = {
            "environment": {
                "iterations": iterations,
                "instances_folder": "sample_instances",
                "instances": json_file
            }
        }
        env = pspAlnsEnv(parameters)
        env.run(model, seed = seed)

        # result
        solution = env.best_solution
        objective = env.best_solution.objective()
        objs.append(objective)
        print("Best objective is {}.".format(objective))

        # generate output file
        save_output("THY_DR_ALNS", solution, "solution" + str(seed))
        
    print(statistics.median(objs))