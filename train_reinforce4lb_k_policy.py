import ecole
import numpy as np
import pyscipopt
from mllocalbranch_fromfiles import RlLocalbranch
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes
import torch
import random
import argparse

"""
Run this script for training the RL model for adapting the value of k
"""


# Argument setting
parser = argparse.ArgumentParser()
# parser.add_argument('--regression_model_path', type = str, default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth')
parser.add_argument('--seed', type=int, default=100, help='Radom seed') #50
args = parser.parse_args()

# regression_model_path = args.regression_model_path
# print(regression_model_path)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


# instance_type = instancetypes[1]
instance_size = instancesizes[1]
# incumbent_mode = 'firstsol'
lbconstraint_mode = 'symmetric'
samples_time_limit = 3

total_time_limit = 60
node_time_limit = 10

reset_k_at_2nditeration = False
use_checkpoint = False
lr_list = [ 0.01] # 0.1, 0.05, 0.01, 0.001,0.0001,1e-5, 1e-6,1e-8
# eps_list = [0, 0.02]
epsilon = 0.0

for lr in lr_list:
    print('learning rate = ', lr)
    print('epsilon = ', epsilon)
    for i in range(0, 1):
        instance_type = instancetypes[i]
        if instance_type == instancetypes[0]:
            lbconstraint_mode = 'asymmetric'
        else:
            lbconstraint_mode = 'symmetric'
        for j in range(0, 1):
            incumbent_mode = incumbent_modes[j]

            print(instance_type + instance_size)
            print(incumbent_mode)
            print(lbconstraint_mode)

            reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=seed)

            reinforce_localbranch.train_agent_policy_k(train_instance_size=instance_size,
                                                       train_incumbent_mode=incumbent_mode,
                                                       total_time_limit=total_time_limit,
                                                       node_time_limit=node_time_limit,
                                                       reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                       lr=lr,
                                                       n_epochs=301,
                                                       epsilon=epsilon,
                                                       use_checkpoint=use_checkpoint
                                                       )

            # reinforce_localbranch.evaluate_localbranching(evaluation_instance_size='-small', total_time_limit=total_time_limit, node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration)

            # reinforce_localbranch.primal_integral(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
            # regression_init_k.solve2opt_evaluation(test_instance_size='-small')
