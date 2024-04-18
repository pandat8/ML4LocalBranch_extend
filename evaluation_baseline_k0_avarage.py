import ecole
import numpy as np
import pyscipopt
import argparse
import gc
from mllocalbranch_fromfiles import RegressionInitialK_KPrime
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes, regression_modes

"""
Run this script for evaluating the local branching heuristic algorithms (including LB baseline, LB-SR, LB-SRM)
"""

# Argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type = str, default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth')
parser.add_argument('--t_total', type=int, default=60)
parser.add_argument('--dataset_id', type=int, default=0)
parser.add_argument('--enable_merged', dest='merged', action='store_true', help='enable_merged')
parser.add_argument('--disable_merged', dest='merged', action='store_false')
parser.set_defaults(merged=False)
parser.add_argument('--seed', type=int, default=100, help='Radom seed') ## 100 50 101
args = parser.parse_args()

regression_model_path = args.regression_model_path
print(regression_model_path)
# instance_type = instancetypes[1]
instance_size = instancesizes[1]
# incumbent_mode = 'firstsol'
lbconstraint_mode = 'symmetric'
samples_time_limit = 3

node_time_limit = 10
total_time_limit = args.t_total
dataset_id = args.dataset_id
reset_k_at_2nditeration = True

merged = args.merged
seed = args.seed
# lr = 0.0001
# print('learning rate:', lr)

for k in range(0, 2):
    test_instance_size = instancesizes[k]

    instance_type = instancetypes[dataset_id]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'
    for j in range(0, 2):
        incumbent_mode = incumbent_modes[j]

        print(instance_type + test_instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)
        if dataset_id<3:
            m_start = 0 # 0
        else:
            m_start = 1 # 1 # 2

        print('lb_baseline_k0_average started!')
        print('merged :,', merged)


        regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=seed)

        # regression_init_k.generate_k_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)

        # regression_init_k.two_examples()
        # regression_init_k.generate_regression_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)

        # regression_init_k.execute_regression_k_prime(lr=0.00001, n_epochs=201) # setcovering small: lr=0.00002; capa-small: samne; independentset-small: first: lr=0.00002, root: lr=0.00003

        # regression_init_k.execute_regression_mergedatasets(lr=lr, n_epochs=201)  # setcovering small: lr=0.00002; capa-small: samne; independentset-small: first: lr=0.00002, root: lr=0.00003
        if not ((dataset_id==3 and k==1) or (dataset_id==4 and k==1) or (dataset_id==3) or (dataset_id==4)):
            gc.collect()
            regression_init_k.evaluate_localbranching_baseline_k0_average(test_instance_size=test_instance_size, train_instance_size='-small', total_time_limit=total_time_limit, node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration, merged=merged)
        print('lb_baseline_k0_average finished!')
        # regression_init_k.solve2opt_evaluation(test_instance_size='-small')

        # regression_init_k.primal_integral_k_prime(test_instance_size=test_instance_size,
        #                                           total_time_limit=total_time_limit,
        #                                           node_time_limit=node_time_limit)

        # regression_init_k.primal_integral_k_prime_012(test_instance_size=test_instance_size,
        #                                              total_time_limit=total_time_limit,
        #                                              node_time_limit=node_time_limit)

        # regression_init_k.primal_integral_k_prime_2(test_instance_size=test_instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)

        # regression_init_k.primal_integral_k_prime_3(test_instance_size=test_instance_size,
        #                                             total_time_limit=total_time_limit,
        #                                             node_time_limit=node_time_limit)

        # regression_init_k.primal_integral_k_prime_3_sepa(test_instance_size=test_instance_size,
        #                                                 total_time_limit=total_time_limit,
        #                                                 node_time_limit=node_time_limit)

        # regression_init_k.primal_integral_k_prime_miplib_bianry39(test_instance_size=test_instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
