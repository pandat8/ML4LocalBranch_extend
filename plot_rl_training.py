import pyscipopt
from pyscipopt import Model
import ecole
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from geco.mips.loading.miplib import Loader
from utility import instancetypes, generator_switcher, instancesizes
import sys
#
# def solve_prob(model):
#     MIP_model = model
#     MIP_model.optimize()
#
#     status = MIP_model.getStatus()
#     obj = MIP_model.getObjVal()
#     incumbent_solution = MIP_model.getBestSol()
#     n_vars = MIP_model.getNVars()
#     n_binvars = MIP_model.getNBinVars()
#     time = MIP_model.getSolvingTime()
#     n_sols = MIP_model.getNSols()
#
#     vars = MIP_model.getVars()
#
#     print('Status: ', status)
#     print('Solving time :', time)
#     print('obj:', obj)
#     print('number of solutions: ', n_sols)
#     print('Varibles: ', n_vars)
#     print('Binary vars: ', n_binvars)
#
#     n_supportbinvars = 0
#     for i in range(n_binvars):
#         val = MIP_model.getSolVal(incumbent_solution, vars[i])
#         # assert MIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
#         if MIP_model.isFeasEQ(val, 1.0):
#             n_supportbinvars += 1
#
#
#     print('Binary support: ', n_supportbinvars)
#     print('\n')
#
# instance_type = instancetypes[1]
# instance_size = instancesizes[0]
# dataset = instance_type + instance_size
#
# directory_opt = './result/generated_instances/' + instance_type + '/' + instance_size + '/'
# pathlib.Path(directory_opt).mkdir(parents=True, exist_ok=True)
#
# generator = generator_switcher(dataset)
# generator.seed(100)
# for i in range(5):
#     instance = next(generator)
#     MIP_model = instance.as_pyscipopt()
#
#     MIP_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy', origcopy=False)
#     MIP_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(problemName='subMIPmodelCopy', origcopy=False)
#
#     # MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
#     # MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
#     MIP_model.setParam('presolving/maxrounds', 0)
#     MIP_model.setParam('presolving/maxrestarts', 0)
#     # MIP_model.setParam("limits/nodes", 1)
#     MIP_model.setParam('limits/solutions', 1)
#
#     MIP_copy.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
#     # MIP_copy.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
#     MIP_copy.setParam('presolving/maxrounds', 0)
#     MIP_copy.setParam('presolving/maxrestarts', 0)
#     MIP_copy.setParam("limits/nodes", 1)
#     # MIP_copy.setParam("limits/gap", 10)
#
#     MIP_copy2.setParam('presolving/maxrounds', 0)
#     MIP_copy2.setParam('presolving/maxrestarts', 0)
#
#     solve_prob(MIP_model)
#     solve_prob(MIP_copy)
#     solve_prob(MIP_copy2)

# import random
# # for k in range(100,200,1):
# random.seed(20210625+31)
# print('*'*10+'福彩七乐彩'+'*'*10)
# print('='*29)
# print(' '*5+'基本号码'+' '*11+'特别号码')
# def seven(n):
#     for i in range(n):
#         basic=[]
#         special=[]
#         while len(basic)<7:
#             i=random.randint(1,50)
#             if i not in basic:
#                 basic.append(i)
#         # while len(special)<1:
#         #     i = random.randint(1, 50)
#         #     if i not in basic:
#         #         special.append(i)
#         basic.sort()
#         for i in basic:
#             print(str(i).zfill(2),end=' ')
#         print(' '*3)
# seven(7)
#
# a = []
# for epoch in range(20,51):
#     if epoch % 10 == 0:
#         a.append(epoch)
#         print(epoch)
# a[-1] =  a[-1] + 10
# c = [a, [1,2]]
# del c
# print(a)


import gzip
import pickle
import matplotlib.pyplot as plt
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes

lr = 0.01
epsilon = 0.0
instance_type = instancetypes[0]
lbconstraint_mode = 'asymmetric'
train_instance_size = instancesizes[0]
incumbent_mode = incumbent_modes[0]
epoch = 7
train_dataset = instance_type + train_instance_size
train_directory = './result/generated_instances/' + instance_type + '/' + train_instance_size + '/' + lbconstraint_mode + '/' + incumbent_mode + '/'
reinforce_train_directory = train_directory + 'rl/' + 'reinforce/train/data/'

filename = f'{reinforce_train_directory}lb-rl-checkpoint-reward3-simplepolicy-lr{str(lr)}-epochs{str(epoch)}.pkl'  # instance 100-199

with gzip.open(filename, 'rb') as f:
    data = pickle.load(f)

epochs_np, returns_np, primal_integrals_np, primal_gaps_np = data

# returns_np = np.array(returns_np).reshape(-1)
# returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + np.finfo(np.float32).eps.item())

plt.close('all')
plt.clf()
fig, ax = plt.subplots(3, 1, figsize=(8, 6.4))
fig.suptitle(train_dataset)
fig.subplots_adjust(top=0.5)
ax[0].set_title('lr= ' + str(lr) + ', epsilon=' + str(epsilon), loc='right')
ax[0].plot(epochs_np, returns_np, label='loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel("return")

ax[1].plot(epochs_np, primal_integrals_np, label='primal ingegral')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel("primal integral")
# ax[1].set_ylim([0, 1.1])
ax[1].legend()

ax[2].plot(epochs_np, primal_gaps_np, label='primal gap')
ax[2].set_xlabel('epoch')
ax[2].set_ylabel("primal gap")
ax[2].legend()
plt.show()

# a = []
# a.append(1)
# a.append(2)
# a.append(3)
#
# b = np.array(a)
# b += 2
#
# b = b.tolist()
#
# a.extend(b)
# b = []
# print(a)
#
