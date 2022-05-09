from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING
from utilities import copy_sol, copy_sol_from_subMIP_to_MIP
import numpy as np

from localbranching import LocalBranching


t_reward_types = ['reward_k', 'reward_k+t']

class HeurLocalbranch(Heur):

    def __init__(self, k_0, agent_k, node_time_limit, total_time_limit, is_symmetric, reset_k_at_2nditeration, device):
        super().__init__()
        self.k_0 = k_0
        self.agent_k = agent_k
        self.node_time_limit = node_time_limit
        self.total_time_limit  = total_time_limit
        self.is_symmetric = is_symmetric
        self.reset_k_at_2nditeration = reset_k_at_2nditeration
        self.device = device

    def heurexec(self, heurtiming, nodeinfeasible):

        incumbent_solution = self.model.getBestSol()
        assert (incumbent_solution is not None), 'initial solution of LB is None'
        assert self.model.checkSol(incumbent_solution), 'initial solution of LB is not feasible'

        self.model.resetParams()

        MIP_model_copy, MIP_copy_vars, success = self.model.createCopy(
            problemName='lb-subMIP',
            origcopy=False)

        MIP_model_copy, sol_MIP_copy = copy_sol(self.model, MIP_model_copy, incumbent_solution,
                                                  MIP_copy_vars)

        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, MIP_vars=MIP_copy_vars, k=self.k_0,
                                   node_time_limit=self.node_time_limit,
                                   total_time_limit=self.total_time_limit)

        status, obj_best, elapsed_time, agent_k, _, success_lb = self.mdp_localbranch(
            localbranch=lb,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=self.reset_k_at_2nditeration,
            agent_k=self.agent_k,
            optimizer_k=None,
            device=self.device)

        if success_lb:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

    def mdp_localbranch(self, localbranch=None, is_symmetric=True, reset_k_at_2nditeration=False, agent_k=None,
                        optimizer_k=None, agent_t=None, optimizer_t=None, device=None, enable_adapt_t=False,
                        t_reward_type=t_reward_types[0]):

        success = False

        # self.total_time_limit = total_time_limit
        localbranch.total_time_available = localbranch.total_time_limit
        localbranch.first = False
        localbranch.diversify = False
        localbranch.t_node = localbranch.default_node_time_limit
        localbranch.div = 0
        localbranch.is_symmetric = is_symmetric
        localbranch.reset_k_at_2nditeration = reset_k_at_2nditeration
        lb_bits = 0
        # t_list = []
        # obj_list = []
        # lb_bits_list = []
        # k_list = []

        # lb_bits_list.append(lb_bits)
        # t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        # obj_list.append(localbranch.MIP_obj_best)
        # k_list.append(localbranch.k)

        k_action = localbranch.actions['unchange']
        t_action = localbranch.actions['unchange']

        # initialize the env to state_0
        lb_bits += 1
        state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action, t_action=t_action,
                                                                             lb_bits=lb_bits)

        if success_step and (localbranch.MIP_vars is not None):
            self.model, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model,
                                                                  localbranch.MIP_sol_best, localbranch.MIP_vars)
            if feasible:
                success = True

        localbranch.MIP_obj_init = localbranch.MIP_obj_best
        # lb_bits_list.append(lb_bits)
        # t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        # obj_list.append(localbranch.MIP_obj_best)
        # k_list.append(localbranch.k)

        if (not done) and reset_k_at_2nditeration:
            lb_bits += 1
            localbranch.default_k = 20
            if not localbranch.is_symmetric:
                localbranch.default_k = 10
            localbranch.k = localbranch.default_k
            localbranch.diversify = False
            localbranch.first = False

            state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action,
                                                                                 t_action=t_action,
                                                                                 lb_bits=lb_bits)
            if success_step and (localbranch.MIP_vars is not None) :
                self.model, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model, localbranch.MIP_sol_best, localbranch.MIP_vars)
                if feasible:
                    success = True

            localbranch.MIP_obj_init = localbranch.MIP_obj_best
            # lb_bits_list.append(lb_bits)
            # t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            # obj_list.append(localbranch.MIP_obj_best)
            # k_list.append(localbranch.k)

        while (not done) and localbranch.div < localbranch.div_max :
            lb_bits += 1

            k_vanilla, t_action = localbranch.policy_vanilla(state)

            # data_sample = [state, k_vanilla]
            #
            # filename = f'{samples_dir}imitation_{localbranch.MIP_model.getProbName()}_{lb_bits}.pkl'
            #
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data_sample, f)

            k_action = k_vanilla
            if agent_k is not None:
                k_action = agent_k.select_action(state)

            if agent_t is not None:
                t_action = agent_t.select_action(state)

                # # for online learning, update policy
                # if optimizer is not None:
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

            # execute one iteration of LB, get the state and rewards

            state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action,
                                                                                 t_action=t_action, lb_bits=lb_bits,
                                                                                 enable_adapt_t=enable_adapt_t)
            if success_step and (localbranch.MIP_vars is not None) :
                self.model, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model, localbranch.MIP_sol_best, localbranch.MIP_vars)
                if feasible:
                    success = True


            if agent_k is not None:
                agent_k.rewards.append(reward_k)
            if agent_t is not None:
                if t_reward_type == t_reward_types[1]:
                    reward_t = reward_k + reward_time
                else:
                    reward_t = reward_k
                agent_t.rewards.append(reward_t)

            # lb_bits_list.append(lb_bits)
            # t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            # obj_list.append(localbranch.MIP_obj_best)
            # k_list.append(localbranch.k)

        print(
            'K_final: {:.0f}'.format(localbranch.k),
            'div_final: {:.0f}'.format(localbranch.div)
        )

        # localbranch.solve_rightbranch()
        # t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        # obj_list.append(localbranch.MIP_obj_best)
        # k_list.append(localbranch.k)

        status = localbranch.MIP_model.getStatus()
        # if status == "optimal" or status == "bestsollimit":
        #     localbranch.MIP_obj_best = localbranch.MIP_model.getObjVal()

        elapsed_time = localbranch.total_time_limit - localbranch.total_time_available

        # lb_bits_list = np.array(lb_bits_list).reshape(-1)
        # times_list = np.array(t_list).reshape(-1)
        # objs_list = np.array(obj_list).reshape(-1)
        # k_list = np.array(k_list).reshape(-1)

        # plt.clf()
        # fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        # fig.suptitle(self.instance_type + 'large' + '-' + self.incumbent_mode, fontsize=13)
        # # ax.set_title(self.insancte_type + test_instance_size + '-' + self.incumbent_mode, fontsize=14)
        #
        # ax[0].plot(times_list, objs_list, label='lb-rl', color='tab:red')
        # ax[0].set_xlabel('time /s', fontsize=12)
        # ax[0].set_ylabel("objective", fontsize=12)
        # ax[0].legend()
        # ax[0].grid()
        #
        # ax[1].plot(times_list, k_list, label='lb-rl', color='tab:red')
        # ax[1].set_xlabel('time /s', fontsize=12)
        # ax[1].set_ylabel("k", fontsize=12)
        # ax[1].legend()
        # ax[1].grid()
        # # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # # fig.tight_layout()
        # # plt.savefig(
        # #     './result/plots/' + self.instance_type + '_' + self.instance_size + '_' + self.incumbent_mode + '.png')
        # plt.show()
        # plt.clf()

        del localbranch.subMIP_sol_best
        del localbranch.MIP_sol_bar
        del localbranch.MIP_sol_best

        return status, localbranch.MIP_obj_best, elapsed_time, agent_k, agent_t, success




