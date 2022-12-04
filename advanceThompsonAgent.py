from typing import Tuple
import copy
import numpy as np
class agent():
    def __init__(self,prior: Tuple[int, int] = (10, 10)) -> None:
        self.prob = None
        self.distr = {"prior": prior, "data": []}
        self.arms_beliefs = None
        self.coeffs_all = None
    def play(self, agent, machine_numbers, total_round, current_round, my_total_rewards, 
    my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute):
        if self.arms_beliefs is None:
            # Init priors on the first step
            self.arms_beliefs = {k: copy.deepcopy(self.distr) 
                                 for k in range(machine_numbers)}
        else:
            if len(my_history_reward) > 0:
                self.arms_beliefs[my_history_choice[-1]]['data'].append(my_history_reward[-1])
        pvals = []
        for key, distr in self.arms_beliefs.items():
            self.coeffs_all = np.array([0.97**(total_round - i - 1) 
                                for i in range(total_round)])
            # Observed counts
            data = np.array(distr["data"])
            num_steps = len(data)

            beta_a = distr["prior"][0]
            beta_b = distr["prior"][1]

            if num_steps > 0:
                # Slice decay coefficients such that coeffs[-1] is always 1.0
                coeffs = self.coeffs_all[-num_steps:] 

                # Multiply 'coeffs' with 'data' to apply more decay to earlier
                # observations.
                beta_a += np.sum(coeffs * data)
                beta_b += np.sum(coeffs * (1-data))

            # Draw p for Bernoulli distribution from Beta distribution
            pvals.append((key, np.random.beta(beta_a, beta_b)))

        # Take the action with highest p
        return max(pvals, key=lambda d: d[1])[0]