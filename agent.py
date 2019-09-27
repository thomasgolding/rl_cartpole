import numpy as np 
from gym.spaces import Discrete

class Agent():
    """
    agentclass
    """

    timestep = 0
    reward = 0

    def __init__(self, action_space: Discrete, state_dim: int, 
            discount: float=0.9, qtype: str='linear', 
            epsilon: float = 0.1, alpha: float = 0.01 ) -> None:
        self.action_space = action_space
        self.discount = discount
        self.state_dim = state_dim
        self.state_action_dim = state_dim + action_space.n
        self.qtype = qtype
        self.epsilon = epsilon
        self.alpha = alpha
        self.qmodel_is_setup = False


        ## setup linear q-model
        if qtype == 'linear':
            self.setup_linear_q()

    def setup_linear_q(self):
        self.theta = np.random.uniform(size=self.state_action_dim)
        self.bias = np.random.uniform(size=1)
        self.theta_prev = np.random.uniform(size=self.state_action_dim)
        self.bias_prev = np.random.uniform(size=1)

        self.qmodel_is_setup=True


    def compute_q(self, state, action, old: bool = False):
        stateaction = self.construct_stateaction(state, action)
        result = {}

        if old:
            theta = self.theta_prev
            bias = self.bias_prev
        else:
            theta = self.theta
            bias = self.bias

        result['q'] = np.sum(stateaction*theta) + bias 
        result['dtheta'] = stateaction
        result['dbias'] = 1

        return result

        
    def decide_next_action_random(self) -> int:
        self.timestep += 1
        return self.action_space.sample()

    def construct_stateaction(self, state: np.ndarray, action: int) -> np.ndarray:
        stateaction = np.zeros((self.state_action_dim))
        stateaction[0:self.state_dim] = state
        stateaction[self.state_dim + action] = 1
        return stateaction



    def decide_next_action_q(
        self, state: np.ndarray, 
        deterministic: bool = False, 
        old: bool = False) -> int:
        """
        derive most valuable action from max_a (q(state, a))
        """
        if not self.qmodel_is_setup:
            print("Q-model not setup")
            return
        
        local_epsilon = self.epsilon
        if deterministic:
            local_epsilon = 0.0

        naction = self.action_space.n
        actions = [i for i in range(naction)] 
        qval    = np.array( [self.compute_q(state, a, old)['q'] for a in actions] )

        ## epsilon-greedy, compute probs
        p = np.zeros((naction)) + local_epsilon/(naction-1)
        p[qval.argmax()] = 1.0 - local_epsilon
        
        ## sample from the actions
        next_action = np.random.choice(np.array(actions), size=1, p=p)[0]
        return next_action

        

    def update_q(self, state: np.ndarray, previous_action: int, next_state: np.ndarray, reward: float) -> None:
        """
        take one step and compute the new target q:
        r + discount * max_a'(q(s', a'))
        """

        ## get optimal action and optimal q using old q-parameters
        old_opt_action = self.decide_next_action_q(state = state, deterministic = True, old = True)
        old_opt_q = self.compute_q(state = next_state, action = old_opt_action, old = True)
        
        old_q = reward + self.discount*old_opt_q['q']

        predicted_q = self.compute_q(state = state, action = previous_action, old = False)
        

        diff = old_q - predicted_q['q']
        loss = (diff)**2
        dlossdtheta = -2*diff*predicted_q['dtheta']
        dlossdbias = -2*diff*predicted_q['dbias']

        self.theta_prev = self.theta.copy()
        self.bias_prev = self.bias.copy()

        self.theta -= self.alpha * dlossdtheta
        self.bias  -= self.alpha * dlossdbias

        max_rel_theta = (np.abs(self.theta - self.theta_prev)/self.theta_prev).max()
        max_rel_bias = np.abs((self.bias - self.bias_prev)/self.bias_prev)
        max_rel_diff = max([max_rel_bias, max_rel_theta])

        return [loss, max_rel_diff]









        







        
        



        
        
        


        



   




    





