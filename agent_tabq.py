import numpy as np 

class AgentTabQ():
    """
    agentclass for tab q
    """

    def __init__(self, 
                 nstate: int,
                 naction: int,  
                 discount: float = 0.9,  
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05, 
                 alpha: float = 0.5,
                 alpha_min: float = 0.1 
                 ) -> None:
        
        self.nstate = nstate
        self.naction = naction
        
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_min = alpha_min

        # set decayfactor ~ exp(-i/tau)
        self.tau_epsilon = 0.05*self.nstate*self.naction
        self.tau_alpha = 0.01*self.nstate*self.naction
        self.decay_epsilon = np.exp(-1.0/self.tau_epsilon)
        self.decay_alpha = np.exp(-1.0/self.tau_alpha)

        # init q
        self.q = np.zeros((self.nstate, self.naction))

    def decide_action_random(self) -> int:
        return np.random.randint(0, self.naction)

    def decide_action_deterministic(self, state: int) -> int:
        return np.argmax(self.q[state, :])


    def decide_action_explore(self, state: int, timestep: int = 0) -> int:
        eps_decay = self.epsilon*self.decay_epsilon**timestep
        eps = max(eps_decay, self.epsilon_min)

        if np.random.uniform() < eps:
            return self.decide_action_random()
        else:
            return self.decide_action_deterministic(state = state)
    
    

    def update_q(self, 
                 state: int, 
                 action: int, 
                 reward: float, 
                 next_state: int,
                 timestep: int = 0) -> None:
        q_target = reward + self.discount*np.max(self.q[next_state,:])
        q_old = self.q[state, action]
        
        ## weights
        alpha_decay = self.alpha*self.decay_alpha**timestep
        alpha = max(alpha_decay, self.alpha)
        w1 = (1.0-alpha)
        w2 = alpha
        
        self.q[state, action] = w1*q_old + w2*q_target
            