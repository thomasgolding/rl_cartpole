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
            

    

    # def setup_states(self):

    #     def gen_linspace(vmin,vmax,n):
    #         if n==1:
    #             return np.array([0.5*(vmax + vmin)])
    #         else:
    #             return np.linspace(vmin, vmax, n)

    #     xsplit = gen_linspace(-self.x_threshold, self.x_threshold, self.ndim['x']-1)
    #     xdotsplit = gen_linspace(-self.x_threshold , self.x_threshold, self.ndim['xdot']-1)
    #     omegasplit = gen_linspace(-self.omega_threshold, self.omega_threshold, self.ndim['omega']-1)
    #     omegadotsplit = 5*gen_linspace(-self.omega_threshold, self.omega_threshold, self.ndim['omega']-1)
    #     self.statebounds = [xsplit, xdotsplit, omegasplit, omegadotsplit]
    #     #self.statebounds = [omegasplit, omegadotsplit]

    #     self.dum0 = np.zeros([v for _,v in self.ndim.items()])

    #     nstate = 1
    #     for _,n in self.ndim.items():
    #         nstate = nstate*n
    #     self.nstate = nstate


    # def map_state(self, state):
    #     [ix, ixdot, iom, iomdot] = [np.sum(xbound < xval) for xbound, xval in zip(self.statebounds, list(state))]

    #     dum = self.dum0.copy()
    #     dum[ix,ixdot,iom,iomdot] = 1
    #     mapped_state = dum.flatten().argmax()
    #     return mapped_state



    # def set_epsilon(self, new_epsilon):
    #     self.epsilon = new_epsilon



    # def init_qtab(self):
    #     self.qtab      = np.random.uniform(size=[self.naction, self.nstate])*0
    #     self.qtab_prev = np.random.uniform(size=[self.naction, self.nstate])*0



        
    # def decide_next_action_random(self) -> int:
    #     self.timestep += 1
    #     return self.action_space.sample()

    # def decide_next_action_q(self, state: int, 
    #     deterministic: bool = False, 
    #     old: bool = False) -> int:
    #     """
    #     derive most valuable action from max_a (q(state, a))
    #     """
        
    #     local_epsilon = self.epsilon
    #     if deterministic:
    #         local_epsilon = 0.0
        
    #     if old:
    #         qval = self.qtab[:,state]
    #     else:
    #         qval = self.qtab_prev[:,state]


    #     ## epsilon-greedy, compute probs
    #     p = np.zeros((self.naction)) + local_epsilon/(self.naction-1)
    #     p[qval.argmax()] = 1.0 - local_epsilon
        
    #     ## sample from the actions
    #     actions = np.arange(self.naction)
    #     next_action = np.random.choice(np.array(actions), size=1, p=p)[0]
    #     return next_action

        
    # def update_q(self, state: int, previous_action: int, next_state: int, reward: float) -> None:
    #     """
    #     take one step and compute the new target q:
    #     r + discount * max_a'(q(s', a'))
    #     """

    #     ## get optimal action and optimal q using old q-parameters
    #     old_opt_action = self.decide_next_action_q(state = next_state, deterministic = True, old = True)
    #     old_opt_q = self.qtab_prev[old_opt_action, next_state] 
        
    #     old_q = reward + self.discount*old_opt_q
    #     new_q = self.qtab[previous_action, state]

    #     self.qtab[previous_action, state] = (1.0-self.alpha)*new_q  +  self.alpha*old_q
         
    #     self.qtab_prev = self.qtab.copy()








        







        
        



        
        
        


        



   




    





