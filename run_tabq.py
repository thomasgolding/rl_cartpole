import gym
import numpy as np
from agent_tabq import AgentTabQ

class EnvCartPole():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.ndim = {
            'x': 6, 
            'xdot': 6,
            'omega': 10,
            'omegadot': 12
        }
        n = 1
        for k,v in self.ndim.items():
            n = n*v
        self.nstate = n
        self.naction = self.env.action_space.n
        self.x_threshold = 2.4
        self.omega_threshold = np.pi/180.*12.
        self.setup_statespace()

    def setup_statespace(self):
        self.dum0 = np.zeros([v for _,v in self.ndim.items()])

        def gen_linspace(vmin,vmax,n):
            if n==1:
                return np.array([0.5*(vmax + vmin)])
            else:
                return np.linspace(vmin, vmax, n)

        xsplit = gen_linspace(-self.x_threshold, self.x_threshold, self.ndim['x']-1)
        xdotsplit = gen_linspace(-self.x_threshold , self.x_threshold, self.ndim['xdot']-1)
        omegasplit = gen_linspace(-self.omega_threshold, self.omega_threshold, self.ndim['omega']-1)
        omegadotsplit = 5*gen_linspace(-self.omega_threshold, self.omega_threshold, self.ndim['omega']-1)
        self.statebounds = [xsplit, xdotsplit, omegasplit, omegadotsplit]

    def map_state(self, state):
        [ix, ixdot, iom, iomdot] = [np.sum(xbound < xval) for xbound, xval in zip(self.statebounds, list(state))]

        dum = self.dum0.copy()
        dum[ix,ixdot,iom,iomdot] = 1
        mapped_state = dum.flatten().argmax()
        return mapped_state


    def step(self, action):
        [s,r,d,info] = self.env.step(action)
        ms = self.map_state(s)
        return [ms, r, d, info]
    
    def reset(self):
        s = self.env.reset()
        ms = self.map_state(s)
        return ms

def play_game(a):
    envir = EnvCartPole()
    s = envir.reset()
    score = 0
    done = False
    while not done:
        action = a.decide_action_deterministic(s)
        [s, r, done, _] = envir.step(action)
        score = score + r

    return score



def n_timesteps(a):
    envir = EnvCartPole()
    s = envir.reset()
    for it in range(210):
        action = a.decide_action_deterministic(s)
        [s,_,d,_] = envir.step(action)
        if d:
            break
    
    return it


def exp_timestep(a):
    n = 50
    nt = np.zeros(n)
    for i in range(n):
        nt[i] = n_timesteps(a)
    return  nt.mean()



def run_experiment(ntime = 100):
    env = EnvCartPole()
    discount = 1.0
    epsilon = 0.5

    qdict = {}
    qdict['info'] = []
    agent = AgentTabQ(nstate = env.nstate, naction = env.naction, discount=discount)

    def score_agent():
        score = 0
        s = env.reset()
        done = False
        while not done:
            a = agent.decide_action_deterministic(s)
            [s,r,done,_] = env.step(a)
            score += r
        return score

    def check_convergence():
        """
        converged if max score 50 times in a row.
        """
        success = True
        n_success = 0
        while success and n_success < 50:
            score = score_agent()
            if score > 199.9:
                n_success += 1
            else:
                success = False

        return [success, n_success]



    
    for t in range(ntime):
        s = env.reset()
        done = False
        while not done:
            a = agent.decide_action_explore(s, timestep=t)
            [s2, r, done, info] = env.step(a)
            agent.update_q(state=s, action=a, reward=r, next_state=s2, timestep = t)
            s = s2

        if np.mod(t, 10) == 0:
            score = score_agent()
            [success, n_success] = check_convergence()
            exp_n = exp_timestep(agent)
            istring = "episode = {}, score = {}, n_success = {}, exp_ts = {}".format(t, score, n_success, exp_n)
            print(istring)
            qdict[t] = agent.q.copy()
            qdict['info'].append(istring)

    return (agent, qdict)


            

