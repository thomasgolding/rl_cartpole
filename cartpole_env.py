import gym
import numpy as np



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
