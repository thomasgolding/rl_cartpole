import numpy as np 
import random
from collections import deque

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.compat.v1 import Session, global_variables_initializer, placeholder, reduce_mean
from tensorflow.compat.v1.train import GradientDescentOptimizer

class AgentDQN():
    """
    agentclass
    """

    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 discount: float = 0.99,  
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = -1.0, 
                 batchsize: int = 32,
                 memorysize: int = 100000,
                 learning_rate: float = 0.001,
                 neurons: list = [10,10],
                 activation: str = 'relu',
                 online_training: bool = True,
                 nupdate_target_qnet: int  = 5,
                 tau = 0.1,
                 ddqn: bool = False,
                 debug = False
                 ) -> None:

        ## problem specific
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        ## agent parameters and setup 
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        if epsilon_decay < 0.0:
            self.epsilon_decay = 1.0 - 1.0/(self.state_dim + self.action_dim)**2
        self.ddqn  = ddqn

        ## training parameters
        self.batchsize = batchsize
        self.memorysize = memorysize
        self.memory = deque(maxlen = memorysize)
        self.online_training = online_training
        self.nupdate_target_qnet = nupdate_target_qnet
        self.tau = tau
        self.i_update = 0
        self.train_history = deque(maxlen=1000)

        ## q-net parameters and setup
        self.neurons = neurons
        self.activation = activation
        self.nlayer = len(self.neurons) 
        self.setup_model()

        ## debug options
        self.debug = debug
        self.debug_dict = {}
        self.debug_dict['update_step'] = deque(maxlen = 10)
        self.debug_dict['loss'] = deque(maxlen = 10)
    

    def reset_agent(self):
        self.epsilon = 1.0
        self.setup_model()
        self.i_update = 0

    def setup_model(self):
        activ = self.activation
        dtype = 'float32'
        self.layers = [Dense(self.neurons[0], input_shape = (self.state_dim,), activation = activ, dtype=dtype)]
        if self.nlayer > 1:
            self.layers +=  [Dense(neuron, activation = activ, dtype=dtype) for neuron in self.neurons[1:]]
        self.layers += [Dense(self.action_dim, activation = 'linear', dtype=dtype)]

        self.qmodel = Sequential(self.layers)
        #self.opt = SGD(learning_rate = self.learning_rate)
        self.opt = Adam(learning_rate = self.learning_rate)
        #self.loss = MeanAbsoluteError()
        #self.opt = GradientDescentOptimizer(learning_rate = self.learning_rate)
        self.loss = MeanSquaredError()
        self.qmodel.compile(optimizer = self.opt, loss = self.loss)
        self.weights = self.qmodel.get_weights()

        self.qmodel_target = clone_model(self.qmodel)
        self.qmodel_target.set_weights(self.weights)


    def set_state_shape(self, s: np.ndarray) -> np.ndarray:
        return s.reshape(1, self.state_dim).copy()  

    def compute_q(self, state: np.ndarray) -> np.ndarray:
        s = self.set_state_shape(state)
        return self.qmodel(s)
    
    def compute_q_target(self, state: np.ndarray) -> np.ndarray:
        s = self.set_state_shape(state)
        return self.qmodel_target(s)

        
    def decide_action_random(self) -> int:
        return np.random.randint(0, self.action_dim)
        
    
    def decide_action_deterministic(self, state: np.ndarray) -> int:
        return np.argmax(self.compute_q(state))


    def decide_action_explore(self, state: np.ndarray) -> int:
        eps = self.epsilon

        if np.random.uniform() < eps:
            return self.decide_action_random()
        else:
            return self.decide_action_deterministic(state = state)

    def save_timestep_to_memory(self,
                               state: np.ndarray,
                               action: int,
                               reward: float,
                               next_state: np.ndarray,
                               done: bool
                               ) -> None:
        self.memory.append((state, action, reward, next_state, done))
    

    def update_qmodel(self):
        if len(self.memory) < self.batchsize: 
            return

        ## setup array for offline batchtraining
        x_obs_array = np.zeros((self.batchsize, self.state_dim))
        q_obs_array = np.zeros((self.batchsize, self.action_dim))

        batch = random.sample(self.memory, self.batchsize)
        i = 0
        for (state, action, reward, next_state, done) in batch:
            new_target = reward
            if not done:
                new_target = reward + self.discount*np.max(self.compute_q_target(state = next_state))
                if self.ddqn:
                    action = self.decide_action_deterministic(next_state)
                    new_target = reward + self.discount*self.compute_q_target(state = next_state)[0, action]
                         
                    
            q_obs = self.compute_q(state = state).numpy()
            q_obs[0, action] = new_target
            x_obs = self.set_state_shape(state)

            ## fill offline training arrays
            x_obs_array[i, :] = state
            q_obs_array[i, :] = q_obs[0,:]

            if self.online_training:
                train = self.qmodel.fit(x = x_obs, y = q_obs, verbose=False)
                self.update_target_network()

        ## offline training:
        if not self.online_training:
            verb = False
            #if self.i_update == 10: verb = True
            train = self.qmodel.fit(x = x_obs_array, y = q_obs_array, verbose = verb)
            self.update_target_network()
            
        self.train_history.append(train.history)

        ## reduce amount of exploration
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)


    def update_target_network(self):
        ## check if it's time and possibly update target-generating network 
        self.i_update +=1
        if np.mod(self.i_update, self.nupdate_target_qnet) == 0:
            #print("updating weights")
            new_weights = self.qmodel.get_weights()
            old_weights = self.qmodel_target.get_weights()
            updated_weights = [(1.0 - self.tau)*w_old + self.tau*w_new for w_old, w_new in zip(old_weights, new_weights)]
            self.qmodel_target.set_weights(updated_weights)
            self.i_update = 0



            

    def _update_q(self, 
                 state: np.ndarray, 
                 action: int, 
                 reward: float, 
                 next_state: np.ndarray,
                 next_state_is_done: bool):
        
        next_q = self.compute_q(state = next_state).numpy()
        
        new_target = reward + self.discount*np.max(next_q)
        if next_state_is_done:
            new_target = -10.

        q = self.compute_q(state = state).numpy()
        q[0,action] = new_target
        
        xx = self.set_state_shape(state)
        
        
        train = self.qmodel.fit(x = xx, y = q, verbose=False)
          
        # lossfunc, _ = self.session.run([self.loss, self.optimizer], 
        #                                feed_dict = {self.state_ph: state, self.new_q: q})

        loss = train.history['loss']

        if self.debug:
            self.debug_dict['update_step'].append((xx, self.qmodel(xx).numpy(), q))
        self.debug_dict['loss'].append(loss[0])
        return loss
        




if __name__ == '__main__':
    a = AgentDQN(state_dim = 4, action_dim = 2)
    s0 = np.arange(4).reshape((1,4))
    res = a.compute_q(s0) 
    print(res)
    print(np.argmax(res))
    
    for i in range(10):
        lossfunc = a.update_q(state = s0, action=0, reward = 1., next_state=s0)
        print(lossfunc)


    
