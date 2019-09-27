## make ten episodes:
import pickle
import imageio

with open('res.pickle', 'rb') as f:
    data = pickle.load(f)

agent = data['agent']
env = data['env']
im = []
for ep in range(10):
    state = env.reset()
    done = False
    while not done:
        action  = agent.decide_next_action_q(state, deterministic = True)
        [state, rew, done, info] = env.step(action)
        xim = env.render(mode = 'rgb_array')
        im.append(xim)
env.close()


imageio.mimsave('dump.gif', im)