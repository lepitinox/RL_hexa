import gymnasium
import numpy as np
from deepq import DQNAgent
import gymnasium
from tqdm import tqdm
env = gymnasium.make("ALE/Tetris-v5", obs_type='grayscale', render_mode='human')
env.reset()
env.render()
image_crop = (24, 64, 27, 203)
a = 0

agent = DQNAgent()
def preprocess(observation):
    data = observation[0][image_crop[2]:image_crop[3], image_crop[0]:image_crop[1]]
    data = data != 111
    data = data[1::8, ::4]
    return data

agent.load("tetris-dqn-8.h5")

state = env.reset()
state = preprocess(state)
state = np.reshape(state, [1, *(22, 10, 1)])
agent.memory.clear()
agent.epsilon = 0
for time_t in tqdm(range(10000)):
    action = agent.action(state)
    print(action)
    observation = env.step(action)
