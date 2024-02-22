import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from deepq import DQNAgent
import gymnasium
from tqdm import tqdm

env = gymnasium.make("ALE/Tetris-v5", obs_type='grayscale', render_mode='human')
env.reset(seed=42069)
env.render()

image_crop = (24, 64, 27, 203)
a = 0
tetrominoes = {
    'I': np.array([[1, 1, 1, 1]]),
    'L': np.array([[1, 1, 1], [1, 0, 0]]),
    'J': np.array([[1, 1, 1], [0, 0, 1]]),
    'O': np.array([[1, 1], [1, 1]]),
    'S': np.array([[0, 1, 1], [1, 1, 0]]),
    'T': np.array([[1, 1, 1], [0, 1, 0]]),
    'Z': np.array([[1, 1, 0], [0, 1, 1]])
}

tetrominoes_spawn_positions = {
    "I": 3,
    "J": 3,
    "L": 3,
    "O": 4,
    "S": 3,
    "T": 3,
    "Z": 3
}


def get_spawn_position(piece):
    data = np.zeros((2, 10))
    for en, line in enumerate(tetrominoes[piece]):
        data[en, tetrominoes_spawn_positions[piece]:tetrominoes_spawn_positions[piece] + len(line)] = line
    return data


agent = DQNAgent()

EPISODES = 500
BATCH_SIZE = 32
pieces_dict = {0: "I", 1: "J", 2: "L", 3: "O", 4: "S", 5: "T", 6: "Z"}


def preprocess(observation):
    data = observation[0][image_crop[2]:image_crop[3], image_crop[0]:image_crop[1]]
    data = data != 111
    data = data[1::8, ::4]
    return data


def is_new_piece_frame(data):
    # Check if the piece is in the top 2 rows
    # use the get_spawn_position function to check if the piece is in the top 2 rows
    for piece in tetrominoes:
        if np.array_equal(data[:2], get_spawn_position(piece)):
            return True
    return False


done = False
for e in range(EPISODES):
    print("Episode ", e)
    # Reset the env to start a new game
    state = np.zeros((22, 10, 1))
    agent.memory.clear()
    for time_t in tqdm(range(10000)):
        action = agent.action(state)
        observation = env.step(action)
        data = preprocess(observation)
        reward = observation[1]
        data = np.reshape(data, [1, *(22, 10, 1)])
        done = observation[2]
        plt.imshow(data[0, :, :, 0])
        plt.show()
        if observation[3]:
            print("Game over")
            break
        agent.remember(state, action, reward, data, done)
        state = data
        if done:  # game over
            print("Game Episode :{}/{} High Score :{} Exploration Rate:{:.2}".format(
                e, EPISODES, time_t, agent.epsilon))
            # agent.replay(len(agent.memory))
            # agent.memory.clear()
            break

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    agent.save(f"tetris-dqn-{e}.h5")
