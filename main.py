import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
from deepq import DQNAgent
import gymnasium
from tqdm import tqdm

# env = gymnasium.make("ALE/Tetris-v5", obs_type='grayscale', render_mode='rgb_array')
env = gymnasium.make("ALE/Tetris-v5", obs_type='grayscale', render_mode='human')

image_crop = (24, 64, 27, 203)

tetrominoes = {
    'I': np.array([[1, 1, 1, 1]]),
    'L': np.array([[1, 1, 1], [1, 0, 0]]),
    'J': np.array([[1, 1, 1], [0, 0, 1]]),
    'O': np.array([[1, 1], [1, 1]]),
    'S': np.array([[0, 1, 1], [1, 1, 0]]),
    'T': np.array([[1, 1, 1], [0, 1, 0]]),
    'Z': np.array([[1, 1, 0], [0, 1, 1]])
}

tetrominoes_with_rotations = {
    'I': [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])],
    'L': [np.array([[1, 1, 1], [1, 0, 0]]), np.array([[1, 0], [1, 0], [1, 1]]),
          np.array([[0, 0, 1], [1, 1, 1]]), np.array([[1, 1], [0, 1], [0, 1]])],
    'J': [np.array([[1, 1, 1], [0, 0, 1]]), np.array([[1, 0], [1, 0], [1, 1]]),
          np.array([[1, 1, 1], [1, 0, 0]]), np.array([[1, 1], [0, 1], [0, 1]])],
    'O': [np.array([[1, 1], [1, 1]])],
    'S': [np.array([[0, 1, 1], [1, 1, 0]]), np.array([[1, 0], [1, 1], [0, 1]])],
    'T': [np.array([[1, 1, 1], [0, 1, 0]]), np.array([[1, 0], [1, 1], [1, 0]]),
          np.array([[0, 1], [1, 1], [0, 1]]), np.array([[0, 1, 0], [1, 1, 1]])],
    'Z': [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[0, 1], [1, 1], [1, 0]])]
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

agent = DQNAgent()

EPISODES = 500
BATCH_SIZE = 32
pieces_dict = {0: "I", 1: "J", 2: "L", 3: "O", 4: "S", 5: "T", 6: "Z"}


def preprocess(observation):
    data = observation[0][image_crop[2]:image_crop[3], image_crop[0]:image_crop[1]]
    data = data != 111
    data = data[1::8, ::4]
    return data


def new_piece_in_top_layers(data):
    for offset in range(-3, 5):
        for piece in tetrominoes:
            if piece == "I":
                if offset == 4:
                    continue
            for rotation in tetrominoes_with_rotations[piece]:
                new_data = np.zeros((4, 10))
                for en, line in enumerate(rotation):
                    new_data[en, (tetrominoes_spawn_positions[piece] + offset):(
                                tetrominoes_spawn_positions[piece] + len(line) + offset)] = line
                    if np.array_equal(data[:4], new_data):
                        return True, piece
    return False, None


# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots()
# fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
datas = []
done = False
b = 0


def is_average_heigh_lower(data, last_data):
    """
    Check if the average height of the board is lower than the last data
     ponderate the sum of row by the height of the row
    :param data:
    :param last_data:
    :return:
    """
    if np.sum(data) == 0:
        return 0
    last_data = np.reshape(last_data, [22, 10]) * np.reshape(np.arange(1, 23), (22, 1))
    data = data * np.reshape(np.arange(1, 23), (22, 1))
    if np.sum(data) < np.sum(last_data):
        return 1
    return 0


def get_reward_from_mean_height(data, before_run_state):
    """
    Get the reward from the mean height of the board and compare it to the last state
    """
    for i in range(22):
        if np.sum(data[i]) == 0:



for e in range(EPISODES):
    env.reset()
    env.render()
    print("Episode ", e)
    # Reset the env to start a new game
    state = np.zeros((1, 22, 10, 1))
    agent.memory.clear()
    a = 0
    before_run_state = np.zeros((22, 10))
    for time_t in tqdm(range(10000)):
        action = agent.action(state)
        # if action[1]:
        #     datas.extend([action[0]])
        #     ax.clear()
        #     ax.hist(datas, bins=np.arange(0, 5) - 0.5, edgecolor='black')
        #     ax.set_title('Updating Histogram')
        #     ax.set_xlabel('Value')
        #     ax.set_ylabel('Frequency')
        #     ax.set_xticks(range(0, 4))
        #     plt.draw()
        #
        # # plotting loss and accuracy
        # ax2.clear()
        # ax2.plot(agent.truc["loss"], label="loss")
        # ax2.set_title('Loss')
        # ax2.set_xlabel('Epoch')
        # ax2.set_ylabel('Loss')
        # plt.draw()
        #
        # ax3.clear()
        # ax3.plot(agent.truc["accuracy"], label="accuracy")
        # ax3.set_title('Accuracy')
        # ax3.set_xlabel('Epoch')
        # ax3.set_ylabel('Accuracy')
        # plt.draw()
        # plt.pause(0.1)
        a += 1
        observation = env.step(0)
        data = preprocess(observation)

        reward = observation[1]
        new_piece, piece = new_piece_in_top_layers(data)
        if new_piece:
            last_detection = time_t
            if time_t - last_detection < 10:
                pass
            else:
                curent_piece = piece
                added_reward = get_reward_from_mean_height(data, before_run_state)
                before_run_state = np.reshape(state.copy(), [22, 10])

        data = np.reshape(data, [1, *(22, 10, 1)])
        done = observation[2]
        if observation[3]:
            print("Game over")
            break
        agent.remember(state, action[0], reward, data, done)
        state = data
        if done:  # game over
            print("Game Episode :{}/{} High Score :{} Exploration Rate:{:.2}".format(
                e, EPISODES, time_t, agent.epsilon))
            # agent.replay(len(agent.memory))
            # agent.memory.clear()
            break
    agent.save(f"tetris-dqn-{e}.h5")
plt.show()
