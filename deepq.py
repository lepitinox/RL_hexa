import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.src.layers.pooling.base_pooling2d import Pooling2D
from tqdm import tqdm

# Parameters
ACTIONS = [0, 1, 2, 3, 4]
STATE_SIZE = (22, 10, 1)  # grayscale image size
ACTION_SIZE = len(ACTIONS)


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        # Neural Networks for Deep Q Learning

        model = Sequential()

        # start of network
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=STATE_SIZE))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        # layer that collapses each column into a single pixel with 64 feature channels
        model.add(Conv2D(64, kernel_size=(1, STATE_SIZE[1]), activation='relu'))
        model.add(BatchNormalization())

        # continue network
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())

        # Fully-connected layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.75))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.75))

        # Output layer depending on the required output dimensions
        model.add(Dense(ACTION_SIZE, activation='softmax'))  # FC-13

        opt = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # Get all states from minibatch
        update_input = np.array([transition[0][0] for transition in minibatch])
        update_target = self.model.predict(update_input, verbose=0, use_multiprocessing=True, workers=4)

        # Get all next states from minibatch
        update_input_next_state = np.array([transition[3][0] for transition in minibatch])
        target_val = self.model.predict(update_input_next_state, verbose=0, use_multiprocessing=True, workers=4)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            true_q = reward if done else (reward + self.gamma * np.amax(target_val[i]))
            update_target[i][action] = true_q

        self.model.fit(update_input, update_target, batch_size=batch_size, verbose=0, shuffle=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory.clear()

    def old_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
