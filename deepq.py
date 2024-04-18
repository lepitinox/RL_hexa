import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from tqdm import tqdm

# Parameters
ACTIONS = [0, 1, 2, 3]
STATE_SIZE = (22, 10, 1)  # grayscale image size
ACTION_SIZE = len(ACTIONS)
BATCH_SIZE = 32


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.model = self.build_model()
        self.truc = {"loss": [], "accuracy": []}

    def build_model(self):
        # Neural Networks for Deep Q Learning

        model = Sequential()

        model.add(Flatten(input_shape=STATE_SIZE))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.75))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.75))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        # Output layer depending on the required output dimensions
        model.add(Dense(ACTION_SIZE, activation='softmax'))

        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE), False
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0]), True

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # Get all states from minibatch
        update_input = np.array([transition[0][0] for transition in minibatch])
        update_target = self.model.predict(update_input, verbose=0, use_multiprocessing=True, workers=8)

        # Get all next states from minibatch
        update_input_next_state = np.array([transition[3][0] for transition in minibatch])
        target_val = self.model.predict(update_input_next_state, verbose=0, use_multiprocessing=True, workers=8)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                true_q = reward
            else:
                true_q = (reward + self.gamma * np.amax(target_val[i]))
            update_target[i][action] = true_q

        ok = self.model.fit(update_input, update_target, batch_size=batch_size, verbose=0, shuffle=False)
        self.truc['loss'].append(ok.history['loss'][0])
        self.truc['accuracy'].append(ok.history['accuracy'][0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
