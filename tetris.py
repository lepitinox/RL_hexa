import numpy as np


class Tetris:

    def __init__(self):
        self.board_shape = (22, 10)
        self.board = np.zeros(self.board_shape)
        self.tetrominoes = {
            'I': np.array([[1, 1, 1, 1]]),
            'J': np.array([[1, 0, 0], [1, 1, 1]]),
            'L': np.array([[0, 0, 1], [1, 1, 1]]),
            'O': np.array([[1, 1], [1, 1]]),
            'S': np.array([[0, 1, 1], [1, 1, 0]]),
            'T': np.array([[0, 1, 0], [1, 1, 1]]),
            'Z': np.array([[1, 1, 0], [0, 1, 1]])
        }
        self.tetrominoes_order = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']

    def reset(self):
        self.board = np.zeros(self.board_shape)
        return self.board
