import numpy as np


class Tetromino:


    def __init__(self, shape):
        self.shape = shape
        self.rotation = 0
        self.height = shape.shape[0]
        self.width = shape.shape[1]

    def rotate(self):
        self.shape = np.rot90(self.shape)
        self.height = self.shape.shape[0]
        self.width = self.shape.shape[1]


class Tetris:

    def __init__(self):
        self.board = np.zeros((20, 10))
        self.piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.score = 0
        self.cleared_lines = 0
        self.game_over = False
