import numpy as np
import pickle
 
BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
 
 
class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps  # probability of choosing random action instead of greedy
        self.alpha = alpha  # learning rate
        self.verbose = False
        self.values = dict()
        self.state_history = []
 
    def train(self, env):
        for hash in env.allStates.keys():
            (winner, isEnd) = env.allStates[hash]
            if isEnd:
                if winner == self.sym:
                    self.values[hash] = 1.0
                else:
                    self.values[hash] = 0
            else:
                self.values[hash] = 0.5
 
    def set_symbol(self, sym):
        self.sym = sym
 
    def set_verbose(self, v):
        # if true, will print values for each position on the board
        self.verbose = v
 
    def reset_history(self):
        self.state_history = []
 
    def take_action(self, env):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            # take a random action
            if self.verbose:
                print("Taking a random action")
            possible_moves = []
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
 
        else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            next_move = None
            best_value = -1
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    if env.is_empty(i, j):
                        # what is the state if we made this move?
                        env.board[i, j] = self.sym
                        state = env.get_hash
                        env.board[i, j] = 0  # don't forget to change it back!
                        if self.values[state] > best_value:
                            best_value = self.values[state]
                            next_move = (i, j)
 
        # make the move
        env.board[next_move[0], next_move[1]] = self.sym
 
    def update_state_history(self, s):
        # cannot put this in take_action, because take_action only happens
        # once every other iteration for each player
        # state history needs to be updated every iteration
        # s = env.get_state() # don't want to do this twice so pass it in
        self.state_history.append(s)
 
    def update(self, env):
        # we want to BACKTRACK over the states, so that:
        # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        # where V(next_state) = reward if it's the most current state
        #
        # NOTE: we ONLY do this at the end of an episode
        # not so for all the algorithms we will study
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.values[prev] + self.alpha * (target - self.values[prev])
            self.values[prev] = value
            target = value
        self.reset_history()
 
    def savePolicy(self):
        fw = open('optimal_policy_' + str(self.sym), 'wb')
        pickle.dump(self.values, fw)
        fw.close()
 
    def loadPolicy(self):
        fr = open('optimal_policy_' + str(self.sym), 'rb')
        self.values = pickle.load(fr)
        fr.close()
 
 
# this class represents a tic-tac-toe game
# is a CS101-type of project
class Environment:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.x = -1  # represents an x on the board, player 1
        self.o = 1  # represents an o on the board, player 2
        self.allStates = dict()
        self.winner = None
        self.ended = False
        self.num_states = 3 ** (BOARD_SIZE)
        self.hash = 0
 
    def is_empty(self, i, j):
        return self.board[i, j] == 0
 
    def clear(self):
        self.board = np.zeros((BOARD_ROWS,  BOARD_COLS))
        self.winner = None
        self.ended = False
 
    def reward(self, sym):
        # no reward until game is over
        if not self.game_over():
            return 0
 
        # if we get here, game is over
        # sym will be self.x or self.o
        return 1 if self.winner == sym else 0
 
    @property
    def get_hash(self):
        # returns the current state, represented as an int
        # from 0...|S|-1, where S = set of all possible states
        # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
        # some states are not possible, e.g. all cells are x, but we ignore that detail
        # this is like finding the integer represented by a base-3 number
        k = 0
        h = 0
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3 ** k) * v
                k += 1
        return h
 
    @staticmethod
    def decode_hash(hash):
        state = np.zeros((BOARD_ROWS, BOARD_COLS))
        index = 0
        while (hash > 0):
            i = index // BOARD_ROWS
            j = index % BOARD_COLS
            state[i, j] = hash % BOARD_COLS
            hash //= BOARD_ROWS
            index += 1
        return state
 
 
    def get_allStates(self):
        for hash, winner, isEnd in get_state_hash_and_winner(self):
            self.allStates[hash] = [winner, isEnd]
 
    def game_over(self, force_recalculate=False):
        # returns true if game over (a player has won or it's a draw)
        # otherwise returns false
        # also sets 'winner' instance variable and 'ended' instance variable
        if not force_recalculate and self.ended:
            return self.ended
 
        # check rows
        for i in range(BOARD_ROWS):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * BOARD_ROWS:
                    self.winner = player
                    self.ended = True
                    return True
 
        # check columns
        for j in range(BOARD_COLS):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * BOARD_COLS:
                    self.winner = player
                    self.ended = True
                    return True
 
        # check diagonals
        for player in (self.x, self.o):
            # top-left -> bottom-right diagonal
            if self.board.trace() == player * BOARD_ROWS:
                self.winner = player
                self.ended = True
                return True
            # top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player * BOARD_ROWS:
                self.winner = player
                self.ended = True
                return True
 
        # check if draw
        if np.all((self.board == 0) == False):
            # winner stays None
            self.winner = None
            self.ended = True
            return True
 
        # game is not over
        self.winner = None
        return False
 
    def is_draw(self):
        return self.ended and self.winner is None
 
    # Example board
    # -------------
    # | x |   |   |
    # -------------
    # |   |   |   |
    # -------------
    # |   |   | o |
    # -------------
    def draw_board(self):
        for i in range(BOARD_ROWS):
            print("-------------")
            for j in range(BOARD_ROWS):
                print("  ")
                if self.board[i, j] == self.x:
                    print("x ")
                elif self.board[i, j] == self.o:
                    print("o ")
                else:
                    print("  ")
            print("")
        print("-------------")
 
 
class Human:
    def __init__(self):
        pass
 
    def set_symbol(self, sym):
        self.sym = sym
 
    def take_action(self, env):
        while True:
            # break if we make a legal move
            move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break
 
    def update(self, env):
        pass
 
    def update_state_history(self, s):
        pass
 
 
# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
# impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
# since that will never happen in a real game
def get_state_hash_and_winner(env, i=0, j=0):
    results = []
 
    for v in (0, env.x, env.o):
        env.board[i, j] = v  # if empty board it should already be 0
        if j == 2:
            # j goes back to 0, increase i, unless i = 2, then we are done
            if i == 2:
                # the board is full, collect results and return
                state = env.get_hash
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            # increment j, i stays the same
            results += get_state_hash_and_winner(env, i, j + 1)
 
    return results
 
 
def play_game(p1, p2, env, draw=False):
    # loops until the game is over
    current_player = None
    while not env.game_over():
        # alternate between players
        # p1 always starts first
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
 
        # draw the board before the user who wants to see it makes a move
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()
 
        # current player makes a move
        current_player.take_action(env)
 
        # update state histories
        state = env.get_hash
        p1.update_state_history(state)
        p2.update_state_history(state)
 
    if draw:
        env.draw_board()
 
    # do the value function update
    p1.update(env)
    p2.update(env)
 
 
def train(epochs=10000):
    # train the agent
    p1 = Agent()
    p2 = Agent()
 
    # set initial V for p1 and p2
    env = Environment()
    env.get_allStates()
 
    # give each player their symbol and update states
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)
 
    p1.train(env)
    p2.train(env)
 
    for t in range(epochs):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, env)
        env.clear()
 
    p1.savePolicy()
    p2.savePolicy()
 
 
if __name__ == '__main__':
 
    train()
    # play human vs. agent
    # do you think the agent learned to play the game well?
    env = Environment()
    p1 = Agent()
    p1.set_symbol(env.x)
 
    human = Human()
    human.set_symbol(env.o)
    p1.loadPolicy()
 
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        # I made the agent player 1 because I wanted to see if it would
        # select the center as its starting move. If you want the agent
        # to go second you can switch the human and AI.
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break