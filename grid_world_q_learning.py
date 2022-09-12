import numpy as np

#global variables
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0,3)
LOSE_STATE = (1,3)
START = (2,0)
DETERMINISTIC = False

class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1,1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC


    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0
    
    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nextPosition(self, action):
        """
            action: up, down, left, right
            -------------
            0 | 1 | 2| 3|
            1 |
            2 |
            return next position
        """
        if self.determine:
            if action == "up":
                nextState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nextState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nextState = (self.state[0], self.state[1] - 1)
            else:
                nextState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            #non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nextState = self.nextPosition(action)

        #check if nextState is legal
        if (nextState[0] >= 0) and (nextState[0] <= BOARD_ROWS-1):
            if (nextState[1] >= 0) and (nextState[1] <= BOARD_COLS-1):
                if nextState != (1,1):
                    return nextState
        
        return self.state
    
    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')



class Agent:
    def __init__(self):
        self.states = [] #record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.2 #learning rate
        self.exp_rate = 0.3 #epsilon, exploration rate
        self.decay_gamma = 0.9 #discount factor

        #initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i,j)] = {}

                for a in self.actions:
                    self.Q_values[(i,j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        #choose action with most expected value
        max_next_reward = 0
        action = ""

        if np.random.uniform(0,1) <= self.exp_rate:
            action = np.random.choice(self.actions)
            print(f"Random action: {action} in chooseAction")
        else:
            #greedy action
            for a in self.actions:
                current_position = self.State.state
                next_reward = self.Q_values[current_position][a]
                if next_reward >= max_next_reward:
                    action = a
                    max_next_reward = next_reward
            print(f"max_reward: {max_next_reward} with non-random action {action}")
        
        return action

    def takeAction(self, action):
        position = self.State.nextPosition(action)
        #update state
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd
    
    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)

                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()

                #append trace
                self.states.append([(self.State.state), action])
                print(f"current position: {self.State.state}\taction: {action}")

                #by taking the action, it reaches the next state
                self.State = self.takeAction(action)

                #mark is end
                self.State.isEndFunc()
                print("next state", self.State.state)
                print("---------------------------")
                self.isEnd = self.State.isEnd

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    print("Initial Q-values ... \n")
    print(ag.Q_values)

    ag.play()
    print("Initial Q-values ... \n")
    print(ag.Q_values)

