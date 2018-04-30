import random
import numpy as np
import matplotlib.pyplot as plt
import time

#import json
#import numpy as np
#import random
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
#import matplotlib.pyplot as plt
# import matplotlib.animation
# import IPython.display


NUMCOUNTRIES = 20
AVGPOP = 100
AVGPRODCOST = 3
DEMAND = 3
# parameters
epsilon = .1  # probability of exploration (choosing a random action instead of the current best one)
state_space = NUMCOUNTRIES ** 4 + NUMCOUNTRIES ** 2
action_space = 2**(NUMCOUNTRIES - 1)
max_memory = 500
hidden_size = 100
batch_size = 50

class Country():

    def __init__(self):
        self.population = int(np.random.normal(AVGPOP, AVGPOP / 5))
        self.production_cost = np.random.normal(AVGPRODCOST, AVGPRODCOST / 5)
        self.demand_slope = DEMAND
        self.relative_gains = float()
        self.tariffs = {self:[0, False]}
        self.state = None
        self.countries = None
        self.new_tariffs = {self:[0, False]}

    @staticmethod
    def index(i, j, p):
        return i * p + j

    def tariffs(self):
        return sum([i[0] for i in list(self.tariffs.values())])

    def initialize(self, countries):
        index = countries.index(self)
        self.countries = countries[:index] + countries[index + 1:] + [self]
        for country in countries[:-1]:
            self.tariffs[country] = [10, round(random.random())]

    def _evaluatePolicy(self):
        p = NUMCOUNTRIES
        y = np.array([self.countries[int(country / p)].demand_slope * self.countries[int(country / p)].tariffs\
        [self.countries[country%p]][0] + self.countries[int(country / p)].population for country in range(p**2)])
        X = np.zeros((p**2, p**2))
        for producer in range(p):
            for market in range(p):
                for i in range(p):
                    for j in range(p):
                        if i == producer:
                            if j == market:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = 2 - 2 * \
                                self.countries[producer].production_cost * self.countries[market].demand_slope
                            else:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * self.countries\
                                [producer].production_cost * self.countries[market].demand_slope
                        elif j == market:
                            try:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = 2
                            except IndexError:
                                print (X.shape, Country.index(producer, market, p), Country.index(i,j,p))
        productions = np.maximum(np.linalg.solve(X,y), 0)
        tariffs = np.array([list(country.tariffs.values()) for country in self.countries]).flatten()
        self.state = np.concatenate((productions.flatten, tariffs))

    def resolve_policies(self):
        for country in self.countries:
            country.tariffs = country.new_tariffs
            if country != self:
                if country.new_tariffs[self][1]:
                    if self.tariffs[country][1]:
                        self.tariffs[country] = [0, True]
                else:
                    self.tariffs[country][1] = False

class Actor(Country):
    """The object for the model that's actually training"""

    def _get_reward(self):
        p = NUMCOUNTRIES
        productions = self.state[:p**2]
        consumptions = [sum(productions.reshape((p,p))[:,i]) for i in range(p)]
        prices = [(self.countries[i].population - consumptions[i]) / self.countries[i].demand_slope for i in range(p)]
        sales = 0
        for market in range(self.countries.index(self) * p, (self.countries.index(self) + 1) * p):
            sales += productions[market] * (prices[market%p] - self.countries[market%p].tariffs[self][0])

        producer_surplus = sales - self.production_cost * sum(productions[self.countries.index(self) * p:\
        (self.countries.index(self) + 1) * p])**2 / 2
        consumer_surplus = (self.population / self.demand_slope - prices[self.countries.index(self)]) * \
        consumptions[self.countries.index(self)] / 2

        return producer_surplus + consumer_surplus

class Agent(Country):
    """The object for the agents interacting with the model but not training"""

    def __init__(self, model):
        Country.__init__(self)
        self.model = model

    def set_policies(self):
        self._evaluatePolicy()
        t = self.model.predict(self.state)    #JEN? Not sure if the syntax is right here...
        self.new_tariffs = {self.countries[i]:[10, t[i]] for i in range(len(t))}
        if self.new_tariffs[self] != [0, False]:
            raise RuntimeError("Reflexive tariff policy is being adjusted")

    def update_model(model):
        self.model = model

class World():

    def __init__(self, starting_model):
        self.countries = [Agent(starting_model) for country in range(NUMCOUNTRIES - 1)] + [Actor()]
        self.reset()

    def display(self):
        for country in range(len(self.countries)):
            FTAs = sum([i[1] for i in self.countries[country].tariffs.values()])
            plt.bar(country, FTAs)
        plt.show(block = False)

    def reset(self):
        for i in self.countries:
            i.initialize(self.countries)
        for i in self.countries:
            i.resolve_policies()

    def _update_state(self, action):
        self.countries[-1].new_tariffs = {self.countries[-1].countries[i]:[10, action[i]] for i in range(len(action))}\
        + {self: [0, False]}
        for country in self.countries[:-1]:
            country.set_policies()
        for country in self.countries:
            country.resolve_policies()

    def _get_reward(self):
        return self.countries[-1]._get_reward()

    def act(self, action):
        self._update_state(self, action)
        observation = self.countries[-1].state
        reward = self.countries[-1]._get_reward()
        return observation, reward




class ExperienceReplay(object):
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, states):
        '''
        Input:
            states: [starting_observation, action_taken, reward_received, new_observation]
            game_over: boolean
        Add the states and game over to the internal memory array. If the array is longer than
        self.max_memory, drop the oldest memory
        '''
        self.memory.append(states)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        '''
        Randomly chooses batch_size memories, possibly repeating.
        For each of these memories, updates the models current best guesses about the value of taking a
            certain action from the starting state, based on the reward received and the model's current
            estimate of how valuable the new state is.
        '''
        len_memory = len(self.memory)
        action_space = model.output_shape[-1] # the number of possible actions
        env_dim = len(self.memory[0][0]) # the size of the state space --> @jen can i just make this state_space?
        input_size = min(len_memory, state_space) #@jen why isn't this env_dim?
        inputs = np.zeros((input_size, env_dim))
        targets = np.zeros((input_size, action_space))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=input_size)):
            starting_observation, action_taken, reward_received, new_observation = self.memory[idx][0]

            # Set the input to the state that was observed in the game before an action was taken
            inputs[i:i+1] = starting_observation

            # Start with the model's current best guesses about the value of taking each action from this state
            targets[i] = model.predict(starting_observation)[0]

            targets[i, action_taken] = reward_received
        return inputs, targets


def build_model():
    '''
     Returns three initialized objects: the model, the environment, and the replay.
    '''
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(state_space,), activation='relu')) #@jen is that comma supposed to be there?
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(action_space))
    model.compile(sgd(lr=.2), "mse")

    # Define environment/game
    env = World(model)  #JEN: I'm not sure if I can actually just give the untrained model as a starting model and it
                        #will correctly act as functionally random...

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    return model, env, exp_replay


def train_model(model, env, exp_replay, num_episodes):
    '''
    Inputs:
        model, env, and exp_replay objects as returned by build_model
        num_episodes: integer, the number of episodes that should be rolled out for training
    '''
    for episode in range(num_episodes):  #I've changed this from Jen's basket game such that countries go through a set
                                         #number of rounds of setting trade policies before the world is reset
        loss = 0.
        env.reset()
        # get initial input
        starting_observation = env.countries[-1].state

        for i in range(30):
            # get next action
            if np.random.rand() <= epsilon:
                # epsilon of the time, we just choose randomly
                action = [np.random.randint(2) for country in countries]
            else:
                # find which action the model currently thinks is best from this state
                q = model.predict(starting_observation)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            new_observation, reward = env.act(action)

            # store experience
            exp_replay.remember([starting_observation, action, reward, new_observation])

            # get data updated based on the stored experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # train model on the updated data
            loss += model.train_on_batch(inputs, targets)

            starting_observation = new_observation # for next time through the loop

        # Print update from this episode
        print("Episode {:04d}/{:04d} | Loss {:.4f}".format(episode, num_episodes-1, loss))
