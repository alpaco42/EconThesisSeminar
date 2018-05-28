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
from keras.regularizers import l2
#import matplotlib.pyplot as plt
# import matplotlib.animation
# import IPython.display

# parameters
NUMCOUNTRIES = 50
AVGPOP = 10000
AVGPRODCOST = 0.001
AVGPRODBASE = 50
DEMAND = 3
TARIFF = 0.1
epsilon = .05  # probability of exploration (choosing a random action instead of the current best one)
state_space = NUMCOUNTRIES ** 2 + 3 * NUMCOUNTRIES
action_space = 2
max_memory = 30000
hidden_size = int(action_space + (state_space - action_space)/ 2)
batch_size = 100
debug_data = []

class Country():

    def __init__(self):
        self.population = int(np.random.normal(AVGPOP, AVGPOP / 5))
        self.production_cost = np.random.normal(AVGPRODCOST, AVGPRODCOST / 5)
        self.production_base = np.random.normal(AVGPRODBASE, AVGPRODBASE / 5)
        self.demand_slope = DEMAND
        self.relative_gains = float()
        self.tariffs = {self:[0, False]}
        self.state = None
        self.countries = None
        self.global_inputs = None
        self.index = None
        self.new_tariffs = {self:[0, False]}

    @staticmethod
    def index(i, j, p):
        return i * p + j

    def tariffs(self):
        """I dont think I use this anymore..."""
        return sum([i[0] for i in self.tariffs.values()])

    def initialize(self, countries, global_inputs):
        self.index = countries.index(self)
        #global_inputs and countries have to be reordered so that the country setting tariffs is last.
        #This is only actually necessary for agent objects, since the actor object's format is what's being immitated
        self.global_inputs = global_inputs[:self.index * 3] + global_inputs[(self.index + 1) * 3:] + \
            global_inputs[self.index * 3: (self.index + 1) * 3]
        self.countries = countries[:self.index] + countries[self.index + 1:] + [self]
        for country in self.countries[:-1]:
            self.new_tariffs[country] = [TARIFF, round(random.random())]
        self.tariffs = self.new_tariffs

    def get_inputs(self, country):
        """reorders self.state and global_inputs so that the country's
        info that tariffs are being set on is always first"""
        p = NUMCOUNTRIES
        state = np.concatenate((self.state[p**2+p * country: p**2+p * (country+1)], self.state[p**2:p**2+p * country], \
            self.state[p**2 + p * (country + 1):]))
        inputs = self.global_inputs[3 * country: 3 * (country + 1)] + self.global_inputs[:3 * country] + \
            self.global_inputs[3 * (country + 1):]
        return np.concatenate((inputs, state))

    def evaluatePolicy(self, ws):
        """reorders the globally updated state (ws) to match the agent object in question"""
        p = NUMCOUNTRIES
        self.state = np.concatenate((ws[: p * self.index],  ws[p * (self.index+1): p**2], \
            ws[p * self.index: p * (self.index+1)], ws[p**2: p**2 + p * self.index], ws[p**2 + p * (self.index+1):], \
            ws[p**2+p * self.index: p**2+p * (self.index+1)]))

    def resolve_policies(self):
        """Determines which diads agree to FTAs"""
        for country in self.countries:
            if country.new_tariffs[self][1]:
                if self.new_tariffs[country][1]:
                    self.tariffs[country] = [0, True]
            else:
                self.tariffs[country] = [TARIFF, False]

class Actor(Country):
    """The object that's actually training"""

    def get_reward(self):
        """Reward is calculated as the average of observed producer surplus divided by producer surplus if the country's
        firm was a global monopoly and observed consumer surplus divded by the entire area under the demand curve"""


        debug_data = []

        p = NUMCOUNTRIES
        productions = self.state[:p**2]
        reshape = productions.reshape((p,p))
        consumptions = [sum(reshape[:,i]) for i in range(p)]
        prices = [(self.countries[i].population - consumptions[i]) / self.countries[i].demand_slope for i in range(p)]
        # print(prices)
        sales = 0
        for market in range(self.countries.index(self) * p, (self.countries.index(self) + 1) * p):
            sales += productions[market] * (prices[market%p] - self.countries[market%p].tariffs[self][0])

        producer_surplus = sales - self.production_cost * sum(productions[self.countries.index(self) * p:\
        (self.countries.index(self) + 1) * p])**2 / 2
        consumer_surplus = (self.population / self.demand_slope - prices[self.countries.index(self)]) * \
            consumptions[self.countries.index(self)] / 2
        max_consumption_surplus = self.population**2 / (2 * self.demand_slope)

        y = np.array([ (self.countries[country].tariffs[self][0] - 1) * \
            self.countries[country].population / self.countries[country].demand_slope + \
            self.production_base for country in range(p)])
        X = np.zeros((p,p))
        for market in range(p):
            for production in range(p):
                if production == market:
                    X[market, production] = -2 * (1 - self.countries[production].tariffs[self][0]) / \
                        self.countries[production].demand_slope  - self.production_cost
                else:
                    X[market, production] = -1 * self.production_cost
        mx_prd = np.linalg.solve(X,y)
        # for prod in mx_prd:
        #     if prod < 0:
        #         print("i", prod)
        mx_prices = [(self.countries[i].population - mx_prd[i]) / self.countries[i].demand_slope for i in range(p)]
        max_prod_surplus = sum([mx_prd[i] * mx_prices[i] * (1 - self.countries[i].tariffs[self][0]) for i in range(p)])
        max_prod_surplus -= self.production_cost * sum(mx_prd)**2 / 2 + self.production_base * sum(mx_prd)


        return producer_surplus / max_prod_surplus + consumer_surplus / max_consumption_surplus

class Agent(Country):
    """The object for the agents interacting with the model but not training"""

    def __init__(self, model=None):
        Country.__init__(self)
        self.model = model


    def set_policies(self, world_state):
        p = NUMCOUNTRIES
        self.evaluatePolicy(world_state)
        st = time.time()
        if self.model == None:
            policies = [1 for i in range(p-1)] #first time around, all other countries always want FTAs so that the
                                               #model's choices actually change the outcome
        else:
            policies = self.model.predict(np.array([self.get_inputs(country) for country in range(p - 1)]))
            policies = [np.argmax(policies[i]) for i in range(p-1)]

        self.new_tariffs = {self.countries[country]:[TARIFF, policies[country]] for country in range(p-1)}
        self.new_tariffs[self] = [0, False]



    def update_model(model):
        self.model = model

class World():

    def __init__(self, starting_model = None):
        self.countries = None
        self.state = None
        self.inputs_to_NN = None
        self.reset(starting_model)

    def display(self):
        for country in range(len(self.countries)):
            FTAs = sum([i[1] for i in self.countries[country].tariffs.values()])
            plt.bar(country, FTAs)
        plt.show(block = False)

    def reset(self, starting_model):
        self.countries = [Agent(starting_model) for country in range(NUMCOUNTRIES - 1)] + [Actor()]
        self.inputs_to_NN = []
        for country in self.countries:
            self.inputs_to_NN += [country.population, country.production_cost, country.production_base]
        for i in self.countries:
            i.initialize(self.countries, self.inputs_to_NN)
        for i in self.countries:
            i.resolve_policies()
        self._evaluatePolicy()
        for i in self.countries:
            i.evaluatePolicy(self.state)


    def _evaluatePolicy(self):
        """This function determines the optimal productions for every firm given the current tariff regimes by finding
        the Nash equilibrium"""
        #this is a sparse linalg implementation that turned out to be much much slower than regular linalg :(((
        # p = NUMCOUNTRIES
        # y = np.array([ (self.countries[country%p].tariffs[self.countries[int(country/p)]][0] - 1) * \
        # self.countries[country%p].population / self.countries[country%p].demand_slope + \
        # self.countries[int(country/p)].production_base for country in range(p**2)])
        # X = sparse.lil_matrix(np.zeros((p**2, p**2)))
        # for producer in range(p):
        #     for market in range(p):
        #         for i in range(p):
        #             for j in range(p):
        #                 if j == market:
        #                     if i == producer:
        #                         X[Country.index(producer, market, p), Country.index(i,j,p)] = -2 * \
        #                         (1 - self.countries[j].tariffs[self.countries[i]][0]) / self.countries[j].demand_slope \
        #                         - self.countries[i].production_cost
        #                     else:
        #                         X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * \
        #                         (1 - self.countries[j].tariffs[self.countries[i]][0]) / self.countries[j].demand_slope
        #                 elif i == producer:
        #                     X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * \
        #                     self.countries[i].production_cost
        # productions = np.maximum(linalg.spsolve(sparse.csc_matrix(X),y).flatten(), 0)
        # # productions = np.linalg.solve(X,y).flatten()
        # tariffs = np.array([[i[0] for i in list(country.tariffs.values())] for country in self.countries]).flatten()
        #
        # self.state = np.concatenate((productions, tariffs))
        # print (time.time() - st)





        p = NUMCOUNTRIES

        # this construction of the X matrix is more intuitive, but O(n^4) instead of O(n^3)
        # old_X = np.zeros((p**2,p**2))
        # for producer in range(p):
        #     for market in range(p):
        #         for i in range(p):
        #             for j in range(p):
        #                 if j == market:
        #                     if i == producer:
        #                         old_X[Country.index(producer, market, p), Country.index(i,j,p)] = -2 * \
        #                         (1 - self.countries[j].tariffs[self.countries[i]][0]) / self.countries[j].demand_slope \
        #                         - self.countries[i].production_cost
        #                     else:
        #                         old_X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * \
        #                         (1 - self.countries[j].tariffs[self.countries[i]][0]) / self.countries[j].demand_slope
        #                 elif i == producer:
        #                     old_X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * \
        #                     self.countries[i].production_cost
        #

        y = np.array([ (self.countries[country%p].tariffs[self.countries[int(country/p)]][0] - 1) * \
            self.countries[country%p].population / self.countries[country%p].demand_slope + \
            self.countries[int(country/p)].production_base for country in range(p**2)])
        X = np.zeros((p**2, p**2))
        for producer in range(p):
            for market in range(p):

                for i in range(p):
                    X[Country.index(producer, market, p), Country.index(i, market, p)] = -1 * \
                        (1 - self.countries[market].tariffs[self.countries[i]][0]) / self.countries[market].demand_slope
                X[Country.index(producer, market, p), Country.index(producer, market, p)] *= 2

                X[Country.index(producer, market, p), p * producer: p * (producer + 1)] -= \
                    [self.countries[producer].production_cost for j in range(p)]

        # if (old_X - X).any():
        #     global debug_data
        #     debug_data =  old_X - X
        #     raise RuntimeError("Xs not equal")
        productions = np.maximum(np.linalg.solve(X,y).flatten(), 0)
        tariffs = np.array([[i[0] for i in list(country.tariffs.values())] for country in self.countries]).flatten()
        self.state = np.concatenate((productions, tariffs))




        # for i in self.state:
        #     if i<0:
        #         print (i)

    def _update_state(self, actions):
        self._evaluatePolicy()
        self.countries[-1].new_tariffs = \
            {self.countries[-1].countries[i]:[TARIFF, actions[i]] for i in range(len(actions))}
        self.countries[-1].new_tariffs[self.countries[-1]] = [0, False]
        for country in self.countries[:-1]:
            country.set_policies(self.state)
        for country in self.countries:
            country.resolve_policies()

    def act(self, actions):
        self._update_state(actions)
        return self.countries[-1].get_reward()


class ExperienceReplay(object):
    def __init__(self, max_memory=500):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, states):
        '''
        Input:
            states: [starting_observation, action_taken, reward_received, new_observation]
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
        action_space = model.output_shape[-1]
        env_dim = len(self.memory[0][0])
        input_size = min(len_memory, batch_size)
        inputs = np.zeros((input_size, env_dim))
        targets = np.zeros((input_size, action_space))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=input_size)):
            starting_observation, action_taken, reward_received, new_observation = self.memory[idx]

            # Set the input to the state that was observed in the game before an action was taken
            inputs[i:i+1] = starting_observation

            # Start with the model's current best guesses about the value of taking each action from this state
            targets[i] = model.predict(starting_observation.reshape((1,state_space)))[0] #honestly i have no clue why i
                                                                                    #have to reshape but it works now

            targets[i, action_taken] = reward_received
        return inputs, targets


def build_model():
    '''
     Returns three initialized objects: the model, the environment, and the replay.
    '''

    model = Sequential()
    model.add(Dense(state_space, input_shape=(state_space,), activation='relu',kernel_regularizer=l2(0.0001)))
    model.add(Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dense(action_space, kernel_regularizer=l2(0.0001)))
    model.compile(sgd(lr=.04, clipvalue = 3.0), "mse")

    agent_model = Sequential()
    agent_model.add(Dense(state_space, input_shape=(state_space,), activation='relu',kernel_regularizer=l2(0.0001)))
    agent_model.add(Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.0001)))
    agent_model.add(Dense(action_space, kernel_regularizer=l2(0.0001)))
    agent_model.compile(sgd(lr=.04, clipvalue = 3.0), "mse")

    # Define environment/game
    env = World()

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)


    return model, agent_model, env, exp_replay


def train_model(model, agent_model, env, exp_replay, num_episodes, update_env = True):
    '''
    Inputs:
        model, env, and exp_replay objects as returned by build_model
        num_episodes: integer, the number of episodes that should be rolled out for training
    '''

    progress = []

    for episode in range(1, num_episodes + 1):

        #every 100 simulations, update the decision calculus of all other agents with the weights of the model
        if update_env and episode%100 == 0:
            agent_model.set_weights(model.get_weights())
            exp_replay.memory = list()

        loss = 0.
        if episode >= 100:
            env.reset(agent_model)

        for i in range(15):
            # get next action
            starting_observations = [env.countries[-1].get_inputs(country) for country in range(NUMCOUNTRIES - 1)]
            q = model.predict(np.array(starting_observations))
            actions = [np.argmax(q[i]) for i in range(NUMCOUNTRIES-1)]

            for action in range(len(actions)):
                if random.random() <= epsilon:
                    actions[action] = int(random.random())


            # apply action, get rewards and new state
            reward = env.act(actions)

            # store experience

            for country in range(NUMCOUNTRIES - 1):
                exp_replay.remember([starting_observations[country], actions[country], \
                    reward, env.countries[-1].get_inputs(country)])



            # get data updated based on the stored experiences
        inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

        # train model on the updated data
        loss += model.train_on_batch(inputs, targets)
        progress.append(loss)



        # Print update from this episode
        print("Episode {:04d}/{:04d} | Loss {:.4f}".format(episode, num_episodes, loss))

    return progress
model, agent_model, env, exp_replay = build_model()
progress = train_model(model, agent_model, env, exp_replay, num_episodes=999)
