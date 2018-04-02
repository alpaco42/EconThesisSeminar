import random
import numpy as np
import matplotlib.pyplot as plt
import time
NUMCOUNTRIES = 20
AVGPOP = 100
AVGPRODCOST = 3
DEMAND = 3


class Country():

    def __init__(self):
        self.population = int(np.random.normal(AVGPOP, AVGPOP / 5))
        #self.production_cost = np.random.normal(AVGPRODCOST, AVGPRODCOST / 10)
        self.production_cost = AVGPRODCOST
        self.demand_slope = np.random.normal(DEMAND, DEMAND / 5)
        self.relative_gains = float()
        self.tariffs = {self:[0, False]}

    @staticmethod
    def index(i, j, p):
        return i * p + j

    def tariffs(self):
        return sum([i[0] for i in list(self.tariffs.values())])

    def __evaluateTariff(self,other_country, tariff):
        total_tariffs = self.tariffs() - self.tariffs[other_country][0] + tariff
        price = (self.production_cost * self.population + total_tariffs) / (len(self.tariffs) + self.demand_slope)
        demand = self.population - self.demand_slope * price

        producer_surplus = price**2 / (2 * self.production_cost)
        consumer_surplus = demand * (self.population - demand) / 2

        return producer_surplus + consumer_surplus

    def __optimizeTariff(self, other_country, debug = False):
        returns = [(i, self.__evaluateTariff(other_country, i)) for i in range(11)]
        if debug:
            for i in returns:
                plt.bar(*i)
            plt.show()
        return max(returns)

    def __evaluateFreeTrade(self, other_country):
        return self.__evaluateTariff(other_country, 0)

    def __optimizeWelfare(self, other_country):
        new_tariff, welfare = self.__optimizeTariff(other_country)
        return [new_tariff, welfare < self.__evaluateFreeTrade(other_country)]

    def set_policies(self, other_countries):
        for country in other_countries:
            if country != self:
                self.tariffs[country] = self.__optimizeWelfare(country)

    def resolve_policies(self, other_countries):
        for country in other_countries:
            if country != self:
                if country.tariffs[self][1]:
                    if self.tariffs[country][1]:
                        self.tariffs[country] = [0, True]
                else:
                    self.tariffs[country][1] = False


class World():

    def __init__(self):
        self.countries = [Country() for country in range(NUMCOUNTRIES)]
        for country in self.countries:
            for other_country in self.countries:
                if other_country != country:
                    country.tariffs[other_country] = [5, False]

    def update_policies(self):
        for country in self.countries:
            country.set_policies(self.countries)
        for country in self.countries:
            country.resolve_policies(self.countries)


    def __evaluatePolicy(self):
        p = len(self.countries)
        y = np.array([self.countries[int(country / p)].demand_slope * self.countries[int(country / p)]\
        .tariffs(self.countries[country%p])[0] + self.countries[int(country / p)].population\
        for country in range(p**2)])
        X = np.zeros((p, p))
        for producer in range(p):
            for market in range(p):
                for i in range(p):
                    for j in range(p):
                        if i = producer:
                            if j = market:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = 2 - 2 * \
                                self.countries[producer].production_cost * self.countries[market].demand_slope
                            else:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * self.countries\
                                [producer].production_cost * self.countries[market].demand_slope
                        elif j = market:
                            X[Country.index(producer, market, p), Country.index(i,j,p)] = 2
        production = np.linalg.solve(X, y)


    def display(self):
        for country in range(len(self.countries)):
            FTAs = sum([i[1] for i in self.countries[country].tariffs.values()])
            plt.bar(country, FTAs)
        plt.show(block = False)

    def evolve(self, epochs):
        for i in range(epochs):
            self.display()
            time.sleep(3)
            self.update_policies()
            plt.close()
        self.display()



class Catch(object):
    def __init__(self, grid_size=10):
        '''
        Input: grid_size (length of the side of the canvas)

        Initializes internal state.
        '''
        self.grid_size = grid_size
        self.min_basket_center = 1
        self.max_basket_center = self.grid_size-2
        self.reset()

    def _update_state(self, action):
        '''
        Input: action (0 for left, 1 for stay, 2 for right)

        Moves basket according to action. Moves fruit down. Updates state to reflect these movements
        '''
        if action == 0:  # left
            movement = -1
        elif action == 1:  # stay
            movement = 0
        elif action == 2: # right
            movement = 1
        else:
            raise Exception('Invalid action {}'.format(action))
        fruit_x, fruit_y, basket_center = self.state
        # move the basket unless this would move it off the edge of the grid
        new_basket_center = min(max(self.min_basket_center, basket_center + movement), self.max_basket_center)
        # move fruit down
        fruit_y += 1
        out = np.asarray([fruit_x, fruit_y, new_basket_center])
        self.state = out

    def _draw_state(self):
        '''
        Returns a 2D numpy array with 1s (white squares) at the locations of the fruit and basket and
        0s (black squares) everywhere else.
        '''
        im_size = (self.grid_size, self.grid_size)
        canvas = np.zeros(im_size)

        fruit_x, fruit_y, basket_center = self.state
        canvas[fruit_y, fruit_x] = 1  # draw fruit
        canvas[-1, basket_center-1:basket_center + 2] = 1  # draw 3-pixel basket
        return canvas

    def _get_reward(self):
        '''
        Returns 1 if the fruit was caught, -1 if it was dropped, and 0 if it is still in the air.
        '''
        fruit_x, fruit_y, basket_center = self.state
        if fruit_y == self.grid_size-1:
            if abs(fruit_x - basket_center) <= 1:
                return 1 # it caught the fruit
            else:
                return -1 # it dropped the fruit
        else:
            return 0 # the fruit is still in the air

    def observe(self):
        '''
        Returns the current canvas, as a 1D array.
        '''
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        '''
        Input: action (0 for left, 1 for stay, 2 for right)

        Returns:
            current canvas (as a 1D array)
            reward received after this action
            True if game is over and False otherwise
        '''
        self._update_state(action)
        observation = self.observe()
        reward = self._get_reward()
        game_over = (reward != 0) # if the reward is zero, the fruit is still in the air
        return observation, reward, game_over

    def reset(self):
        '''
        Updates internal state
            fruit in a random column in the top row
            basket center in a random column
        '''
        fruit_x = random.randint(0, self.grid_size-1)
        fruit_y = 0
        basket_center = random.randint(self.min_basket_center, self.max_basket_center)
        self.state = np.asarray([fruit_x, fruit_y, basket_center])


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        '''
        Input:
            states: [starting_observation, action_taken, reward_received, new_observation]
            game_over: boolean
        Add the states and game over to the internal memory array. If the array is longer than
        self.max_memory, drop the oldest memory
        '''
        self.memory.append([states, game_over])
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
        num_actions = model.output_shape[-1] # the number of possible actions
        env_dim = self.memory[0][0][0].shape[1] # the number of pixels in the image
        input_size = min(len_memory, batch_size)
        inputs = np.zeros((input_size, env_dim))
        targets = np.zeros((input_size, num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=input_size)):
            starting_observation, action_taken, reward_received, new_observation = self.memory[idx][0]
            game_over = self.memory[idx][1]

            # Set the input to the state that was observed in the game before an action was taken
            inputs[i:i+1] = starting_observation

            # Start with the model's current best guesses about the value of taking each action from this state
            targets[i] = model.predict(starting_observation)[0]

            # Now we need to update the value of the action that was taken
            if game_over:
                # if the game is over, give the actual reward received
                targets[i, action_taken] = reward_received
            else:
                # if the game is not over, give the reward received (always zero in this particular game)
                # plus the maximum reward predicted for state we got to by taking this action (with a discount)
                Q_sa = np.max(model.predict(new_observation)[0])
                targets[i, action_taken] = reward_received + self.discount * Q_sa
        return inputs, targets


# parameters
epsilon = .1  # probability of exploration (choosing a random action instead of the current best one)
num_actions = 3  # [move_left, stay, move_right]
max_memory = 500
hidden_size = 100
batch_size = 50
grid_size = 10


def build_model():
    '''
     Returns three initialized objects: the model, the environment, and the replay.
    '''
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # Define environment/game
    env = Catch(grid_size)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    return model, env, exp_replay


def train_model(model, env, exp_replay, num_episodes):
    '''
    Inputs:
        model, env, and exp_replay objects as returned by build_model
        num_episodes: integer, the number of episodes that should be rolled out for training
    '''
    catch_count = 0
    for episode in range(num_episodes):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        starting_observation = env.observe()

        while not game_over:
            # get next action
            if np.random.rand() <= epsilon:
                # epsilon of the time, we just choose randomly
                action = np.random.randint(0, num_actions, size=1)
            else:
                # find which action the model currently thinks is best from this state
                q = model.predict(starting_observation)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            new_observation, reward, game_over = env.act(action)
            if reward == 1:
                catch_count += 1

            # store experience
            exp_replay.remember([starting_observation, action, reward, new_observation], game_over)

            # get data updated based on the stored experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # train model on the updated data
            loss += model.train_on_batch(inputs, targets)

            starting_observation = new_observation # for next time through the loop

        # Print update from this episode
        print("Episode {:04d}/{:04d} | Loss {:.4f} | Catch count {}".format(episode, num_episodes-1, loss, catch_count))


def create_animation(model, env, num_games):
    '''
    Inputs:
        model and env objects as returned from build_model
        num_games: integer, the number of games to be included in the animation

    Returns: a matplotlib animation object
    '''
    # Animation code from
    # https://matplotlib.org/examples/animation/dynamic_image.html
    # https://stackoverflow.com/questions/35532498/animation-in-ipython-notebook/46878531#46878531

    # First, play the games and collect all of the images for each observed state
    observations = []
    for _ in range(num_games):
        env.reset()
        observation = env.observe()
        observations.append(observation)
        game_over = False
        while game_over == False:
            q = model.predict(observation)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            observation, reward, game_over = env.act(action)
            observations.append(observation)

    fig = plt.figure()
    image = plt.imshow(np.zeros((grid_size, grid_size)),interpolation='none', cmap='gray', animated=True, vmin=0, vmax=1)

    def animate(observation):
        image.set_array(observation.reshape((grid_size, grid_size)))
        return [image]

    animation = matplotlib.animation.FuncAnimation(fig, animate, frames=observations, blit=True, )
    return animation
