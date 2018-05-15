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

    def _evaluatePolicy(self, countries):
        p = len(countries)
        y = np.array([countries[int(country / p)].demand_slope * countries[int(country / p)]\
        .tariffs[countries[country%p]][0] + countries[int(country / p)].population for country in range(p**2)])
        X = np.zeros((p**2, p**2))
        for producer in range(p):
            for market in range(p):
                for i in range(p):
                    for j in range(p):
                        if i == producer:
                            if j == market:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = 2 - 2 * \
                                countries[producer].production_cost * countries[market].demand_slope
                            else:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = -1 * countries[producer]\
                                .production_cost * countries[market].demand_slope
                        elif j == market:
                            try:
                                X[Country.index(producer, market, p), Country.index(i,j,p)] = 2
                            except IndexError:
                                print (X.shape, Country.index(producer, market, p), Country.index(i,j,p))
        productions = np.maximum(np.linalg.solve(X,y), 0)
        consumptions = [sum(productions.reshape((p,p))[:,countries.index(i)]) for i in range(p)]
        prices = [(countries[i].population - consumptions[i]) / countries[i].demand_slope for i in range(p)]
        sales = 0
        for market in range(countries.index(self) * p, (countries.index(self) + 1) * p):
            sales += productions[market] * (prices[market%p] - countries[market%p].tariffs(self)[0])

        producer_surplus = sales - self.production_cost * sum(productions[countries.index(self) * p:\
        (countries.index(self) + 1) * p])**2 / 2
        consumer surplus = (self.population / self.demand_slope - prices[countries.index(self)]) * \
        consumptions[countries.index(self)] / 2
        tariffs = np.array([list(country.tariffs.values()) for country in countries]).flatten()

        state = np.concatenate((productions.flatten, tariffs))
        reward = producer_surplus + consumer_surplus

        return state, reward


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
