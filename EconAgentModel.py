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
        self.tariffs = dict()

    def __evaluateTariff(self,other_country, tariff):
        total_tariffs = sum([i[0] for i in list(self.tariffs.values())]) - self.tariffs[other_country][0] + tariff
        price = (self.production_cost * self.population + total_tariffs) / ((len(self.tariffs) + 1) + self.demand_slope)
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
                if country.tariffs[self][1] == True:
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
