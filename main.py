import numpy as np
from numpy.random import poisson


class LOB:
    def __init__(self, n, lambda_, theta_, mu_, timesteps):
        self.lr = 0.2
        self.gamma = 0.6
        self.alpha = 1
        self.a = 0.5
        self.b = 0.5
        self.r = 0.001
        self.n = n
        self.ask_price = self.n + 1
        self.bid_price = 0
        self.lambda_ = lambda_
        self.theta_ = theta_
        self.mu_ = mu_
        self.x = [0] * n
        self.t = 0
        self.inv = 0
        self.money = 0
        self.state = 0
        self.timesteps = timesteps
        self.agent_order = []
        self.Q = np.zeros((timesteps * 3, (n//2) * (n//2)))
        self.cur_timestep = 1
        self.bid_order_size = 100
        self.ask_order_size = 100
        self.old_value = 0
        self.old_inv = 0

    def place_limit_buy(self, price, count):
        if price < self.ask_price:
            self.x[price - 1] -= count

    def place_limit_sell(self, price, count):
        if price > self.bid_price:
            self.x[price - 1] += count

    def place_market_buy(self, count):
        if self.ask_price < self.n + 1:
            self.x[self.ask_price - 1] -= count

    def place_market_sell(self, count):
        if self.bid_price > 0:
            self.x[self.bid_price - 1] += count

    def cancel_limit_buy(self, price, count):
        if price < self.ask_price:
            self.x[price - 1] += count

    def cancel_limit_sell(self, price, count):
        if price > self.bid_price:
            self.x[price - 1] -= count

    def find_ask(self):
        """
        Finds ask price, i.e. smallest price at which someone is willing to sell.
        :return: the bid price
        """
        try:
            return [i for i in range(self.n) if self.x[i] > 0][0] + 1
        except IndexError:
            return self.n + 1

    def find_bid(self):
        """
        Finds bid price, i.e. largest price at which someone is willing to buy.
        :return: the bid price
        """
        try:
            return [i for i in range(self.n)[::-1] if self.x[i] < 0][0] + 1
        except IndexError:
            return 0

    def compute_instant_rates(self, price):
        """
        Finds current instantaneous rates of arrival for 4 non-homogenous poisson processes.
        :param price: price at which to compute the rates.
        :return: quadruple of rates.
        """
        lbr, lsr, clbr, clsr = 0, 0, 0, 0

        if price < self.ask_price:
            lbr = self.lambda_ * (self.ask_price - price)
            clbr = self.theta_ * (self.ask_price - price) * abs(self.x[price - 1])

        if price > self.bid_price:
            lsr = self.lambda_ * (price - self.bid_price)
            clsr = self.theta_ * (price - self.bid_price) * abs(self.x[price - 1])

        return lbr, lsr, clbr, clsr

    def place_price_orders(self, order_counts, price):
        """
        Places and cancels specified number of buy/cell orders at the given price.
        :param order_counts: array of shape (4,) of number of orders to place and cancel.
        :param price: price at which to place or cancel orders.
        """
        self.place_limit_buy(price, order_counts[0])
        self.place_limit_sell(price, order_counts[1])
        self.cancel_limit_buy(price, order_counts[2])
        self.cancel_limit_sell(price, order_counts[3])

    def place_market_orders(self, order_counts):
        """
        Places the specified numbers of market orders.
        :param order_counts:
        """
        self.place_market_buy(order_counts[0])
        self.place_market_sell(order_counts[1])

    def update(self, train=False):
        """
        Completes one iteration of simulation.
        """

        for price in range(1, self.n + 1):
            poisson_rates = self.compute_instant_rates(price)
            order_counts = poisson(poisson_rates)
            self.place_price_orders(order_counts, price)

        self.place_market_orders(poisson(self.mu_, 2))

        if self.ask_order_size > 0:
            try:
                if self.x[self.ask_price - self.agent_order[0] - 1] < 0:
                    vol = np.min([-self.x[self.ask_price - self.agent_order[0] - 1], self.ask_order_size])
                    self.x[self.ask_price - self.agent_order[0] - 1] += vol
                    self.inv += vol
                    self.ask_order_size -= vol
                    self.money -= (self.ask_price - self.agent_order[0]) * vol
            except IndexError:
                pass

        if self.bid_order_size > 0:
            try:
                if self.x[self.bid_price + self.agent_order[1] - 1] > 0:
                    vol = np.min([self.x[self.bid_price + self.agent_order[1] - 1], self.bid_order_size])

                    self.x[self.bid_price + self.agent_order[1] - 1] -= vol
                    self.inv -= vol
                    self.bid_order_size -= vol
                    self.money += (self.bid_price + self.agent_order[1]) * vol
            except IndexError:
                pass

        self.ask_price = self.find_ask()
        self.bid_price = self.find_bid()

        if train:
            if self.inv > 0:
                new_value = self.inv * self.bid_price + self.money
            else:
                new_value = self.inv * self.ask_price + self.money
            new_inv = self.inv
            self.update_qtable(new_value, self.old_value, new_inv, self.old_inv, self.agent_order)
            self.old_value = new_value
            self.old_inv = new_inv
        self.t += 1

    def update_cara(self):
        old_state = self.state
        profit = self.money
        liqiudate_money = 0
        if self.inv > 0:
            index = self.n - 1
            while self.inv > 0 or index != 1:
                if self.x[index] > 0:
                    vol = np.min([self.x[index], self.inv])
                    self.inv -= vol
                    liqiudate_money += (index + 1) * vol
                index -= 1

        if self.inv < 0:
            index = 0
            while self.inv > 0 or index != (self.n - 2):
                if self.x[index] < 0:
                    vol = np.min([-self.x[index], self.inv])
                    self.inv += vol
                    liqiudate_money -= (index + 1) * vol
                index += 1
        reward = self.alpha - np.exp(-self.r * (profit - liqiudate_money))
        for i in range(len(self.Q[old_state])):
            self.Q[old_state][i] = (1 - self.lr) * self.Q[old_state][i] + \
                                   self.lr * (reward + self.gamma * np.max(self.Q[self.state]))

        return profit + liqiudate_money

    def update_qtable(self, v_n, v_o, i_n, i_o, action, add_reward=0):
        index = action[0] * (self.n // 2) + action[1]
        old_state = self.state
        self.upd_state()
        reward = add_reward + self.a * (v_n - v_o) + \
                 np.exp(self.b * (self.timesteps - self.cur_timestep)) * np.sign(abs(i_n) - abs(i_o))
        try:
            self.Q[old_state][index] = (1 - self.lr) * self.Q[old_state][index] + \
                                   self.lr * (reward + self.gamma * np.max(self.Q[self.state]))
        except IndexError:
            pass

    def write_state(self, file):
        """
        Writes current state of the simulation to file
        :param file: file opened in append mode
        """
        file.write(','.join([str(num) for num in self.x]) + '\n')

    def run_simulation(self, sec_count, write=False):
        """
        Runs the simulation for specified number of steps.
        :param write: flag indicating if save data to file
        :param sec_count: steps to run.
        """
        if write:
            file = open('output.csv', 'a')
            file.truncate(0)
        for i in range(sec_count):
            self.update()
            if write:
                self.write_state(file)
        if write:
            file.close()

    def upd_state(self):
        self.state = np.minimum(self.inv // 200, 2) * 12 + (self.cur_timestep - 1)

    def choose_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n//2, 2)
        try:
            index = np.argmax(self.Q[self.state])
        except:
            pass
        bid_shift = index // (self.n // 2)
        ask_shift = index % (self.n // 2)

        return [bid_shift, ask_shift]

    def train_simulation(self, epsilon):
        """
        Runs the training simulation for specified number of timesteps.
        """
        for i in range(self.timesteps):
            self.upd_state()
            self.agent_order = self.choose_action(epsilon)
            self.update(True)
            for j in range(9):
                self.update()
            self.cur_timestep += 1

        return self.update_cara()

    def flush_simulation(self):
        """
        'flushes' the simulation by discarding the order book and setting time to 0
        :return:
        """
        self.x = [0] * self.n
        self.t = 0
        self.cur_timestep = 0
        self.ask_price = self.n + 1
        self.bid_price = 0
        self.money = 0
        self.inv = 0
        self.old_inv = 0
        self.old_value = 0

    def get_money(self):
        return self.money


lob = LOB(20, 1.85, 0.71, 2, 12)
lob.run_simulation(10000, True)
# episode_count = 5000
# profits = []
# for i in range(episode_count):
#     if i % 50 == 0:
#         print(f'{i//50}%')
#     lob.flush_simulation()
#     lob.run_simulation(300)
#     profit = lob.train_simulation(epsilon = 1 - i/episode_count)
#     if i%10 == 0:
#         profits.append(profit)
# print('Done')
#
# with open('profits.csv', 'w') as file:
#     file.write(','.join([str(num) for num in profits]) + '\n')
