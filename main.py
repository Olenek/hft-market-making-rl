from numpy.random import poisson


class LOB:
    def __init__(self, n, lambda_, theta_, mu_):
        self.n = n
        self.ask_price = self.n + 1
        self.bid_price = 0
        self.lambda_ = lambda_
        self.theta_ = theta_
        self.mu_ = mu_
        self.x = [0] * n
        self.t = 0

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

    def update(self):
        """
        Completes one iteration of simulation.
        """
        for price in range(1, self.n + 1):
            poisson_rates = self.compute_instant_rates(price)
            order_counts = poisson(poisson_rates)
            self.place_price_orders(order_counts, price)

        self.place_market_orders(poisson(self.mu_, 2))
        self.t += 1
        self.ask_price = self.find_ask()
        self.bid_price = self.find_bid()

    def write_state(self, file):
        """
        Writes current state of the simulation to file
        :param file: file opened in append mode
        """
        file.write(','.join([str(num) for num in self.x]) + '\n')

    def run_simulation(self, step_count, write=False):
        """
        Runs the simulation for specified number of steps.
        :param write: flag indicating if save data to file
        :param step_count: steps to run.
        """
        if write:
            file = open('output.csv', 'a')
            file.truncate(0)
        for i in range(step_count):
            self.update()
            if write:
                self.write_state(file)
        if write:
            file.close()

    def flush_simulation(self):
        """
        'flushes' the simulation by discarding the order book and setting time to 0
        :return:
        """
        self.x = [0] * self.n
        self.t = 0
        self.ask_price = self.n + 1
        self.bid_price = 0


lob = LOB(20, 1.85, 0.71, 2)
lob.run_simulation(10000, True)
