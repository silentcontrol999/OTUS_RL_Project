class FixedOffsetMMAgent:

    def __init__(self, quantity, price_delta):
        self.quantity = quantity
        self.price_delta = price_delta
        self.rewards_history = []

    def choose_action(self, state):
        # get current best bid/best ask prices
        bid_price, ask_price = state["bid_ask"][-1]["bid_price"], state["bid_ask"][-1]["ask_price"]
        los = [
            {
                "price": bid_price - self.price_delta,
                "quantity": self.quantity,
                "buySell": "BUY"
            },
            {
                "price": ask_price + self.price_delta,
                "quantity": self.quantity,
                "buySell": "SELL"
            }
        ]
        return los

    def calculate_pnl_reward(self, state):
        self.rewards_history.append(
            {
                "received_time": state["bid_ask"][-1]["received_time"],
                "realized_pnl": state["realized_pnl"],
                "unrealized_pnl": state["unrealized_pnl"]
            }
        )
        return state["realized_pnl"]

    def exploit(self, env):
        state, done = env.reset()
        total_reward = 0

        while not done:
            action = self.choose_action(state)
            next_state, done = env.step(action)
            reward = self.calculate_pnl_reward(state)
            total_reward += reward
            state = next_state
        return total_reward
            