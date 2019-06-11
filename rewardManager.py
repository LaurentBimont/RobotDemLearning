class Grasp(object):
    def __init__(self):
        super(Grasp, self).__init__()
        self.grasp_success = True             # To change when Q-learning is enable

    def check_success(self):
        # To be written
        pass

    def reward(self):
        if self.grasp_success:
            return 1
        else:
            return 0


class RewardManager(object):
    def __init__(self):
        self.future_reward_discount = 0.8
        self.grasp = Grasp()
        self.reward = self.grasp.reward()

    def compute_reward(self, state, future_reward):
        # Compute current reward
        action_reward = self.reward
        if not state.grasp_success:
            future_reward = 0
        # print('Current reward: {}'.format(action_reward))
        # print('Future reward: {}'.format(future_reward))
        expected_reward = action_reward + self.future_reward_discount * future_reward
        # print('Expected reward: {} + {} x {} = {}'.format(action_reward, self.future_reward_discount, future_reward, expected_reward))
        return expected_reward, action_reward


if __name__ == '__main__':
    RM = RewardManager()
