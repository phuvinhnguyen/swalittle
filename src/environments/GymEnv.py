import gym

class GymEnv:
    def __init__(self, env_name="CartPole-v1"):
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        output = self.env.step(action)
        return output[0], output[1], output[2], output[4]

    def get_dims(self):
        return self.env.observation_space.shape[0], self.env.action_space.n