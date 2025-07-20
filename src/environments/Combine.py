import random
import importlib

class CombineEnv:
    def __init__(self, **kwargs):
        self.envs = []
        for v in kwargs.values():
            self.envs.append({
                'class': v['class'],
                'params': v['params'],
                'prob': v['probability']
            })

    def step(self, **kwargs):
        return self.env.step(**kwargs)

    def reset(self):
        del self.env
        env = random.Random.choices(self.envs, weights=[i['prob'] for i in self.envs])
        self.env = importlib.import_module(f'environments.{env["class"]}')(**env['params'])
        return self.env.reset()