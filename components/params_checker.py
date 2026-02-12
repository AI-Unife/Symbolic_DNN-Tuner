import random

from skopt.space import Integer, Real


class paramsChecker:
    def checker(self,i, dVector):
        if isinstance(i, Integer):
            dVector[i.name] = random.randint(i.low, i.high)
        else:
            if i.prior == 'uniform':
                dVector[i.name] = abs(random.uniform(i.low, i.high))
            elif i.prior == 'log-uniform':
                dVector[i.name] = abs(random.lognormvariate(i.low, i.high))
        return dVector

    def choice(self, space, toChange=None, params=None):
        dVector = {}
        for i in space:
            if toChange or params:
                if i.name in toChange:
                    dVector = self.checker(i,dVector)
                else:
                    dVector[i.name] = params[i.name]
            else:
                dVector = self.checker(i, dVector)
        return dVector
