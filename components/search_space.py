from skopt.space import Integer, Real, Categorical


class search_space:
    """
    class used to define the search space for hyper-parameters,
    which will be those that will be optimised during iterations
    """
    def __init__(self):
        """
        initialisation of attributes, used when adding hyperparameters
        in order to define their range of values
        """
        self.epsilon_r1 = 10 ** -3
        self.epsilon_r2 = 10 ** 2
        self.epsilon_i = 2
        self.epsilon_d = 4

    def search_sp(self):
        """
        method used to define the search space, which reflects the structure of the neural network
        :return: hyperparameters search space
        """
        self.search_space = [
            Integer(16, 64, name='unit_c1'),
            Real(0.002, 0.3, name='dr1_2'),
            Integer(64, 128, name='unit_c2'),
            Integer(256, 512, name='unit_d'),
            Real(0.03, 0.5, name='dr_f'),
            Real(1e-4, 1e-3, name='learning_rate'),
            Integer(16, 256, name='batch_size'),
            Categorical(['Adam', 'Adamax', 'Adagrad', 'Adadelta'], name='optimizer'),
            Categorical(['relu', 'elu', 'selu', 'swish'], name='activation')
        ]

        return self.search_space

    def add_params(self, params):
        """
        method used for adding hyperparameters in the search space
        :param params: parameters to be added to the search space
        :return: new search space with added hyperparameters
        """
        # initialize the accumulator of new hyperparameters as an empty list
        new_Hp = []

        # iter on all parameters to be added
        for p in params.keys():
            # define the hyperparameter type and range, using the initial attributes for upper and lower range
            if type(params[p]) == float:
                np = Real(abs(params[p] / self.epsilon_r2), (params[p] / self.epsilon_r1), name=p)
                new_Hp.append(np)
            elif type(params[p]) == int:
                if p == 'new_fc':
                    np = Integer(abs(int(params[p] / self.epsilon_d)), params[p] * self.epsilon_i, name=p)
                else:
                    np = Integer(abs(params[p] - self.epsilon_i), params[p] + self.epsilon_i, name=p)
                new_Hp.append(np)

        # add hyperparameters to the search space
        self.search_space = self.search_space + new_Hp
        return self.search_space


if __name__ == '__main__':
    ss = search_space()
    sp = ss.search_sp()

    dtest = {'reg': 1e-4}
    res_final = ss.add_params(dtest)
    print(sp)
    print("-----------------------")
    print(res_final)
