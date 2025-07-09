from skopt.space import Integer, Real, Categorical, Space

import numpy as np
import copy

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
        self.search_space = Space([
            Integer(16, 64, name='unit_c1'),
            Real(0.002, 0.3, name='dr1_2'),
            Integer(64, 128, name='unit_c2'),
            Integer(0, 2048, name='unit_d'),
            Real(0.03, 0.5, name='dr_f'),
            Real(1e-4, 1e-3, name='learning_rate'),
            Integer(16, 64, name='batch_size'),
            Categorical(['Adam', 'Adamax'], name='optimizer'), # removed 'Adagrad', 'Adadelta'
            Categorical(['relu', 'elu', 'selu', 'swish'], name='activation'),
        ])

        return self.search_space

    def count_initial_layers(self, params):
        """
        method used to count the number of layers in the search space
        :param params: parameters to be added to the search space
        :return: number of layers in the search space
        """
        # initialize the accumulator of new hyperparameters as an empty list
        tot_conv = 0
        tot_fc = 0

        # iter on all parameters to be added
        for p in params:
            if 'unit_c' in p.name:
                tot_conv += 2
            elif 'unit_d' in p.name:
                tot_fc += 1
        print("tot_conv: ", tot_conv)
        print("tot_fc: ", tot_fc)
        return tot_conv, tot_fc
    
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
                if 'new_fc' in p:
                    np = Integer(abs(int(params[p] / self.epsilon_d)), params[p] * self.epsilon_i, name=p)
                elif 'new_conv' in p:
                    np = Integer(abs(int(params[p] / self.epsilon_d)), params[p] * self.epsilon_i, name=p)
                else:
                    np = Integer(abs(params[p] - self.epsilon_i), params[p] + self.epsilon_i, name=p)
                new_Hp.append(np)

        # add hyperparameters to the search space
        self.search_space = Space(self.search_space.dimensions + new_Hp)
        
        return self.search_space
    
    def remove_params(self, params):
        """
        method used for remove hyperparameters in the search space
        :param params: parameters to be added to the search space
        :return: new search space with added hyperparameters
        """
        # initialize the accumulator of new hyperparameters as an empty list
        new_Hp = []

        # iter on all parameters to be added
        for i, dim in enumerate(self.search_space.dimensions):
            if dim.name not in params.keys(): 
                new_Hp.append(dim)

        # add hyperparameters to the search space
        self.search_space = Space(new_Hp)
        
        return self.search_space
    
    def count_layer(self, type):
        """
        method used to count the number of layers of a certain type
        :param type: type of layer
        :return: number of layers of the specified type
        """
        count = 0
        for hp in self.search_space.dimensions:
            if type in hp.name:
                count += 1
        return count
    
    def get_copy(self, space, constrain_fn=None):
        """
        method used to get a copy of the search space
        :return: copy of the search space
        """
        self.search_space = copy.deepcopy(space.dimensions)
        return self.search_space


if __name__ == '__main__':
    ss = search_space()
    sp = ss.search_sp()

    dtest = {'reg': 1e-4}
    res_final = ss.add_params(dtest)
    print(sp)
    print("-----------------------")
    print(res_final)
