import random as ra
import sys
from time import time


class tuning_rules_symbolic:
    """
    class to call the tuning methods in order to solve problems in the architecture of the neural network
    """
    def __init__(self, params, ss, controller):
        """
        initialisation of attributes to determine the parameters of the network tuning methods,
        for example if it's necessary to add data augmentation 
        """
        self.space = params
        self.ss = ss
        self.controller = controller
        self.count_lr = 0
        self.count_da = 0
        self.count_br = 0
        self.count_new_fc = 0
        self.count_new_cv = 0

    def reg_l2(self):
        """
        method used to add regularization and batch normalization in the neural network
        """
        self.count_br += 1
        if self.count_br <= 1:
            print("I've try to fix OVERFITTING by adding regularization and batch normalization\n")
            model = 'batch'
            self.controller.set_case(True)
            new_p = {'reg': 1e-4}
            self.space = self.ss.add_params(new_p)

    def new_fc_layer(self):
        """
        method used to add a dense layer
        """
        self.count_new_fc += 1
        self.controller.add_fc_layer(True, self.count_new_fc)
        new_p = {'new_fc': 512}
        self.space = self.ss.add_params(new_p)

    def new_conv_layer(self):
        """
        method used to add a convolutional layer
        """
        self.count_new_cv += 1
        self.controller.add_conv_section(True, self.count_new_cv)

    def data_augmentation(self):
        """
        method used to add data augmentation
        """
        self.controller.set_data_augmentation(True)

    def inc_dropout(self, params):
        """
        method used to increment dropout
        """
        self.controller.set_data_augmentation(False)

        # iterate over hyperparameters space
        for hp in self.space:
            # check if dropout is present in the search space and in that case
            # proceed by increasing the lower range
            if 'dr' in hp.name:
                hp.low = params[hp.name] - params[hp.name] / 100

    def decr_lr(self, params):
        """
        method used to decrement learning rate
        """
        for hp in self.space:
            # check if learning rate is present in the search space and in that case
            # proceed by increasing the upper range adding half of its value
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + (params['learning_rate'] / 2)

    def inc_lr(self, params):
        """
        method used to increment learning rate
        """
        for hp in self.space:
            # check if learning rate is present in the search space and in that case
            # proceed by incresing the upper range with the current learning rate value
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + hp.high

    def inc_neurons(self, params):
        """
        method used to increment the number of neurons
        """
        # itereate over each hyperparameter and if one of these is a convolutional
        # or dense layer decrease the lower value of the range
        for hp in self.space:
            if 'unit_c1' in hp.name:
                hp.low = params['unit_c1'] - 1
            if 'unit_c2' in hp.name:
                hp.low = params['unit_c2'] - 1
            if 'unit_d' in hp.name:
                hp.low = params['unit_d'] - 1
    
    def inc_batch_size(self, params):
        for hp in self.space:
            if hp.name == 'batch_size':
                hp.low = params['batch_size'] - 1
                
    # new action for hardware constraints
    def dec_neurons(self, params):
        """
        method used to decrement the number of neurons
        """
        # itereate over each hyperparameter and if one of these is a convolutional
        # or dense layer increase the upper value of the range
        for hp in self.space:
            if 'unit_c1' in hp.name:
                hp.high = params['unit_c1'] + 1
            if 'unit_c2' in hp.name:
                hp.high = params['unit_c2'] + 1
            if 'unit_d' in hp.name:
                hp.high = params['unit_d'] + 1

    def dec_layers(self):
        """
        method used to remove a convolutional layer from the neural network
        """
        self.controller.remove_conv_section(True)

    def dec_fc(self):
        """
        method used to remove a dense layer from the neural network
        """
        self.controller.remove_fully_connected(True)

    def new_config(self):
        """
        method used to manage the hardware configurations, trying to find
        the one that can contain most efficiently the neural network
        """
        self.controller.manage_configuration()
  
    # ------------------------------------

    def repair(self, sym_tuning, model, params):
        """
        Method for fix the issues
        :return: new hp_space and new model
        """
        # delete old model saved in controller class
        del self.controller.model

        # iterate over each tuning rules and eveluate them, passing parameters if necessary
        for d in sym_tuning:
            if d != 'reg_l2' and d != 'data_augmentation' and d != 'new_fc_layer' and d != 'new_conv_layer' and d != 'dec_layers' and d != 'dec_fc' and d != 'new_config':
                d = "self." + d + "(params)"
            else:
                d = "self." + d + "()"
            eval(d)

        return self.space, model
