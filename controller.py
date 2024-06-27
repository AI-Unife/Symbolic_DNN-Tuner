import os
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from colors import colors
from diagnosis import diagnosis
from neural_network import neural_network
from search_space import search_space
from tuning_rules import tuning_rules
from tuning_rules_symbolic import tuning_rules_symbolic
from neural_sym_bridge import NeuralSymbolicBridge
from lfi_integration import LfiIntegration
from storing_experience import StoringExperience
from improvement_checker import ImprovementChecker
from integral import integrals
from shutil import copyfile

from module import module

class controller:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        # self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_classes = n_classes
        self.ss = search_space()
        self.space = self.ss.search_sp()
        self.tr = tuning_rules_symbolic(self.space, self.ss, self)
        self.nsb = NeuralSymbolicBridge() # problems e initial_facts
        self.db = StoringExperience()
        self.db.create_db()
        self.lfi = LfiIntegration(self.db)
        self.symbolic_tuning = []
        self.symbolic_diagnosis = []
        self.issues = []
        self.weight = 0.6
        #self.epsilon = 0.33
        self.new = None
        self.new_fc = None
        self.new_conv = None
        self.rem_conv = None
        self.da = None
        self.model = None
        self.params = None
        self.iter = 0
        self.lacc = 0.15
        self.hloss = 1.2
        self.levels = [7, 10, 13]
        self.imp_checker = ImprovementChecker(self.db, self.lfi)
        self.modules = module(["new_flop_calculator", "accuracy_module"])

    def set_case(self, new):
        self.new = new

    def add_fc_layer(self, new_fc, c):
        self.new_fc = [new_fc, c]

    def add_conv_section(self, new_conv, c):
        self.new_conv = [new_conv, c]
        
    def remove_conv_section(self, rem_conv):
        self.rem_conv = rem_conv

    def set_data_augmentation(self, da):
        self.da = da

    def smooth(self, scalars):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            # Calculate smoothed value
            smoothed_val = last * self.weight + (1 - self.weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def training(self, params):
        """
        Training and tasting the neural network
        training(Train, Labels_train, Test, Label_test)
        :return: model and training history self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        """
        self.params = params
        print(colors.OKBLUE, "|  --> START TRAINING\n", colors.ENDC)
        K.clear_session()
        self.nn = neural_network(self.X_train, self.Y_train, self.X_test, self.Y_test, self.n_classes)
        self.score, self.history, self.model = self.nn.training(params, self.new, self.new_fc, self.new_conv, self.rem_conv, self.da,
                                                                self.space)

        # update state of modules
        # each module will take the necessary args internally
        self.modules.state(self.score[0], self.score[1], self.model)
 
        # print modules informations
        self.modules.print()
        
        # log module values of training
        self.modules.log()

        self.rem_conv = False
        self.iter += 1

        # if no module has been loaded, accuracy will be the value to be optimised
        # otherwise return the finale function value to be optimised
        if len(self.modules.modules_obj) == 0:
            return -self.score[1]
        else:
            _, _, opt_value = self.modules.optimiziation()
            return opt_value

    def diagnosis(self):
        """
        method for diagnose possible issue like overfitting
        :return: call to tuning method or hp_space, model and accuracy(*-1)
        """
        print(colors.CYAN, "| START SYMBOLIC DIAGNOSIS ----------------------------------  |\n", colors.ENDC)
        diagnosis_logs = open("algorithm_logs/diagnosis_symbolic_logs.txt", "a")
        tuning_logs = open("algorithm_logs/tuning_symbolic_logs.txt", "a")

        improv = self.imp_checker.checker(self.score[1], self.score[0])
        self.db.insert_ranking(self.score[1], self.score[0])

        # get rules and problems to create prolog model
        rules, actions, problems = self.modules.get_rules()

        # add facts and problems to NeuralSymbolicBridge
        # and create dynamic prolog file contains list of possible problems
        # only during first diagnosis iteration
        if self.iter == 1:
            for i in range(len(self.modules.modules_obj)):
                self.nsb.initial_facts += self.modules.modules_obj[i].facts
                self.nsb.problems += self.modules.modules_obj[i].problems

            self.nsb.build_sym_prob(problems)

        if improv is not None:
            _, lfi_problem = self.lfi.learning(improv, self.symbolic_tuning, self.symbolic_diagnosis, actions)
            sy_model = lfi_problem.get_model()
            self.nsb.edit_probs(sy_model)

        int_loss, int_slope = integrals(self.history['val_loss'])

        for level in self.levels:
            if self.iter == level:
                self.lacc = self.lacc/2 + 0.05
                self.hloss = self.hloss/2 + 0.15

        # base facts list
        facts_list_module = [self.history['loss'], self.smooth(self.history['loss']),
             self.history['accuracy'], self.smooth(self.history['accuracy']),
             self.history['val_loss'], self.history['val_accuracy'], int_loss, int_slope, self.lacc, self.hloss]

        # add facts list from loaded modules
        facts_list_module += self.modules.values()

        self.symbolic_tuning, self.symbolic_diagnosis = self.nsb.symbolic_reasoning(
            facts_list_module, diagnosis_logs, tuning_logs, rules)

        diagnosis_logs.close()
        tuning_logs.close()

        print(colors.CYAN, "| END SYMBOLIC DIAGNOSIS   ----------------------------------  |\n", colors.ENDC)

        if self.symbolic_tuning:
            self.space, to_optimize, self.model = self.tuning()
            return self.space, to_optimize
        else:
            return self.space, -self.score[1]

    def tuning(self):
        """
        tuning the hyper-parameter space or add new hyper-parameters
        :return: new hp_space, new_model and accuracy(*-1) for the Bayesian Optimization
        """
        print(colors.FAIL, "| START SYMBOLIC TUNING    ----------------------------------  |\n", colors.ENDC)
        # tuning_logs = open("algorithm_logs/tuning_logs.txt", "a")
        # new_space, self.model = self.tr.repair(self, self.symbolic_tuning, tuning_logs, self.model, self.params)
        new_space, self.model = self.tr.repair(self.symbolic_tuning, self.model, self.params)
        # tuning_logs.close()
        self.issues = []
        print(colors.FAIL, "| END SYMBOLIC TUNING      ----------------------------------  |\n", colors.ENDC)

        return new_space, -self.score[1], self.model

    def plotting_obj_function(self):
        # plot graphs from each module
        self.modules.plot()

    def save_experience(self):
        db_split = os.path.splitext(self.db.db_name)
        db_dir = (db_split[0] + "-{}" + db_split[1]).format(self.nn.last_model_id)
        try:
            copyfile(self.db.db_name, db_dir)
        except:
            print(colors.FAIL, "|  -------------- FAILED TO SAVE DB -------------  |\n", colors.ENDC)
