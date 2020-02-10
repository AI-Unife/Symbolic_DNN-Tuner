from tensorflow.keras import backend as K

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


class controller:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        # self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_classes = n_classes
        # self.d = diagnosis()
        self.ss = search_space()
        self.space = self.ss.search_sp()
        self.tr = tuning_rules_symbolic(self.space, self.ss)
        self.nsb = NeuralSymbolicBridge()
        self.lfi = LfiIntegration()
        self.symbolic_tuning = []
        self.symbolic_diagnosis = []
        self.issues = []
        self.weight = 0.6
        self.new = None
        self.model = None
        self.params = None
        self.db = StoringExperience()
        self.db.create_db()
        self.imp_checker = ImprovementChecker(self.db, self.lfi)

    def set_case(self, new):
        self.new = new

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
        :return: model and training historyself.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        """
        self.params = params
        print(colors.OKBLUE, "|  --> START TRAINING\n", colors.ENDC)
        K.clear_session()
        self.nn = neural_network(self.X_train, self.Y_train, self.X_test, self.Y_test, self.n_classes)
        self.score, self.history, self.model = self.nn.training(params, self.new, self.model)

        return -self.score[1]

    def diagnosis(self):
        """
        method for diagnose possible issue like overfitting
        :return: call to tuning method or hp_space, model and accuracy(*-1)
        """
        print(colors.CYAN, "| START SYMBOLIC DIAGNOSIS ----------------------------------  |\n", colors.ENDC)
        diagnosis_logs = open("algorithm_logs/diagnosis_symbolic_logs.txt", "a")
        tuning_logs = open("algorithm_logs/tuning_symbolic_logs.txt", "a")
        # self.d.reset_diagnosis()
        # self.issues = self.d.diagnosis(self.history, self.score, diagnosis_logs, "controller")

        print(colors.CYAN, "| END SYMBOLIC DIAGNOSIS   ----------------------------------  |\n", colors.ENDC)

        self.db.insert(self.score[1], self.score[0])

        self.symbolic_tuning, self.symbolic_diagnosis = self.nsb.symbolic_reasoning(
            [self.history['loss'], self.smooth(self.history['loss']),
             self.smooth(self.history['accuracy']),
             self.history['accuracy'],
             self.history['val_loss'], self.history['val_accuracy']],
            diagnosis_logs, tuning_logs)

        improv = self.imp_checker.checker(self.score[1], self.score[0])
        _, lfi_problem = self.lfi.learning(improv, self.symbolic_tuning, self.symbolic_diagnosis)
        sy_model = lfi_problem.get_model()
        self.nsb.edit_probs(sy_model)

        diagnosis_logs.close()
        tuning_logs.close()

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
        new_space, self.model = self.tr.repair(self, self.symbolic_tuning, self.model, self.params)
        # tuning_logs.close()
        self.issues = []
        print(colors.FAIL, "| END SYMBOLIC TUNING      ----------------------------------  |\n", colors.ENDC)

        return new_space, -self.score[1], self.model
