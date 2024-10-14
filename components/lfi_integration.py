from problog.program import PrologString
from problog.logic import Term
from problog.learning import lfi


class LfiIntegration:
    """
    class used to generate evidence during training, from which to learn and model the probability
    with which to apply certain action to fix problems from which the network is affected
    """
    def __init__(self, db):
        """
        initialise the attributes in which to save the evidence
        and define the actions that will populate the symbolic part
        """
        self.db = db
        self.experience = []
        self.ove = Term('ove')
        self.und = Term('und')
        self.flo = Term('flo')
        self.reg_l2 = Term('reg_l2')
        self.inc_dropout = Term('inc_dropout')
        self.decr_lr = Term('decr_lr')
        self.inc_neurons = Term('inc_neurons')
        self.inc_batch_size = Term('inc_batch_size')
        self.data_augmentation = Term('data_augmentation')
        self.overfitting = Term('overfitting')
        self.underfitting = Term('underfitting')
        self.inc_loss = Term('inc_loss')
        self.floating_loss = Term('floating_loss')
        self.high_lr = Term('high_lr')
        self.low_lr = Term('low_lr')
        self.inc_lr = Term('inc_lr')
        self.action1 = Term('action', self.reg_l2, self.overfitting)
        self.action2 = Term('action', self.inc_dropout, self.overfitting)
        self.action3 = Term('action', self.decr_lr, self.underfitting)
        self.action4 = Term('action', self.inc_neurons, self.underfitting)
        self.action5 = Term('action', self.decr_lr, self.inc_loss, self.inc_loss)
        self.action6 = Term('action', self.inc_batch_size, self.floating_loss)
        self.action7 = Term('action', self.decr_lr, self.floating_loss)
        self.action8 = Term('action', self.data_augmentation, self.overfitting)
        self.action9 = Term('action', self.inc_lr, self.low_lr)
        self.action10 = Term('action', self.decr_lr, self.high_lr)

    def get_str(self, s, before, after):
        return (i.split(after)[0] for i in s.split(before)[1:] if after in i)

    def create_evidence(self, t, d, bool):
        """
        method for creating an evidence
        :param t d bool: tuning rule 't' with associated diagnosis 'd', with a performance improvement indicated by 'bool'
        :return: evidence created according to the tuning rule
        """
        # once the names of the tuning rule and its diagnosis have been converted into terms,
        # proceeds by creating the final term as the triple ('action' tuning diagnosis)
        t1 = Term(str(t))
        t2 = Term(str(d))
        prob = Term(d[:3])
        action = Term('action', t1, t2)
        evidence1 = (action, bool)
        evidence2 = (prob, bool)
        return evidence1, evidence2

    def evidence(self, improve, tuning, diagnosis):
        """
        method for generating evidence, inserting them in db and in a dedicated log file
        :param improve tuning diagnosis: list of tuning rules and diagnosis, with 'improve' bool to indicate if their application led to an improvement
        :return: list of evidence
        """
        # initialise the evidence list as an empty list
        evidence = []
        
        # for each pair 'anomaly' and possible resolution
        # create the corresponding evidence and add it to the initial list
        for t, d in zip(tuning, diagnosis):
            if improve:
                e1, e2 = self.create_evidence(t, d, improve)
            else:
                e1, e2 = self.create_evidence(t, d, improve)
            evidence.append(e1)
            # evidence.append(e2)

        # enters the evidence in the corresponding log file and db
        e = open("algorithm_logs/evidence.txt", "a")
        e.write(str(evidence))
        self.db.insert_evidence(evidence[0])
        e.close()
        return evidence

    def learning(self, improve, tuning, diagnosis, actions):
        """
        method used to learn, based on the past evidence, the probability with which to apply the tuning rules
        :param improve tuning diagnosi actions: used to generate evidence, dynamically create the file from which to learn
        :return: probability list, one for each action and model containing the actions
        """
        # generate evidence and add them to the main list
        evidence = self.evidence(improve, tuning, diagnosis)
        self.experience.append(evidence)

        # read the set of actions from the prolog file
        f1 = open("symbolic/lfi.pl", "r")
        to_learn = f1.read()
        f1.close()

        # dynamically add actions from modules loaded by the controller
        to_learn += actions

        # get probabilities of each action and the model on which the learning was performed
        _, weights, _, _, lfi_problem = lfi.run_lfi(PrologString(to_learn), self.experience)

        return weights, lfi_problem
