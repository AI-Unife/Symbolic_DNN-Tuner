import sys
import re
from problog.program import PrologString
from problog import get_evaluatable
from problog.tasks import sample


class NeuralSymbolicBridge:
    def __init__(self):
        #self.initial_facts = ['l', 'sl', 'a', 'sa', 'vl', 'va',
        #                      'int_loss', 'int_slope', 'lacc', 'hloss', 'flops', 'flops_th', 'nparams', 'nparams_th']
        #self.problems = ['overfitting', 'underfitting', 'inc_loss', 'floating_loss', 'high_lr', 'low_lr', 'latency', 'model_size']

        self.initial_facts = ['l', 'sl', 'a', 'sa', 'vl', 'va',
                              'int_loss', 'int_slope', 'lacc', 'hloss']
        self.problems = ['overfitting', 'underfitting', 'inc_loss', 'floating_loss', 'high_lr', 'low_lr']

    def build_symbolic_model(self, facts, rules):
        """
        build logic program
        :param facts: facts to code dynamically into the symbolic program
        :return: logic program
        """
        # reading model from file
        f = open("symbolic/symbolic_analysis.pl", "r")
        sym_model = f.read()
        f.close()

        p = open("symbolic/sym_prob.pl", "r")
        sym_prob = p.read()
        p.close()

        # create facts string for complete the symbolic model
        sym_facts = ""
        for fa, i in zip(facts, self.initial_facts):
            sym_facts = sym_facts + i + "(" + str(fa) + ").\n"

        output = open("symbolic/final.pl", "w")
        #output.write(sym_facts + "\n" + sym_prob + "\n" + sym_model)
        output.write(sym_facts + "\n" + sym_prob + "\n" + sym_model + "\n" + rules)
        output.close()

        # return the assembled model
        #return PrologString(sym_facts + "\n" + sym_prob + "\n" + sym_model)
        return PrologString(sym_facts + "\n" + sym_prob + "\n" + sym_model + "\n" + rules)

    def complete_probs(self, sym_model, prev_model):
        new_str = ""
        temp = sym_model.split("\n")
        res = [temp[0]]
        for a, p in zip(temp[1:], prev_model[1:]):
            if "eve" in a:
                a = a[:-8] + "."
            prob_st = p.find(":-")
            problem = p[prob_st:]
            new = a[:-1] + problem
            res.append(new)
        # for t in temp[1:]:
        #     cprob = 0
        #     for p in self.problems:
        #         if p in t:
        #             where = t.find(p)
        #             cprob += 1
        #             if "eve" in t:
        #                 new_str = t[:len(t) - 1] + ", problem(" + p + "), "
        #                 # res.append(t[:len(t) - 1] + ", problem(" + p + ").")
        #                 continue
        #             else:
        #                 if cprob == 1:
        #                     new_str = t[:len(t) - 1] + "problem(" + p + "), "
        #                 else:
        #                     if t[where-2:where] == 'n_':
        #                         new_str = new_str + ":- \+problem(" + p + "), "
        #                     else:
        #                         new_str = new_str + ":- problem(" + p + "), "
        #     res.append(new_str[:len(new_str)-2] + ".")
        return "\n".join(res)

    def build_sym_prob(self, problems):
        base_model = open("symbolic/sym_prob_base.pl", "r").read()
        base_model += problems

        f = open("symbolic/sym_prob.pl", "w")
        f.write(base_model)
        f.close()

    def edit_probs(self, sym_model):
        prev_model = open("symbolic/sym_prob.pl", "r").read()

        x = re.findall("[0-9][.].*[:][:]['a']", sym_model)
        for i in range(len(x)):
            xx = re.findall("[0-9][.].*[:][:]['a']", prev_model)
            new = re.sub(xx[i], x[i], sym_model)
        new = self.complete_probs(new, prev_model.split("\n"))
        f = open("symbolic/sym_prob.pl", "w")
        f.write(new)
        f.close()

    def symbolic_reasoning(self, facts, diagnosis_logs, tuning_logs, rules):
        """
        Start symbolic reasoning
        :param facts: facts to code into the symbolic program
        :return: result of symbolic reasoning in form of list
        """
        tuning = []
        diagnosis = []
        res = {}
        problems = []
        symbolic_model = self.build_symbolic_model(facts, rules)
        symbolic_evaluation = get_evaluatable().create_from(symbolic_model).evaluate()

        for i in symbolic_evaluation.keys():
            problems.append(str(i)[str(i).find(",") + 1:str(i).find(")")])

        problems = list(dict.fromkeys(problems))

        for i in problems:
            inner = {}
            for j in symbolic_evaluation.keys():
                if i in str(j):
                    inner[str(j)[str(j).find("(") + 1:str(j).find(",")]] = symbolic_evaluation[j]
            res[i] = inner

        for i in res.keys():
            if i == "overfitting":
                res[i]["reg_l2"] = 0
                tuning.append("reg_l2")
                diagnosis.append(i)
            diagnosis.append(i)
            tuning.append(max(res[i], key=res[i].get))

        to_log_tuning = list(dict.fromkeys(tuning))
        to_log_diagnosis = list(dict.fromkeys(diagnosis))
        diagnosis_logs.write(str(to_log_diagnosis) + "\n")
        tuning_logs.write(str(to_log_tuning) + "\n")
        return tuning, diagnosis
