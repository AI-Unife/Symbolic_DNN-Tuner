import sys
import re
from problog.program import PrologString
from problog import get_evaluatable
from problog.tasks import sample


class NeuralSymbolicBridge:
    """
    class used to interact with the prolog part, managing the model and
    updating it as needed as a result of the reasoning
    """
    def __init__(self):
        """
        init attributes, defining the list containing the terms that maps the initial facts and problems of the symbolic part
        """
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
        output.write(sym_facts + "\n" + sym_prob + "\n" + sym_model + "\n" + rules)
        output.close()

        # return the assembled model
        return PrologString(sym_facts + "\n" + sym_prob + "\n" + sym_model + "\n" + rules)

    def complete_probs(self, sym_model, prev_model):
        """
        method used to complete each action by adding the body of rules
        :param sym_model prev_model: old and new set of actions used to update various rules
        :return: complete model with the new probabilities
        """
        new_str = ""

        # divide the new set of actions into list of lines
        temp = sym_model.split("\n")

        # save the "eve" atom as the first element of the new model
        res = [temp[0]]

        # iter on each pair of new and old actions
        for a, p in zip(temp[1:], prev_model[1:]):
            # if the eve atom is in the body of the rule,
            # remove it and end the body of the rule with a dot
            if "eve" in a:
                a = a[:-8] + "."

            # find the index to subdivide the head and body of the rule
            # and use it to get the problems of the current action
            prob_st = p.find(":-")
            problem = p[prob_st:]

            # add them to the head with the new probability
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

    def clean_problems(self, problems):
        """
        method used to delete space and dots from problems definition
        :param problems: problems to clean up from certain chars
        :return: problem definitions without the specified chars
        """
        clean_p = r'[.\s]'
        return [re.sub(clean_p, '', p) for p in problems]

    def build_sym_prob(self, problems):
        """
        method used to dynamically add actions from the various modules
        :param problems: rules in which are defined new problems and the actions to use to solve them
        """
        # read the file containing the set of actions
        base_model = open("symbolic/sym_prob_base.pl", "r").read()
        
        # add the new problems from modules
        base_model += problems
 
        # init dict containing the actions as an empty dict and the final rules string as empty string
        rules_dict = {}
        rules = ""

        # iterate over each action
        for problem in base_model.splitlines():
            # filter model based on different types of rules using regular expressions
            # atoms, probilistic and deterministic rules
            # probabilistic rules splitted in prob, action and problems
            prob_rules = re.search(r'(.*)(?<=::)(.*)(?<=:-)(.*)', problem)
            det_rules = re.search(r'^([\D].*)', problem)
            atoms = re.search(r'^((?!:-).)*$', problem)

            # adding atoms and deterministic rules to the final string
            if atoms is not None:
                rules += "".join(atoms.group()) + "\n"

            if det_rules is not None:
                rules += "".join(det_rules.group()) + "\n"

            if prob_rules is not None:
                #get probabilty and action
                base_prob, action = prob_rules.group(1), prob_rules.group(2)

                # use the name of the action as a key
                # each element of the dict has a list of two elements
                # first is the rule's probability, the second a list of possible problems
                if not action in rules_dict:
                    rules_dict[action] = [base_prob, []]
  
                # add to the actions list the possible problems from modules
                new_problems = self.clean_problems(prob_rules.group(3).split(','))
                rules_dict[action][1] += new_problems

        # iterate over every probabilistic action
        for action in rules_dict:
            # init rule with prob and name of the action
            new_rule = rules_dict[action][0] + action
 
            # delete duplicate problems with set
            merged_problem = list(set(rules_dict[action][1]))

            # complete the rule and add it to the model
            for new_p in merged_problem:
                new_rule += " " + new_p +  ","    
            rules += new_rule[:-1] + ".\n"

        f = open("symbolic/sym_prob.pl", "w")
        f.write(rules)
        f.close()

    def edit_probs(self, sym_model):
        """
        method used to update the probabilities of actions in the symbolic part
        :param sym_model: set of actions that need to be updated
        """
        # read the file containing the old set of actions
        prev_model = open("symbolic/sym_prob.pl", "r").read()

        # get the new probabilities of the tuning actions as a result of reasoning,
        # this using a regular expression with the pattern defined as the first parameter
        x = re.findall("[0-9][.].*[:][:]['a']", sym_model)

        # iterate over each actions probabilty
        for i in range(len(x)):
            # proceed using the same regular expression to get the old probabilities
            # and replace them with the 'sub' function of regular expressions
            xx = re.findall("[0-9][.].*[:][:]['a']", prev_model)
            new = re.sub(xx[i], x[i], sym_model)

        # call the method for completing each action with the body of the each rule
        new = self.complete_probs(new, prev_model.split("\n"))

        # updates the file on which the actions are stored
        f = open("symbolic/sym_prob.pl", "w")
        f.write(new)
        f.close()

    def symbolic_reasoning(self, facts, diagnosis_logs, tuning_logs, rules):
        """
        Start symbolic reasoning
        :param facts diagnosis_logs tuning_logs rules: facts and new rules to code into the symbolic program
        :return: result of symbolic reasoning in form of list
        """
        tuning = []
        diagnosis = []
        res = {}
        problems = []

        # create symbolic model, joining the various parts
        symbolic_model = self.build_symbolic_model(facts, rules)

        # based on the model, create a dict that maps each query term to its probability
        symbolic_evaluation = get_evaluatable().create_from(symbolic_model).evaluate()

        # collect all the problem, specifically the second argument of each action, in the problem list
        for i in symbolic_evaluation.keys():
            problems.append(str(i)[str(i).find(",") + 1:str(i).find(")")])

        # turn problems into keys of a dictionary, allowing to remove duplicate problems,
        # and then turn it into a list
        problems = list(dict.fromkeys(problems))

        # iterate on each pair (nested loop) of problems and actions obtained from reasoning
        for i in problems:
            # set the dict used to collect possible solutions in this iteration as an empty dict
            inner = {}
            for j in symbolic_evaluation.keys():
                # if problem "i" is present in action "j"
                if i in str(j):
                    # put into the partial dict "inner" the probability of that action,
                    # using the name of the possible solution to problem "i" as the key.
                    # this allow to collect for a specific problem the possible solutions
                    inner[str(j)[str(j).find("(") + 1:str(j).find(",")]] = symbolic_evaluation[j]

            # put collected solutions into the dictionary using the problem as a key
            res[i] = inner

        # iterate over each problem
        for i in res.keys():
            # if one of them is overfitting
            if i == "overfitting":
                # Set the value of the regularization probability to 0 and
                # add overfitting and regl to the tuning and diagnosis lists
                res[i]["reg_l2"] = 0
                tuning.append("reg_l2")
                diagnosis.append(i)
            diagnosis.append(i)
            # find the solution with maximum probability and add it to the tuning operations
            tuning.append(max(res[i], key=res[i].get))

        # remove duplicates from tuning and diagnosis and then store them on dedicated log files
        to_log_tuning = list(dict.fromkeys(tuning))
        to_log_diagnosis = list(dict.fromkeys(diagnosis))
        diagnosis_logs.write(str(to_log_diagnosis) + "\n")
        tuning_logs.write(str(to_log_tuning) + "\n")
        return tuning, diagnosis
