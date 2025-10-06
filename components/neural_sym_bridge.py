import sys
import re
from problog.program import PrologString
from problog import get_evaluatable
from problog.tasks import sample

import config as cfg
from tensorflow.python.keras.utils.np_utils import normalize


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
        f = open("{}/symbolic/symbolic_analysis.pl".format(cfg.NAME_EXP), "r")
        sym_model = f.read()
        f.close()

        p = open("{}/symbolic/sym_prob.pl".format(cfg.NAME_EXP), "r")
        sym_prob = p.read()
        p.close()

        # create facts string for complete the symbolic model
        sym_facts = ""
        for fa, i in zip(facts, self.initial_facts):
            sym_facts = sym_facts + i + "(" + str(fa) + ").\n"
        # print("Symbolic facts: ", sym_facts, "\nSymbolic problems: ", sym_prob, "\nSymbolic model: ", sym_model, "\nrules: ", rules)
        output = open("{}/symbolic/final.pl".format(cfg.NAME_EXP), "w")
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
        fact_re = re.compile(r'^\s*([0-9.]+)\s*::\s*([a-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*\.\s*$')
        rule_re = re.compile(r'^\s*([0-9.]+)\s*::\s*([a-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*:-\s*(.+?)\s*\.\s*$')

        # mappa head -> nuova probabilità da progA
        prob_map = {}
        for line in sym_model.splitlines():
            m = fact_re.match(line)
            if m:
                p, head = m.group(1), m.group(2)
                prob_map[head] = p

        out_lines = []
        for line in prev_model.splitlines():
            m = rule_re.match(line)
            if m:
                _, head, body = m.groups()
                p = prob_map.get(head)
                if p is not None:
                    out_lines.append(f"{p}::{head} :- {body}.")
                    continue
            out_lines.append(line)  # se non c'è match o non c'è prob sostitutiva

        return "\n".join(out_lines)

    def clean_problems(self, problems):
        """
        method used to delete space and dots from problems definition
        :param problems: problems to clean up from certain chars
        :return: problem definitions without the specified chars
        """
        clean_p = r'[.\s]'
        return [re.sub(clean_p, '', p) for p in problems]

    import re

    def build_sym_prob(self, problems):
        """
        Unisce TUTTE le regole probabilistiche che hanno la stessa testa in UNA SOLA regola:
        - congiunge i corpi con AND (',')
        - imposta SEMPRE la probabilità a 0.5 (ignora le probabilità originali)
        - preserva deterministic rules/atomi così come sono
        """
        # Carica base + aggiunte
        with open(f"{cfg.NAME_EXP}/symbolic/sym_prob_base.pl", "r") as f:
            base_model = f.read()
        full_model = base_model + ("\n" if not base_model.endswith("\n") else "") + problems

        # Accumulatori
        # Chiave: head (inclusi argomenti e arità) -> insieme di corpi (stringhe normalizzate)
        heads_to_bodies = {}
        other_lines = []
        seen_other = set()

        # Regex
        prob_with_body_re = re.compile(
            r'^\s*([0-9]*\.?[0-9]+)\s*::\s*([a-z][A-Za-z0-9_]*(?:\([^()]*\))?)\s*:-\s*(.+?)\s*\.\s*$'
        )
        comment_or_empty_re = re.compile(r'^\s*(%.*)?$')

        def normalize_text(s: str) -> str:
            s = re.sub(r'\s+', ' ', s.strip())
            s = re.sub(r'\s*,\s*', ', ', s)
            s = re.sub(r'\s*;\s*', ' ; ', s)  # nel caso ci siano OR espliciti
            return s

        def parenthesize(body: str) -> str:
            """Per sicurezza, racchiude ogni corpo tra parentesi quando lo congiungiamo con AND."""
            if body == 'true':
                return body
            return f'({body})'

        # Parse
        for raw_line in full_model.splitlines():
            line = raw_line.rstrip()

            if comment_or_empty_re.match(line):
                if line not in seen_other:
                    other_lines.append(line)
                    seen_other.add(line)
                continue

            m = prob_with_body_re.match(line)
            if m:
                prob = float(m.group(1))
                head = normalize_text(m.group(2))
                body = normalize_text(m.group(3))
                if head not in heads_to_bodies:
                    heads_to_bodies[head] = {'prob': prob, 'body': body}
                else:
                    heads_to_bodies[head]['prob'] = (heads_to_bodies[head]['prob'] + prob)/2
                    tot_body = set((heads_to_bodies[head]['body'] +', ' + body).split(', '))
                    heads_to_bodies[head]['body'] = ', '.join(b for b in tot_body)
                continue

            # deterministic/atomi
            if line not in seen_other:
                other_lines.append(line)
                seen_other.add(line)

        # Ricostruzione
        out_lines = []
        # out_lines.extend(other_lines)

        # Per ogni testa, congiungi i corpi e imposta 0.5
        for idx, item in heads_to_bodies.items():
            out_lines.append(f"{item['prob']}::{idx} :- {item['body']}.")

        final_text = "\n".join(out_lines).rstrip() + "\n"
        with open(f"{cfg.NAME_EXP}/symbolic/sym_prob.pl", "w") as f:
            f.write(final_text)

    def edit_probs(self, sym_model):
        """
        method used to update the probabilities of actions in the symbolic part
        :param sym_model: set of actions that need to be updated
        """
        # read the file containing the old set of actions
        prev_model = open("{}/symbolic/sym_prob.pl".format(cfg.NAME_EXP), "r").read()

        # call the method for completing each action with the body of the each rule
        new = self.complete_probs(sym_model, prev_model)
        # updates the file on which the actions are stored
        f = open("{}/symbolic/sym_prob.pl".format(cfg.NAME_EXP), "w")
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
