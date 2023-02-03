import os.path

import lark.exceptions
from lark import Lark
from .exp_parser import ExpTree
from .language import Language, DataType
from .logic import Predicate, NeuralPredicate, InventedPredicate, FuncSymbol, Const


class DataUtils(object):
    """Utilities about logic.

    A class of utilities about first-order logic.

    Args:
        dataset_type (str): A dataset type (kandinsky or clevr).
        dataset (str): A dataset to be used.

    Attrs:
        base_path: The base path of the dataset.
    """

    def __init__(self, lark_path, lang_base_path, dataset_type='kandinsky', dataset='twopairs'):
        self.base_path = lang_base_path / dataset_type / dataset
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

    def load_clauses(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]
                    tree = self.lp_clause.parse(line)
                    clause = ExpTree(lang).transform(tree)
                    clauses.append(clause)
        return clauses

    def load_pi_clauses(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]

                    # substitude placeholder predicates to exist predicates

                    clause_candidates = self.get_clause_candidates(lang, line)
                    for clause_str in clause_candidates:
                        tree = self.lp_clause.parse(clause_str)
                        clause = ExpTree(lang).transform(tree)
                        clauses.append(clause)

        return clauses

    def gen_pi_clauses(self, path, lang, clause_str_list):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        for clause_str in clause_str_list:
            tree = self.lp_clause.parse(clause_str)
            clause = ExpTree(lang).transform(tree)
            clauses.append(clause)

        return clauses

    def get_clause_candidates(self, lang, clause_template):
        """

        Args:
            lang: language
            clause_template: a predicate invention template

        Returns:
            all the possible clauses, which satisfy the template by replacing the template_predicates to exist predicates
            in the language.
        """

        [head, body] = clause_template.split(":-")
        body_predicates = body.split(";")

        body_candidates = []
        for body_predicate in body_predicates:
            predicate_candidates = []
            pred_arity = len(body_predicate.split(","))
            arguments = body_predicate.split("(")[1].split(")")[0]

            for p in lang.preds:
                if p.arity == pred_arity:
                    predicate_candidates.append(p.name + "(" + arguments + ")")
            body_candidates.append(predicate_candidates)

        new_clauses = []
        for invented_preds in lang.invented_preds:
            clause_head = invented_preds.name
            arity = invented_preds.arity
            arguments = "(X,Y)" if arity == 2 else "(X)"
            clause_head += arguments

            new_clauses += [clause_head + ":-" + i + "." for i in body_candidates[0]]
        return new_clauses

    def load_atoms(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        atoms = []

        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-2]
                    else:
                        line = line[:-1]
                    tree = self.lp_atom.parse(line)
                    atom = ExpTree(lang).transform(tree)
                    atoms.append(atom)
        return atoms

    def load_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_pred(line) for line in lines]
        return preds

    def load_neural_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_neural_pred(line) for line in lines]
        return preds

    def load_invented_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = []
        for line in lines:
            new_preds = self.parse_invented_pred(line)
            if new_preds is not None:
                preds += new_preds
        return preds

    def load_consts(self, path):
        f = open(path)
        lines = f.readlines()
        consts = []
        for line in lines:
            consts.extend(self.parse_const(line))
        return consts

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return Predicate(pred, int(arity), dtypes)

    def parse_neural_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return NeuralPredicate(pred, int(arity), dtypes)

    def parse_invented_pred(self, line):
        """Parse string to invented predicates.
        """
        line = line.replace('\n', '')
        if (len(line)) == 0:
            return None

        pred, arity, dtype_names_str, pred_num = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(dtypes), 'Invalid arity and dtypes in ' + pred + '.'

        invented_predicates = []
        for i in range(int(pred_num)):
            # pred_with_id = pred + f"_{i}"
            pred_with_id = pred
            invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes)
            invented_predicates.append(invented_pred)

        return invented_predicates

    def parse_funcs(self, line):
        """Parse string to function symbols.
        """
        funcs = []
        for func_arity in line.split(','):
            func, arity = func_arity.split(':')
            funcs.append(FuncSymbol(func, int(arity)))
        return funcs

    def parse_const(self, line):
        """Parse string to function symbols.
        """
        line = line.replace('\n', '')
        dtype_name, const_names_str = line.split(':')
        dtype = DataType(dtype_name)
        const_names = const_names_str.split(',')
        return [Const(const_name, dtype) for const_name in const_names]

    def parse_clause(self, clause_str, lang):
        tree = self.lp_clause.parse(clause_str)
        return ExpTree(lang).transform(tree)

    def get_clauses(self, lang):
        return self.load_clauses(str(self.base_path / 'clauses.txt'), lang)

    def get_bk(self, lang):
        return self.load_atoms(str(self.base_path / 'bk.txt'), lang)

    def load_language(self):
        """Load language, background knowledge, and clauses from files.
        """
        preds = self.load_preds(str(self.base_path / 'preds.txt'))
        preds += self.load_neural_preds(str(self.base_path / 'neural_preds.txt'))
        invented_preds = self.load_invented_preds(str(self.base_path / 'invent_preds.txt'))
        preds += invented_preds
        consts = self.load_consts(str(self.base_path / 'consts.txt'))
        lang = Language(preds, [], consts, invented_preds)
        return lang
