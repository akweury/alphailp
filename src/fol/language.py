import glob
from lark import Lark

from .exp_parser import ExpTree
from .logic import *
from fol import bk


# from fol import mode_declaration

class Language(object):
    """Language of first-order logic.

    A class of languages in first-order logic.

    Args:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.

    Attrs:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.
    """

    def __init__(self, args, funcs):
        self.vars = [Var(f"O{i + 1}") for i in range(args.e)]
        self.var_num = args.e
        self.atoms = []
        self.preds = []
        self.invented_preds = []
        self.invented_preds_with_scores = []
        self.funcs = funcs
        self.consts = []
        self.clauses = []
        self.pi_clauses = []
        # self.pi_templates = pi_templates

        ## BK
        self.bk_inv_preds = []
        self.all_invented_preds = []
        self.all_pi_clauses = []
        self.invented_preds_number = args.p_inv_counter

        self.base_path = args.lang_base_path / args.dataset_type / args.dataset
        with open(args.lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")
        with open(args.lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")

        self.load_lang(args)
        self.load_init_clauses(args)

    def __str__(self):
        s = "===Predicates===\n"
        for pred in self.preds:
            s += pred.__str__() + '\n'
        s += "===Function Symbols===\n"
        for func in self.funcs:
            s += func.__str__() + '\n'
        s += "===Constants===\n"
        for const in self.consts:
            s += const.__str__() + '\n'
        s += "===Invented Predicates===\n"
        for invented_predicates in self.invented_preds:
            s += invented_predicates.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def generate_atoms(self):
        p_ = Predicate('.', 1, [DataType('spec')])
        false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
        true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])

        spec_atoms = [false, true]
        atoms = []
        for pred in self.preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    atoms.append(Atom(pred, args))
        pi_atoms = []
        for pred in self.invented_preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    pi_atoms.append(Atom(pred, args))
        bk_pi_atoms = []
        for pred in self.bk_inv_preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                # check if args and pred correspond are in the same area
                if pred.dtypes[0].name == 'area':
                    if pred.name[0] + pred.name[5:] != args[0].name:
                        continue
                if len(args) == 1 or len(set(args)) == len(args):
                    pi_atoms.append(Atom(pred, args))
        self.atoms = spec_atoms + sorted(atoms) + sorted(pi_atoms) + sorted(bk_pi_atoms)

    def update_mode_declarations(self, args):
        self.mode_declarations = get_mode_declarations(args, self)

    def load_init_clauses(self, args):
        """Read lines and parse to Atom objects.
        """
        init_clause = "kp(X):-"
        for i in range(args.e):
            init_clause += f"in(O{i + 1},X),"
        init_clause = init_clause[:-1]
        init_clause += "."
        tree = self.lp_clause.parse(init_clause)
        self.clauses = ExpTree(self).transform(tree)
        print("Initial clauses: ", self.clauses)
        self.clauses = [self.clauses]
        return self.clauses

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]

        return Predicate(pred, int(arity), dtypes)

    def parse_const(self, args, const, const_type):
        """Parse string to function symbols.
        """
        const_data_type = DataType(const)
        if "amount_" in const_type:
            _, num = const_type.split('_')
            if num == 'e':
                num = args.e
            const_names = []
            for i in range(int(num)):
                const_names.append(str(const) + str(i))
        elif "enum" in const_type:
            if const == 'color':
                const_names = bk.color
            elif const == 'shape':
                const_names = bk.shape
            # elif const == 'group_shape':
            #     const_names = group_shape
            else:
                raise ValueError
        elif 'target' in const_type:
            const_names = ['image']
        else:
            raise ValueError

        return [Const(const_name, const_data_type) for const_name in const_names]

    def load_consts(self, args):
        consts_str = []
        for const_name, const_type in bk.const_dict.items():
            consts_str.extend(self.parse_const(args, const_name, const_type))
        return consts_str

    def rename_bk_preds_in_clause(self, bk_prefix, line):
        """Parse string to invented predicates.
        """
        new_line = line.replace('\n', '')
        new_line = new_line.replace('inv_pred', "inv_pred_bk" + str(bk_prefix) + "_")
        return new_line

    def parse_invented_bk_clause(self, line, lang):
        """Parse string to invented predicates.
        """

        tree = self.lp_clause.parse(line)
        clause = ExpTree(lang).transform(tree)

        return clause

    def parse_invented_bk_pred(self, bk_prefix, line):
        """Parse string to invented predicates.
        """
        head, body = line.split(':-')
        arity = len(head.split(","))
        head_dtype_names = arity * ['object']
        dtypes = [DataType(dt) for dt in head_dtype_names]

        # pred_with_id = pred + f"_{i}"
        pred_with_id = head.split("(")[0]
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes, args=None, pi_type=None)

        return invented_pred

    def load_invented_preds(self, bk_prefix, path):
        f = open(path)
        lines = f.readlines()
        lines = [self.rename_bk_preds_in_clause(bk_prefix, line) for line in lines]
        preds = [self.parse_invented_bk_pred(bk_prefix, line) for line in lines]
        return preds

    def load_lang(self, args):
        self.preds = [self.parse_pred(line) for line in bk.target_predicate]
        # preds += self.load_neural_preds()
        self.consts = self.load_consts(args)
        # pi_templates = self.load_invented_preds_template(str(self.base_path / 'neural_preds.txt'))

        if args.with_bk:
            bk_pred_files = glob.glob(str(self.base_path / ".." / "bg_predicates" / "*.txt"))
            for bk_i, bk_file in enumerate(bk_pred_files):
                self.bk_inv_preds += self.load_invented_preds(bk_i, bk_file)

    def get_var_and_dtype(self, atom):
        """Get all variables in an input atom with its dtypes by enumerating variables in the input atom.

        Note:
            with the assumption with function free atoms.

        Args:
            atom (Atom): The atom.

        Returns:
            List of tuples (var, dtype)
        """
        var_dtype_list = []
        for i, arg in enumerate(atom.terms):
            if arg.is_var():
                dtype = atom.pred.dtypes[i]
                var_dtype_list.append((arg, dtype))
        return var_dtype_list

    def get_by_dtype(self, dtype):
        """Get constants that match given dtypes.

        Args:
            dtype (DataType): The data type.

        Returns:
            List of constants whose data type is the given data type.
        """
        return [c for c in self.consts if c.dtype == dtype]

    def get_by_dtype_name(self, dtype_name):
        """Get constants that match given dtype name.

        Args:
            dtype_name (str): The name of the data type to be used.

        Returns:
            List of constants whose datatype has the given name.
        """
        return [c for c in self.consts if c.dtype.name == dtype_name]

    def term_index(self, term):
        """Get the index of a term in the language.

        Args:
            term (Term): The term to be used.

        Returns:
            int: The index of the term.
        """
        terms = self.get_by_dtype(term.dtype)
        return terms.index(term)

    def get_const_by_name(self, const_name):
        """Get the constant by its name.

        Args:
            const_name (str): The name of the constant.

        Returns:
            Const: The matched constant with the given name.

        """
        const = [c for c in self.consts if const_name == c.name]
        assert len(const) == 1, 'Too many match in ' + const_name
        return const[0]

    def get_pred_by_name(self, pred_name):
        """Get the predicate by its name.

        Args:
            pred_name (str): The name of the predicate.

        Returns:
            Predicate: The matched preicate with the given name.
        """
        pred = [pred for pred in self.preds if pred.name == pred_name]
        assert len(pred) == 1, 'Too many or less match in ' + pred_name
        return pred[0]

    def get_invented_pred_by_name(self, invented_pred_name):
        """Get the predicate by its name.

        Args:
            invented_pred_name (str): The name of the predicate.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        invented_pred = [invented_pred for invented_pred in self.invented_preds if
                         invented_pred.name == invented_pred_name]
        if not len(invented_pred) == 1:
            raise ValueError('Too many or less match in ' + invented_pred_name)
        return invented_pred[0]

    def get_bk_invented_pred_by_name(self, invented_pred_name):
        """Get the predicate by its name.

        Args:
            invented_pred_name (str): The name of the predicate.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        invented_pred = [invented_pred for invented_pred in self.bk_inv_preds if
                         invented_pred.name == invented_pred_name]
        if not len(invented_pred) > 0:
            raise ValueError('Too less match in ' + invented_pred_name)
        return invented_pred[0]

    def get_new_invented_predicate(self, args, arity, pi_dtypes, p_args, pi_types):
        """Get the predicate by its id.

        Args:
            pi_template (str): The name of the predicate template.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        prefix = "inv_pred"
        new_predicate_id = self.invented_preds_number
        args.p_inv_counter += 1
        self.invented_preds_number = args.p_inv_counter
        pred_with_id = prefix + str(new_predicate_id)
        #
        new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes, p_args, pi_types)
        # self.invented_preds.append(new_predicate)
        return new_predicate

    def update_bk(self, neural_pred=None, full_bk=True):

        # put everything into the bk
        if full_bk:
            if neural_pred is not None:
                self.preds = self.preds[:2] + neural_pred[-1]
            self.invented_preds += self.all_invented_preds
            self.pi_clauses += self.all_pi_clauses
        else:
            # only consider one category by the given nerual pred
            self.preds = self.preds[:2] + neural_pred
            self.invented_preds = []
            self.pi_clauses = []
        self.generate_atoms()
        # self.mode_declaration = mode_declaration.get_mode_declarations(args, self)

        # PM = get_perception_module(args)
        # VM = get_valuation_module(args, lang)
        # PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
        # FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
        #                                     pi_valuation_module=PI_VM, device=args.device)
        # # Neuro-Symbolic Forward Reasoner for clause generation
        # NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, pi_clauses, FC)
        # PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
        #
        # mode_declarations = get_mode_declarations(args, lang)
        # clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
        #                                    no_xil=args.no_xil)  # torch.device('cpu'))

        # pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang,
        #                                         no_xil=args.no_xil)  # torch.device('cpu'))


class DataType(object):
    """Data type in first-order logic.

    A class of data types in first-order logic.

    Args:
        name (str): The name of the data type.

    Attrs:
        name (str): The name of the data type.
    """

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        else:
            return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class ModeDeclaration(object):
    """from https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html
    p(ModeType, ModeType,...)

    Here are some examples of how they appear in a file:

    :- mode(1,mem(+number,+list)).
    :- mode(1,dec(+integer,-integer)).
    :- mode(1,mult(+integer,+integer,-integer)).
    :- mode(1,plus(+integer,+integer,-integer)).
    :- mode(1,(+integer)=(#integer)).
    :- mode(*,has_car(+train,-car)).
    Each ModeType is either (a) simple; or (b) structured.
    A simple ModeType is one of:
    (a) +T specifying that when a literal with predicate symbol p appears in a
    hypothesised clause, the corresponding argument should be an "input" variable of type T;
    (b) -T specifying that the argument is an "output" variable of type T; or
    (c) #T specifying that it should be a constant of type T.
    All the examples above have simple modetypes.
    A structured ModeType is of the form f(..) where f is a function symbol,
    each argument of which is either a simple or structured ModeType.
    Here is an example containing a structured ModeType:


    To make this more clear, here is an example for the mode declarations for
    the grandfather task from
     above::- modeh(1, grandfather(+human, +human)).:-
      modeb(*, parent(-human, +human)).:-
       modeb(*, male(+human)).
       The  first  mode  states  that  the  head  of  the  rule
        (and  therefore  the  target predicate) will be the atom grandfather.
         Its parameters have to be of the type human.
          The  +  annotation  says  that  the  rule  head  needs  two  variables.
            The second mode declaration states the parent atom and declares again
             that the parameters have to be of type human.
              Here,  the + at the second parameter tells, that the system is only allowed to
              introduce the atom parent in the clause if it already contains a variable of type human.
               That the first attribute introduces a new variable into the clause.
    The  modes  consist  of  a  recall n that  states  how  many  versions  of  the
    literal are allowed in a rule and an atom with place-markers that state the literal to-gether
    with annotations on input- and output-variables as well as constants (see[Mug95]).
    Args:
        recall (int): The recall number i.e. how many times the declaration can be instanciated
        pred (Predicate): The predicate.
        mode_terms (ModeTerm): Terms for mode declarations.
    """

    def __init__(self, mode_type, recall, pred, mode_terms, ordered=True):
        self.mode_type = mode_type  # head or body
        self.recall = recall
        self.pred = pred
        self.mode_terms = mode_terms
        self.ordered = ordered

    def __str__(self):
        s = 'mode_' + self.mode_type + '('
        if self.mode_terms is None:
            raise ValueError
        for mt in self.mode_terms:
            s += str(mt)
            s += ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class ModeTerm(object):
    """Terms for mode declarations. It has mode (+, -, #) and data types.
    """

    def __init__(self, mode, dtype):
        self.mode = mode
        assert mode in ['+', '-', '#'], "Invalid mode declaration."
        self.dtype = dtype

    def __str__(self):
        return self.mode + self.dtype.name

    def __repr__(self):
        return self.__str__()


def get_mode_declarations(args, lang):
    obj_num = args.e
    if args.dataset_type == 'kandinsky':
        basic_mode_declarations = get_mode_declarations_kandinsky(lang, obj_num)
        pi_model_declarations = get_pi_mode_declarations(lang, obj_num)
        return basic_mode_declarations + pi_model_declarations
    elif args.dataset_type == "hide":
        basic_mode_declarations = get_mode_declarations_kandinsky(lang, obj_num)
        pi_model_declarations = get_pi_mode_declarations(lang, obj_num)
        return basic_mode_declarations + pi_model_declarations
    else:
        assert False, "Invalid data type."


def get_mode_declarations_kandinsky(lang, obj_num):
    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))

    m_group = ModeTerm('-', DataType('group'))
    p_group = ModeTerm('+', DataType('group'))

    s_color = ModeTerm('#', DataType('color'))
    s_shape = ModeTerm('#', DataType('shape'))

    s_rho = ModeTerm('#', DataType('rho'))
    s_phi = ModeTerm('#', DataType('phi'))
    s_group_shape = ModeTerm('#', DataType('group_shape'))

    modeb_list = []
    considered_pred_names = [p.name for p in lang.preds]
    if "in" in considered_pred_names:
        modeb_list.append(ModeDeclaration('body', obj_num, lang.get_pred_by_name('in'), [m_group, p_image]))
    if "color" in considered_pred_names:
        modeb_list.append(ModeDeclaration('body', obj_num, lang.get_pred_by_name('color'), [p_group, s_color]))
    if "shape" in considered_pred_names:
        modeb_list.append(ModeDeclaration('body', obj_num, lang.get_pred_by_name('shape'), [p_group, s_shape]))
    if "rho" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('rho'), [p_group, p_group, s_rho], ordered=False))
    if "phi" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('phi'), [p_group, p_group, s_phi], ordered=False))
    if "group_shape" in considered_pred_names:
        modeb_list.append(
            ModeDeclaration('body', obj_num, lang.get_pred_by_name('group_shape'), [p_group, s_group_shape],
                            ordered=False))

    return modeb_list


def get_pi_mode_declarations(lang, obj_num):
    p_object = ModeTerm('+', DataType('object'))

    pi_mode_declarations = []
    for pi_index, pi in enumerate(lang.invented_preds):
        pi_str = pi.name
        objects = [p_object] * pi.arity
        mode_declarations = ModeDeclaration('body', obj_num, lang.get_invented_pred_by_name(pi_str), objects,
                                            ordered=False)
        pi_mode_declarations.append(mode_declarations)
    for pi_index, pi in enumerate(lang.bk_inv_preds):
        pi_str = pi.name
        objects = [p_object] * pi.arity
        mode_declarations = ModeDeclaration('body', obj_num, lang.get_bk_invented_pred_by_name(pi_str), objects,
                                            ordered=False)
        pi_mode_declarations.append(mode_declarations)
    return pi_mode_declarations
