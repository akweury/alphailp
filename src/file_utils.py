import json
import os
from fol.language import DataType
from fol.logic import NeuralPredicate
def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            # if key not in ['conflict_th', 'sc_th','nc_th']:  # Do not overwrite these keys
            setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))
    return None


def load_neural_preds(path):
    f = open(path)
    lines = f.readlines()
    preds = [[parse_neural_pred(line)] for line in lines]
    return preds


def parse_neural_pred(line):
    """Parse string to predicates.
    """
    line = line.replace('\n', '')
    pred, arity, dtype_names_str = line.split(':')
    dtype_names = dtype_names_str.split(',')
    dtypes = [DataType(dt) for dt in dtype_names]
    assert int(arity) == len(
        dtypes), 'Invalid arity and dtypes in ' + pred + '.'
    return NeuralPredicate(pred, int(arity), dtypes)