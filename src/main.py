# Created by shaji on 21-Mar-23

import time
import datetime

import mechanics
import log_utils
import pi
from pi import final_evaluation
from mechanic_utils import update_args

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def main():
    # set up the environment, load the dataset and results from perception models
    args, rtpt, percept_dict = mechanics.init()

    # ILP and PI system
    start = time.time()
    # grouping objects to reduce the problem complexity
    obj_groups = mechanics.detect_obj_groups_with_bk(args, percept_dict["val_pos"], percept_dict["val_neg"])
    # update arguments
    update_args(args, percept_dict, obj_groups)
    # describe the scenes with clauses, invent new predicates if necessary
    NSFR = pi.train_and_eval(args, percept_dict, obj_groups, rtpt)
    end = time.time()

    log_utils.add_lines(f"=============================", args.log_file)
    log_utils.add_lines(f"Experiment time: {((end - start) / 60):.2f} minute(s)", args.log_file)
    log_utils.add_lines(f"=============================", args.log_file)

    final_evaluation(NSFR, percept_dict, args)


if __name__ == "__main__":
    main()
