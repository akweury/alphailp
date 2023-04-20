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
    # set up the environment, load the dataset
    args, rtpt, percept_dict = mechanics.init()

    # grouping objects to reduce the problem complexity
    obj_groups = mechanics.detect_obj_groups(args, percept_dict["val_pos"], percept_dict["val_neg"])
    eval_res_val = mechanics.eval_groups(percept_dict["val_pos"], percept_dict["val_neg"], obj_groups)
    is_done = mechanics.check_group_result(args, eval_res_val)

    # update arguments
    update_args(args, percept_dict, obj_groups)

    if False and is_done:
        # Dataset is too simple. Finish the program.
        eval_result_test = mechanics.eval_groups(percept_dict["test_pos"], percept_dict["test_neg"], obj_groups)
        is_done = mechanics.check_group_result(args, eval_result_test)
        log_utils.print_dataset_simple(args, is_done, eval_result_test)
    else:
        # ILP and PI system
        start = time.time()
        NSFR = pi.train_and_eval(args, percept_dict, obj_groups, rtpt)
        end = time.time()

        log_utils.add_lines(f"=============================", args.log_file)
        log_utils.add_lines(f"Experiment time: {((end - start) / 60):.2f} minute(s)", args.log_file)
        log_utils.add_lines(f"=============================", args.log_file)

        final_evaluation(NSFR, percept_dict, args)


if __name__ == "__main__":
    main()
