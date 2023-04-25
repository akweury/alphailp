# Created by shaji on 21-Mar-23

import time
import datetime

from mechanics import *
import log_utils


date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def main():
    # set up the environment, load the dataset and results from perception models
    args, rtpt, percept_dict, obj_groups = init()

    # ILP and PI system
    start = time.time()
    # update arguments
    update_args(args, percept_dict, obj_groups)
    # describe the scenes with clauses, invent new predicates if necessary
    NSFR = train_and_eval(args, rtpt)
    end = time.time()

    log_utils.add_lines(f"=============================", args.log_file)
    log_utils.add_lines(f"Experiment time: {((end - start) / 60):.2f} minute(s)", args.log_file)
    log_utils.add_lines(f"=============================", args.log_file)

    final_evaluation(NSFR, args)


if __name__ == "__main__":
    main()
