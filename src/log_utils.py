import datetime
import config


def create_file(exp_output_path, file_name):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    file_name = str(exp_output_path / f"log_{date_now}_{time_now}_{file_name}.txt")
    with open(file_name, "w") as f:
        f.write(f"{file_name} from {date_now}, {time_now}\n")

    return str(exp_output_path / file_name)


def create_log_file(exp_output_path):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    file_name = str(exp_output_path / f"log_{date_now}_{time_now}.txt")
    with open(file_name, "w") as f:
        f.write(f"Log from {date_now}, {time_now}")

    return str(exp_output_path / file_name)


def add_lines(line_str, log_file):
    print(line_str)
    with open(log_file, "a") as f:
        f.write(str(line_str) + "\n")


def get_unused_args(c):
    unused_args = []
    used_args = []
    for body in c.body:
        if "in" == body.pred.name:
            unused_args.append(body.terms[0])
    for body in c.body:
        if not "in" in body.pred.name:
            for term in body.terms:
                if "O" in term.name and term not in used_args:
                    unused_args.remove(term)
                    used_args.append(term)
    return unused_args, used_args


def write_clause_to_file(clauses, pi_clause_file):
    with open(pi_clause_file, "a") as f:
        for c in clauses:
            f.write(str(c) + "\n")


def write_predicate_to_file(invented_preds, inv_predicate_file):
    with open(inv_predicate_file, "a") as f:
        for inv_pred in invented_preds:
            arg_str = "("
            for a_i, a in enumerate(inv_pred.args):
                arg_str += str(a)
                if a_i != len(inv_pred.args) - 1:
                    arg_str += ","
            arg_str += ")"
            head = inv_pred.name + arg_str
            for body in inv_pred.body:
                clause_str = head + ":-" + str(body).replace(" ", "")[1:-1] + "."
                print(str(clause_str))
                f.write(str(clause_str) + "\n")


def print_time(args, iteration, step, max_step):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    add_lines(f"\n({date_now} {time_now}) Iteration: {iteration} Step {step}/{iteration}",
              args.log_file)


def print_dataset_simple(args, is_done, eval_res):
    if is_done:
        add_lines(f"Dataset is too simple!", args.log_file)
        add_lines(f"shape group score 1: {eval_res['shape_group']['score_1']}", args.log_file)
        add_lines(f"shape group score 2: {eval_res['shape_group']['score_2']}", args.log_file)
        if eval_res["shape_group"]["result"][0] > args.group_conf_th:
            add_lines(f"Distinguishable by group types and its numbers.", args.log_file)
        if eval_res["shape_group"]["result"][1] > args.group_conf_th:
            add_lines(f"Distinguishable by group types.", args.log_file)

        add_lines(f"color score 1: {eval_res['color']['score_1']}", args.log_file)
        add_lines(f"color score 2: {eval_res['color']['score_2']}", args.log_file)
        if eval_res["color"]["result"][0] > args.group_conf_th:
            add_lines(f"Distinguishable by color types and its numbers.", args.log_file)
        if eval_res["color"]["result"][1] > args.group_conf_th:
            add_lines(f"Distinguishable by color types.", args.log_file)

        add_lines(f"shape score 1: {eval_res['shape']['score_1']}", args.log_file)
        add_lines(f"shape score 2: {eval_res['shape']['score_2']}", args.log_file)
        if eval_res["shape"]["result"][0] > args.group_conf_th:
            add_lines(f"Distinguishable by shape types and its numbers.", args.log_file)
        if eval_res["shape"]["result"][1] > args.group_conf_th:
            add_lines(f"Distinguishable by shape types.", args.log_file)
