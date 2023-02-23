import datetime


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
        f.write(line_str + "\n")


def get_unused_args(c):
    unused_args = []
    used_args = []
    for body in c.body:
        if "in" == body.pred.name:
            unused_args.append(body.terms[0])
    for body in c.body:
        if not "in" in body.pred.name:
            for term in body.terms:
                if "O" in term.name and term.name not in used_args:
                    unused_args.remove(term)
                    used_args.append(term)
    return unused_args, used_args


