import datetime


def create_log_file(exp_output_path):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    with open(str(exp_output_path / f"log_{date_now}_{time_now}.txt"), "w") as f:
        f.write(f"Log from {date_now}, {time_now}")

    return str(exp_output_path / "log.txt")


def add_lines(line_str, log_file):
    print(line_str)
    with open(log_file, "a") as f:
        f.write(line_str)

