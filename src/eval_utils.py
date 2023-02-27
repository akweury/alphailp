def is_sn(score, data_size):
    if score[1] == data_size:
        return True
    return False


def is_sn_th_good(score, data_size, threshold):
    if score[1] / data_size > threshold:
        return True
    else:
        return False


def is_nc(score, data_size, threshold):
    if score[1] + score[3] == data_size:
        return True
    else:
        return False


def is_nc_th_good(score, data_size, threshold):
    if (score[1] + score[3]) / data_size > threshold and score[1] > score[3]:
        return True
    else:
        return False


def is_sc(score, data_size, threshold):
    if score[0] + score[1] == data_size:
        return True
    else:
        return False


def is_sc_th_good(score, data_size, threshold):
    if (score[0] + score[1]) / data_size > threshold and score[1] > 0:
        return True
    else:
        return False


def is_uc_th_good(score, threshold):
    if score[0] < score[1] and score[2] < score[1] and score[3] < score[1]:
        return True
    else:
        return False
