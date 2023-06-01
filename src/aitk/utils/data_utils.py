# Created by jing at 31.05.23

import config


def prop2index(props, g_type="group"):
    indices = []
    if g_type == "group":
        for prop in props:
            indices.append(config.group_tensor_index[prop])

    elif g_type == "object":
        for prop in props:
            indices.append(config.obj_tensor_index[prop])
    else:
        raise ValueError
    return indices


