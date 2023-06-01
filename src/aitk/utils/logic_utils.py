# Created by jing at 30.05.23

"""
Root utils file, only import modules that don't belong to this project.
"""
import os
import torch


def get_index_by_predname(pred_str, atoms):
    indices = []
    for p_i, p_str in enumerate(pred_str):
        p_indices = []
        for i, atom in enumerate(atoms):
            if atom.pred.name == p_str:
                p_indices.append(i)
        indices.append(p_indices)
    return indices


def data_ordering(data):
    data_ordered = torch.zeros(data.shape)
    delta = data[:, :, :3].max(dim=1, keepdims=True)[0] - data[:, :, :3].min(dim=1, keepdims=True)[0]
    order_axis = torch.argmax(delta, dim=2)
    for data_i in range(len(data)):
        data_order_i = data[data_i,:,order_axis[data_i]].sort(dim=0)[1].squeeze(1)
        data_ordered[data_i] = data[data_i,data_order_i,:]

    return data_ordered


def convert_data_to_tensor(args, od_res):
    if os.path.exists(od_res):
        pm_res = torch.load(od_res)
        pos_pred = pm_res['pos_res']
        neg_pred = pm_res['neg_res']
    else:
        raise ValueError
    # data_files = glob.glob(str(pos_dataset_folder / '*.json'))
    # data_tensors = torch.zeros((len(data_files), args.e, 9))
    # for d_i, data_file in enumerate(data_files):
    #     with open(data_file) as f:
    #         data = json.load(f)
    #     data_tensor = torch.zeros(1, args.e, 9)
    #     for o_i, obj in enumerate(data["objects"]):
    #
    #         data_tensor[0, o_i, 0:3] = torch.tensor(obj["position"])
    #         if "blue" in obj["material"]:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([0, 0, 1])
    #         elif "green" in obj["material"]:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([0, 1, 0])
    #         else:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([1, 0, 0])
    #         if "sphere" in obj["material"]:
    #             data_tensor[0, o_i, 6] = 0.99
    #         if "cube" in obj["material"]:
    #             data_tensor[0, o_i, 7] = 0.99
    #         data_tensor[0, o_i, 8] = 0.99
    #     data_tensors[d_i] = data_tensor[0]

    return pos_pred, neg_pred


def vertex_normalization(data):
    return data

    if len(data.shape) != 3:
        raise ValueError

    ax = 0
    min_value = data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[0].repeat(1,data.shape[1], 3)
    max_value = data[:, :, ax:ax + 1].max(axis=1, keepdims=True)[0].repeat(1,data.shape[1], 3)
    data[:, :, :3] = (data[:, :, :3] - min_value) / (max_value - min_value + 1e-10)

    ax = 2
    data[:, :, ax] = data[:, :, ax] - data[:, :, ax].min(axis=1, keepdims=True)[0]
    # for i in range(len(data)):
    #     data_plot = np.zeros(shape=(5, 2))
    #     data_plot[:, 0] = data[i, :5, 0]
    #     data_plot[:, 1] = data[i, :5, 2]
    #     chart_utils.plot_scatter_chart(data_plot, config.buffer_path / "hide", show=True, title=f"{i}")
    return data
