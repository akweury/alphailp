import config
import percept
import chart_utils
import numpy as np
import torch


def vertex_normalization(data):
    if len(data.shape) != 3:
        raise ValueError
    torch.set_printoptions(precision=2)

    ax = 0
    data[:, :, :3] = (data[:, :, :3] - data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[0]) / (
            data[:, :, ax:ax + 1].max(axis=1, keepdims=True)[0] - data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[
        0] + 1e-10)

    ax = 2
    data[:, :, ax] = data[:, :, ax] - data[:, :, ax].min(axis=1, keepdims=True)[0]
    # for i in range(len(data)):
    #     data_plot = np.zeros(shape=(5, 2))
    #     data_plot[:, 0] = data[i, :5, 0]
    #     data_plot[:, 1] = data[i, :5, 2]
    #     chart_utils.plot_scatter_chart(data_plot, config.buffer_path / "hide", show=True, title=f"{i}")
    return data


def get_pred_res(args, data_type):
    od_res = config.buffer_path / "hide" / f"{args.dataset}_pm_res_{data_type}.pth.tar"
    pred_pos, pred_neg = percept.convert_data_to_tensor(args, od_res)

    # normalize the position
    pred_pos_norm = vertex_normalization(pred_pos)
    pred_neg_norm = vertex_normalization(pred_neg)
    # value_max = max(pred_pos_norm[:, :, :3].max(), pred_neg[:, :, :3].max())
    # value_min = min(pred_pos_norm[:, :, :3].min(), pred_neg[:, :, :3].min())
    # pred_pos[:, :, :3] = percept.normalization(pred_pos[:, :, :3], value_max, value_min)
    # pred_neg[:, :, :3] = percept.normalization(pred_neg[:, :, :3], value_max, value_min)

    if args.top_data < len(pred_pos_norm):
        pred_pos_norm = pred_pos_norm[:args.top_data]
        pred_neg_norm = pred_neg_norm[:args.top_data]
    return pred_pos_norm, pred_neg_norm
