import config
import percept


def get_pred_res(args, data_type):
    pos_dataset_folder = config.buffer_path / "hide" / args.dataset / data_type / 'true'
    neg_dataset_folder = config.buffer_path / "hide" / args.dataset / data_type / 'false'
    pred_pos = percept.convert_data_to_tensor(args, pos_dataset_folder)
    pred_neg = percept.convert_data_to_tensor(args, neg_dataset_folder)

    # normalize the position
    value_max = max(pred_pos[:, :, :3].max(), pred_neg[:, :, :3].max())
    value_min = min(pred_pos[:, :, :3].min(), pred_neg[:, :, :3].min())
    pred_pos[:, :, :3] = percept.normalization(pred_pos[:, :, :3], value_max, value_min)
    pred_neg[:, :, :3] = percept.normalization(pred_neg[:, :, :3], value_max, value_min)

    if args.top_data < len(pred_pos):
        pred_pos = pred_pos[:args.top_data]
        pred_neg = pred_neg[:args.top_data]
    return pred_pos, pred_neg
