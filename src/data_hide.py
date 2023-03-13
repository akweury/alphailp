import config
import percept


def get_pred_res(args, data_type):
    pos_dataset_folder = config.buffer_path / "hide" / args.dataset / data_type / 'true'
    neg_dataset_folder = config.buffer_path / "hide" / args.dataset / data_type / 'false'
    pos_pred = percept.convert_data_to_tensor(args, pos_dataset_folder)
    neg_pred = percept.convert_data_to_tensor(args, neg_dataset_folder)

    # normalize the position
    max_value = max(pos_pred[:, :, :3].max(), neg_pred[:, :, :3].max())
    min_value = min(pos_pred[:, :, :3].min(), neg_pred[:, :, :3].min())
    pos_pred[:, :, :3] = percept.normalization(pos_pred[:, :, :3], max_value, min_value)
    neg_pred[:, :, :3] = percept.normalization(neg_pred[:, :, :3], max_value, min_value)

    if args.top_data < len(pos_pred):
        pos_pred = pos_pred[:args.top_data]
        neg_pred = neg_pred[:args.top_data]
    return pos_pred, neg_pred
