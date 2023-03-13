import config
import percept


def get_pred_res(args, data_type):
    pos_dataset_folder = config.buffer_path / "hide" / args.dataset / data_type / 'true'
    neg_dataset_folder = config.buffer_path / "hide" / args.dataset / data_type / 'false'
    pos_pred = percept.convert_data_to_tensor(args, pos_dataset_folder)
    neg_pred = percept.convert_data_to_tensor(args, neg_dataset_folder)

    # normalize the position
    max_value = max(pos_pred.max(), neg_pred.max())
    min_value = min(pos_pred.min(), neg_pred.min())
    pos_pred_norm = percept.normalization(pos_pred, max_value, min_value)
    neg_pred_norm = percept.normalization(neg_pred, max_value, min_value)

    if args.top_data < len(pos_pred_norm):
        pos_pred_norm = pos_pred_norm[:args.top_data]
        neg_pred_norm = neg_pred_norm[:args.top_data]
    return pos_pred_norm, neg_pred_norm
