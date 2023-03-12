import config
import percept


def get_pred_res(args, data_type):
    pos_dataset_folder = config.data_path / "hide" / args.dataset / data_type / 'true'
    neg_dataset_folder = config.data_path / "hide" / args.dataset / data_type / 'false'
    pos_pred = percept.convert_data_to_tensor(args, pos_dataset_folder)
    neg_pred = percept.convert_data_to_tensor(args, neg_dataset_folder)
    if args.top_data < len(pos_pred):
        pos_pred = pos_pred[:args.top_data]
        neg_pred = neg_pred[:args.top_data]
    return pos_pred, neg_pred
