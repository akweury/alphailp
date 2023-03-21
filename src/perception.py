# Created by shaji on 21-Mar-23
from src import config, percept, data_hide, log_utils


def get_perception_predictions(args, val_pos_loader, val_neg_loader, train_pos_loader, train_neg_loader,
                               test_pos_loader, test_neg_loader):
    if args.dataset_type == "kandinsky":
        pm_val_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_val.pth.tar")
        pm_train_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_train.pth.tar")
        pm_test_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_test.pth.tar")

        val_pos_pred, val_neg_pred = percept.eval_images(args, pm_val_res_file, args.device, val_pos_loader,
                                                         val_neg_loader)
        train_pos_pred, train_neg_pred = percept.eval_images(args, pm_train_res_file, args.device, train_pos_loader,
                                                             train_neg_loader)
        test_pos_pred, test_neg_pred = percept.eval_images(args, pm_test_res_file, args.device, test_pos_loader,
                                                           test_neg_loader)

    elif args.dataset_type == "hide":
        train_pos_pred, train_neg_pred = data_hide.get_pred_res(args, "train")
        test_pos_pred, test_neg_pred = data_hide.get_pred_res(args, "test")
        if args.small_data:
            val_pos_pred, val_neg_pred = data_hide.get_pred_res(args, "val_s")
        else:
            val_pos_pred, val_neg_pred = data_hide.get_pred_res(args, "val")

    else:
        raise ValueError

    log_utils.add_lines(f"==== positive image number: {len(val_pos_pred)}", args.log_file)
    log_utils.add_lines(f"==== negative image number: {len(val_neg_pred)}", args.log_file)
    pm_prediction_dict = {
        'val_pos': val_pos_pred,
        'val_neg': val_neg_pred,
        'train_pos': train_pos_pred,
        'train_neg': train_neg_pred,
        'test_pos': test_pos_pred,
        'test_neg': test_neg_pred
    }

    return pm_prediction_dict
