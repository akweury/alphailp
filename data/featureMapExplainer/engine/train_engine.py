# Created by jing at 26.06.23

import os
import torch
import shutil
import numpy as np
import datetime

from data.featureMapExplainer.engine import visual


# ----------------------------------- train/test epoch -----------------------------------------------------------------
def train_epoch(args):
    exp_start_time = f"start from {args.start_date} - {args.start_time}"
    epoch_time = datetime.datetime.now().strftime('%H:%M:%S')
    epoch_lr = f"lr={args.optimizer.param_groups[0]['lr']:.1e}"
    epoch_best_loss = f"best_loss: {float(args.eval_loss_best):.1e}"
    # ------------ switch to train mode -------------------
    args.model.train()
    losses = torch.tensor([0.0])

    for i, (dataX, dataY, train_i) in enumerate(args.train_loader):
        # put input and target to device
        dataX = dataX.float().permute(0, 3, 1, 2).to(args.device)
        dataY = dataY.float().to(args.device)
        # Wait for all kernels to finish
        if args.device != "cpu":
            torch.cuda.synchronize()

        # Clear the gradients
        args.optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        # Forward pass

        out = args.model(dataX)
        # Compute the loss
        args.loss = args.criterion(out, dataY) / int(args.batch_size)
        args.loss.backward()
        # Update the parameters
        args.optimizer.step()

        if i == 0:
            # print statistics
            np.set_printoptions(precision=5)
            torch.set_printoptions(sci_mode=True, precision=3)
            if args.loss is not None:
                print(f"Epoch[{args.epoch}] training loss: {args.loss:.2e} {epoch_lr} {epoch_best_loss} {epoch_time}")

    # save loss and plot
    # if args.args.normal_loss:
    #     plot_loss_per_axis(normal_loss_total, args, epoch, title="normal_loss")
    # if args.args.normal_huber_loss:
    #     plot_loss_per_axis(normal_loss_total, args, epoch, title="normal_huber_loss")
    # if args.args.g_loss:
    #     plot_loss_per_axis(g_loss_total, args, epoch, title="g_loss")
    # if args.args.light_loss:
    #     plot_loss_per_axis(light_loss_total, args, epoch, title="light_loss")
    # if args.args.albedo_loss:
    #     args.losses[9, epoch] = albedo_loss_total / len(args.train_loader.dataset)
    #     draw_line_chart(np.array([args.losses[9]]), args.output_folder,
    #                     log_y=True, label="albedo", epoch=epoch, start_epoch=0, title="albedo_loss", cla_leg=True)


def test_epoch(args):
    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        args.model.eval()
        # loop over the validation set
        for i, (dataX, dataY, train_i) in enumerate(args.test_loader):
            # send the input to the device
            dataX = dataX.float().permute(0, 3, 1, 2).to(args.device)
            dataY = dataY.float().to(args.device)
            # make the predictions and calculate the validation loss
            out = args.model(dataX)
            args.loss = args.criterion(out, dataY) / int(args.batch_size)

            # logs
            if i == 0:
                # print statistics
                np.set_printoptions(precision=5)
                torch.set_printoptions(sci_mode=True, precision=3)
                if args.loss is not None:
                    print(f"\t eval loss: {args.loss:.2e}")


def save_checkpoint(args):
    checkpoint_filename = os.path.join(args.output_path, 'checkpoint-' + str(args.epoch) + '.pth.tar')
    state = {'model': args.model}

    torch.save(state, checkpoint_filename)
    if args.is_best:
        best_filename = os.path.join(args.output_path, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)

    # remove previous checkpoint, but keep one in every 50 epochs
    if args.epoch > 0:
        prev_checkpoint_filename = os.path.join(args.output_path, 'checkpoint-' + str(args.epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            if args.epoch % 50 != 1:
                os.remove(prev_checkpoint_filename)


def main(args):
    ############ TRAINING LOOP ############
    args.losses_eval = torch.zeros((args.epochs))
    args.losses_train = torch.zeros((args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch
        train_epoch(args)
        # Learning rate scheduler
        args.lr_decayer.step()
        args.losses_train[epoch] = args.loss

        # evaluation
        test_epoch(args)
        args.losses_eval[epoch] = args.loss

        # draw line chart
        if epoch % 10 == 9:
            visual.draw_line_chart(args, "eval", log_y=True, label="eval_loss", title="eval_loss", cla_leg=True)
            visual.draw_line_chart(args, "train", log_y=True, label="train_loss", title="train_loss", cla_leg=True)
        if args.loss < args.eval_loss_best:
            args.eval_loss_best = args.loss
            args.is_best = True

        # Save checkpoint in case evaluation crashed
        save_checkpoint(args)

        if args.loss > 1e+4:
            print("loss is greater than 1e+4.")
            break
