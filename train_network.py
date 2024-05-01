import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from utils.visualisation.plot import draw_grasps_on_image, save_results_grasp


def parse_args():
    parser = argparse.ArgumentParser(description="Train network")

    # Network
    parser.add_argument(
        "--network",
        type=str,
        default="grconvnet3",
        help="Network name in inference/models",
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help="Input image size for the network"
    )
    parser.add_argument(
        "--use-depth", type=int, default=1, help="Use Depth image for training (1/0)"
    )
    parser.add_argument(
        "--use-rgb", type=int, default=1, help="Use RGB image for training (1/0)"
    )
    parser.add_argument(
        "--use-instruction",
        default=False,
        action="store_true",
        help="Use instruction for training",
    )
    parser.add_argument(
        "--use-dropout", type=int, default=1, help="Use dropout for training (1/0)"
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        default=0.1,
        help="Dropout prob for training (0-1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--channel-size",
        type=int,
        default=32,
        help="Internal channel size for the network",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.25, help="Threshold for IOU matching"
    )

    # Datasets
    parser.add_argument(
        "--dataset", type=str, help='Dataset Name ("cornell" or "jaquard")'
    )
    parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Fraction of data for training (remainder is validation)",
    )
    parser.add_argument(
        "--ds-shuffle", action="store_true", default=False, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--ds-easy-setting",
        action="store_true",
        default=False,
        help="Easy setting without augmentation",
    )

    parser.add_argument(
        "--ds-rotate",
        type=float,
        default=0.0,
        help="Shift the start point of the dataset to use a different test/train split",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Dataset workers")

    # Training
    parser.add_argument(
        "--pretrained", action="store_true", default=False, help="Load from pretrained",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--batches-per-epoch", type=int, default=1000, help="Batches per Epoch"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="Optmizer for the training. (adam or SGD)",
    )

    # Logging etc.
    parser.add_argument(
        "--description", type=str, default="", help="Training description"
    )
    parser.add_argument("--logdir", type=str, default="logs/", help="Log directory")
    parser.add_argument(
        "--vis", action="store_true", help="Visualise the training process"
    )
    parser.add_argument(
        "--cpu",
        dest="force_cpu",
        action="store_true",
        default=False,
        help="Force code to run in CPU mode",
    )
    parser.add_argument(
        "--random-seed", type=int, default=123, help="Random seed for numpy"
    )
    parser.add_argument(
        "--seen",
        type=int,
        default=1,
        help="Flag for using seen classes, only work for Grasp-Anything dataset",
    )

    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_threshold, dir_vis=None):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {"correct": 0, "failed": 0, "loss": 0, "losses": {}}

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:

            if not isinstance(x, list):
                xc = x.to(device)
            else:
                xc = (x[0].to(device), x[1].to(device))

            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd["loss"]

            results["loss"] += loss.item() / ld
            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(
                lossd["pred"]["pos"],
                lossd["pred"]["cos"],
                lossd["pred"]["sin"],
                lossd["pred"]["width"],
            )

            s = evaluation.calculate_iou_match(
                q_out,
                ang_out,
                val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                no_grasps=1,
                grasp_width=w_out,
                threshold=iou_threshold,
            )

            if s:
                results["correct"] += 1
            else:
                results["failed"] += 1

    return results


def train(
    epoch,
    net,
    device,
    train_data,
    optimizer,
    batches_per_epoch,
    vis=False,
    dir_vis=None,
    ds_easy_setting=False
):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {"loss": 0, "losses": {}}

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.

    if ds_easy_setting:
        batches_per_epoch = len(train_data)


    while batch_idx <= batches_per_epoch:
        for x, y, didx, rot, zoom in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            # check if x is along with condition or not
            if not isinstance(x, list):
                xc = x.to(device)
            else:
                xc = (x[0].to(device), x[1].to(device))

            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd["loss"]

            if batch_idx % 100 == 0:
                logging.info(
                    "Epoch: {}, Batch: {}, Loss: {:0.4f}".format(
                        epoch, batch_idx, loss.item()
                    )
                )

            results["loss"] += loss.item()
            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis and batch_idx % 100 == 0:

                y_pos = y[0].detach()
                y_cos = y[1].detach()
                y_sin = y[2].detach()
                y_width = y[3].detach()

                with torch.no_grad():
                    imgs = []
                    n_img = min(4, x[0].shape[0])
                    for idx in range(n_img):

                        q_img_gr, ang_img_gr, width_img_gr = post_process_output(
                            y_pos[idx], y_cos[idx], y_sin[idx], y_width[idx]
                        )

                        q_img, ang_img, width_img = post_process_output(
                            lossd["pred"]["pos"][idx].detach(),
                            lossd["pred"]["cos"][idx].detach(),
                            lossd["pred"]["sin"][idx].detach(),
                            lossd["pred"]["width"][idx].detach(),
                        )

                        # draw_gt_image = draw_grasps_on_image(
                        #     rgb_img=train_data.dataset.get_rgb(
                        #         didx[idx], rot[idx], zoom[idx], normalise=False
                        #     ),
                        #     grasp_q_img=q_img,
                        #     grasp_angle_img=ang_img,
                        #     no_grasps=1,
                        #     grasp_width_img=width_img,
                        # )

                        # print(draw_gt_image.shape)
                        # print(x[0][idx,].numpy().squeeze()[[2, 1, 0], :, :].shape)

                        # imgs.extend(
                        #     [x[0][idx,].numpy().squeeze()[[2, 1, 0], :, :]]
                        #     + [yi[idx,].numpy().squeeze() for yi in y]
                        #     + [x[0][idx,].numpy().squeeze()[[2, 1, 0], :, :]]
                        #     + [
                        #         pc[idx,].detach().cpu().numpy().squeeze()
                        #         for pc in lossd["pred"].values()
                        #     ]
                        # )

                        path_to_save = os.path.join(
                            dir_vis,
                            "train_vis_epoch_{}_batch_{}_{}".format(
                                epoch, batch_idx, idx
                            ),
                        )

                        save_results_grasp(
                            rgb_img=train_data.dataset.get_rgb(
                                didx.numpy().tolist()[idx],
                                rot.numpy().tolist()[idx],
                                zoom.numpy().tolist()[idx],
                                normalise=False,
                            ),
                            grasp_q_img=q_img,
                            grasp_angle_img=ang_img,
                            grasp_width_img=width_img,
                            gr_grasp_q_img=q_img_gr,
                            gr_grasp_angle_img=ang_img_gr,
                            gr_grasp_width_img=width_img_gr,
                            no_grasps=1,
                            prompt=train_data.dataset.get_prompt(
                                didx.numpy().tolist()[idx]
                            ),
                            result_dir=path_to_save,
                        )

                    # gridshow(
                    #     "Display",
                    #     imgs,
                    #     [
                    #         (xc[0].min().item(), xc[0].max().item()),
                    #         (0.0, 1.0),
                    #         (0.0, 1.0),
                    #         (-1.0, 1.0),
                    #         (0.0, 1.0),
                    #     ]
                    #     * 2
                    #     * n_img,
                    #     [cv2.COLORMAP_BONE] * 10 * n_img,
                    #     10,
                    #     path_to_save=path_to_save,
                    # )

                    # cv2.waitKey(2)

    results["loss"] /= batch_idx
    for l in results["losses"]:
        results["losses"][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    net_desc = "{}_{}".format(dt, "_".join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    vis_save_folder = os.path.join(save_folder, "train_vis")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(vis_save_folder, exist_ok=True)

    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, "commandline_args.json")
        with open(params_path, "w") as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, "log"),
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info("Loading {} Dataset...".format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    # dataset = Dataset(
    #     args.dataset_path,
    #     output_size=args.input_size,
    #     ds_rotate=args.ds_rotate,
    #     random_rotate=True,
    #     random_zoom=True,
    #     include_depth=args.use_depth,
    #     include_rgb=args.use_rgb,
    #     include_embedding=args.use_instruction,
    #     seen=args.seen,
    # )

    if args.ds_easy_setting:
        dataset = Dataset(
            args.dataset_path,
            output_size=args.input_size,
            ds_rotate=args.ds_rotate,
            random_rotate=False,
            random_zoom=False,
            include_depth=args.use_depth,
            include_rgb=args.use_rgb,
            include_embedding=args.use_instruction,
            seen=args.seen,
        )
    else:
        dataset = Dataset(
            args.dataset_path,
            output_size=args.input_size,
            ds_rotate=args.ds_rotate,
            random_rotate=True,
            random_zoom=True,
            include_depth=args.use_depth,
            include_rgb=args.use_rgb,
            include_embedding=args.use_instruction,
            seen=args.seen,
        )

    logging.info("Dataset size is {}".format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info("Training size: {}".format(len(train_indices)))
    logging.info("Validation size: {}".format(len(val_indices)))

    # Creating data samplers and loaders
    if args.ds_easy_setting:

        train_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=torch.utils.data.SequentialSampler(train_indices),
            shuffle=False,
        )

        val_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=torch.utils.data.SequentialSampler(val_indices),
            shuffle=False,
        )
    else:

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler,
        )
        val_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=val_sampler,
        )

    logging.info("Done")

    # Load the network
    logging.info(f"Loading Network {args.network}...")
    input_channels = 1 * args.use_depth + 3 * args.use_rgb

    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size,
    )

    if args.network == 'trans_ragt':
        net.load_state_dict(torch.load("weights/mobilevit_s.pt"), strict=False)
    elif args.network == 'trans_grconvnet' and args.pretrained:
        logging.info("Loading pretrained model...")
        net.load_state_dict(torch.load("weights/model_grasp_anything_state.pth"), strict=False)

    net = net.to(device)
    # I want to print number of trainable parameter of net
    logging.info(
        "Number of trainable parameters: {}".format(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )
    )

    logging.info("Done")

    if args.optim.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optim.lower() == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError("Optimizer {} is not implemented".format(args.optim))

    # Print model architecture.
    # summary(net, (input_channels, args.input_size, args.input_size))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, args.input_size, args.input_size))
    # sys.stdout = sys.__stdout__
    # f.close()

    

    best_iou = 0.0
    scheduler_step_size = 10  # Number of epochs after which to reduce the learning rate
    scheduler_gamma = 0.1  # Factor by which to reduce the learning rate

    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    for epoch in range(args.epochs):
        logging.info("Beginning Epoch {:02d}".format(epoch))
        train_results = train(
            epoch,
            net,
            device,
            train_data,
            optimizer,
            args.batches_per_epoch,
            vis=args.vis,
            dir_vis=vis_save_folder,
            ds_easy_setting=args.ds_easy_setting
        )

        # Log training losses to tensorboard
        tb.add_scalar("loss/train_loss", train_results["loss"], epoch)
        for n, l in train_results["losses"].items():
            tb.add_scalar("train_loss/" + n, l, epoch)

        # Run Validation
        logging.info("Validating...")
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info(
            "%d/%d = %f"
            % (
                test_results["correct"],
                test_results["correct"] + test_results["failed"],
                test_results["correct"]
                / (test_results["correct"] + test_results["failed"]),
            )
        )

        # Log validation results to tensorbaord
        tb.add_scalar(
            "loss/IOU",
            test_results["correct"]
            / (test_results["correct"] + test_results["failed"]),
            epoch,
        )
        tb.add_scalar("loss/val_loss", test_results["loss"], epoch)
        for n, l in test_results["losses"].items():
            tb.add_scalar("val_loss/" + n, l, epoch)

        # Save best performing network
        iou = test_results["correct"] / (
            test_results["correct"] + test_results["failed"]
        )
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(
                net, os.path.join(save_folder, "epoch_%02d_iou_%0.2f" % (epoch, iou))
            )
            best_iou = iou

        # Step the scheduler
        scheduler.step()


if __name__ == "__main__":
    run()
