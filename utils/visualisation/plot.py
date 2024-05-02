import warnings
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

warnings.filterwarnings("ignore")


def plot_results(
    fig,
    rgb_img,
    grasp_q_img,
    grasp_angle_img,
    depth_img=None,
    no_grasps=1,
    grasp_width_img=None,
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(
        grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps
    )

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title("RGB")
    ax.axis("off")

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap="gray")
        ax.set_title("Depth")
        ax.axis("off")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title("Grasp")
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Q")
    ax.axis("off")
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title("Angle")
    ax.axis("off")
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap="jet", vmin=0, vmax=100)
    ax.set_title("Width")
    ax.axis("off")
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
    fig,
    grasps=None,
    save=False,
    rgb_img=None,
    grasp_q_img=None,
    grasp_angle_img=None,
    no_grasps=1,
    grasp_width_img=None,
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(
            grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps
        )

    plt.ion()
    plt.clf()

    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title("Grasp")
    ax.axis("off")

    plt.pause(0.1)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.savefig("results/{}.png".format(time))


def save_results(
    rgb_img,
    grasp_q_img,
    grasp_angle_img,
    depth_img=None,
    no_grasps=1,
    grasp_width_img=None,
    result_dir=None,
):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """

    os.makedirs(result_dir, exist_ok=True)

    gs = detect_grasps(
        grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps
    )

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    # ax.imshow(rgb_img)
    ax.set_title("RGB")
    ax.axis("off")
    # fig.savefig("results/rgb.png")
    fig.savefig(os.path.join(result_dir, "rgb.png"))

    # if depth_img.any():
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.ion()
    #     plt.clf()
    #     ax = plt.subplot(111)
    #     ax.imshow(depth_img, cmap='gray')
    #     for g in gs:
    #         g.plot(ax)
    #     ax.set_title('Depth')
    #     ax.axis('off')
    #     fig.savefig('results/depth.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    # ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title("Grasp")
    ax.axis("off")
    # fig.savefig("results/grasp.png")
    fig.savefig(os.path.join(result_dir, "grasp.png"))

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    # plot = ax.imshow(grasp_q_img, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Q")
    ax.axis("off")
    # plt.colorbar(plot)
    # fig.savefig("results/quality.png")
    fig.savefig(os.path.join(result_dir, "quality.png"))

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    # plot = ax.imshow(grasp_angle_img, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title("Angle")
    ax.axis("off")
    # plt.colorbar(plot)
    # fig.savefig("results/angle.png")
    fig.savefig(os.path.join(result_dir, "angle.png"))

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    # plot = ax.imshow(grasp_width_img, cmap="jet", vmin=0, vmax=100)
    ax.set_title("Width")
    ax.axis("off")
    # plt.colorbar(plot)
    # fig.savefig("results/width.png")
    fig.savefig(os.path.join(result_dir, "width.png"))

    fig.canvas.draw()
    plt.close(fig)


def save_results_grasp(
    rgb_img,
    grasp_q_img,
    grasp_angle_img,
    grasp_width_img,
    gr_grasp_q_img,
    gr_grasp_angle_img,
    gr_grasp_width_img,
    depth_img=None,
    no_grasps=1,
    prompt=None,
    result_dir=None,
):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :param prompt: Prompt text for the figure title
    :param result_dir: Directory to save the figures
    :return:
    """

    os.makedirs(result_dir, exist_ok=True)

    gs = detect_grasps(
        grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps
    )

    gr_gs = detect_grasps(
        gr_grasp_q_img,
        gr_grasp_angle_img,
        width_img=gr_grasp_width_img,
        no_grasps=no_grasps,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(prompt)

    ax1.imshow(rgb_img)
    ax1.set_title("Predicted")
    for g in gs:
        g.plot(ax1)
    ax1.axis("off")

    ax2.imshow(rgb_img)
    ax2.set_title("Ground Truth")
    for g in gr_gs:
        g.plot(ax2)
    ax2.axis("off")

    fig.savefig(os.path.join(result_dir, "grasp.png"), bbox_inches="tight")


def draw_grasps_on_image(
    rgb_img,
    grasp_q_img,
    grasp_angle_img,
    depth_img=None,
    no_grasps=1,
    grasp_width_img=None,
):
    gs = detect_grasps(
        grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps
    )

    fig, ax = plt.subplots()
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )

    return image
