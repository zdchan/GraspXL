import colorsys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# connection between the 8 points of 3d bbox
BONES_3D_BBOX = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
]


def plot_2d_bbox(bbox_2d, bones, color, ax):
    if ax is None:
        axx = plt
    else:
        axx = ax
    colors = cm.rainbow(np.linspace(0, 1, len(bbox_2d)))
    for pt, c in zip(bbox_2d, colors):
        axx.scatter(pt[0], pt[1], color=c, s=50)

    if bones is None:
        bones = BONES_3D_BBOX
    for bone in bones:
        sidx, eidx = bone
        # bottom of bbox is white
        if min(sidx, eidx) >= 4:
            color = "w"
        axx.plot(
            [bbox_2d[sidx][0], bbox_2d[eidx][0]],
            [bbox_2d[sidx][1], bbox_2d[eidx][1]],
            color,
        )
    return axx


def plot_3d_bbox(ax, bbox_pts, bones, color):
    assert isinstance(bbox_pts, np.ndarray)
    assert len(bbox_pts.shape) == 2
    assert bbox_pts.shape[1] == 3
    ax.scatter(bbox_pts[:, 0], bbox_pts[:, 1], bbox_pts[:, 2], color=color)
    if bones is None:
        bones = BONES_3D_BBOX
    for start_end in bones:
        start_idx, end_idx = start_end
        ax.plot(
            [bbox_pts[start_idx, 0], bbox_pts[end_idx, 0]],
            [bbox_pts[start_idx, 1], bbox_pts[end_idx, 1]],
            [bbox_pts[start_idx, 2], bbox_pts[end_idx, 2]],
            color=color,
            linewidth=1,
        )
    return ax


def random_cmap(
    nlabels, type="bright", first_color_black=True, last_color_black=False, verbose=True
):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    Example:
    cmap = rand_cmap(33, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """

    if type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Display colorbar
    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )

    return random_colormap


def imshow_attn(im, att, im_alpha, att_alpha, ax, cmap="gnuplot"):
    """
    Plot image with attention overlayed on it.
    im: color or BW image (only one), (dim, dim, channel)
    att: attention weights (dim, dim)
    """
    assert isinstance(im, np.ndarray)
    assert isinstance(att, (np.ndarray, torch.FloatTensor))
    if isinstance(att, np.ndarray):
        att = torch.FloatTensor(att)
    assert len(im.shape) >= 2
    assert len(att.shape) == 2
    dim0, dim1 = im.shape[:2]
    att_dim0, att_dim1 = att.shape[:2]

    ax.imshow(im, alpha=im_alpha)
    ax.imshow(
        F.interpolate(att[None, None, :, :], (dim0, dim1)).squeeze().numpy(),
        cmap=cmap,
        alpha=att_alpha,
    )
    return ax


def plot_origin(ax):
    ax.plot([0], [0], [0], "o", color="purple", markersize=12, alpha=0.3)


def axis_equal_3d(ax):
    """
    Make the the aspect ratio the same for xyz
    """
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def plot_grad_flow(named_parameters):
    """
    Plot avg gradient flow per layer.
    Usage:
        loss = self.criterion(outputs, labels)
        loss.backward()
        plot_grad_flow(model.named_parameters())
        https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    fig = plt.figure()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return fig


def plot_bbox(xyxy, line="-", color="r", linewidth=2):
    """
    Plot a bounding box on a figure.
    """
    if not isinstance(xyxy, list):
        xyxy = xyxy.tolist()
    x1, y1, x2, y2 = xyxy

    top_left = [x1, y1]
    down_left = [x1, y2]
    top_right = [x2, y1]
    down_right = [x2, y2]

    plt.plot(
        [top_left[0], top_right[0]],
        [top_left[1], top_right[1]],
        linestyle=line,
        color=color,
        linewidth=linewidth,
    )
    plt.plot(
        [top_left[0], down_left[0]],
        [top_left[1], down_left[1]],
        linestyle=line,
        color=color,
        linewidth=linewidth,
    )
    plt.plot(
        [top_right[0], down_right[0]],
        [top_right[1], down_right[1]],
        linestyle=line,
        color=color,
        linewidth=linewidth,
    )
    plt.plot(
        [down_left[0], down_right[0]],
        [down_left[1], down_right[1]],
        linestyle=line,
        color=color,
        linewidth=linewidth,
    )


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D
    numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image
    in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tobytes())


def concat_pil_images(images):
    """
    Put a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def stack_pil_images(images):
    """
    Stack a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def im_list_to_plt(image_list, figsize, title_list=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(image_list), figsize=figsize)
    for idx, (ax, im) in enumerate(zip(axes, image_list)):
        ax.imshow(im)
        ax.set_title(title_list[idx])
    fig.tight_layout()
    im = fig2img(fig)
    plt.close()
    return im
