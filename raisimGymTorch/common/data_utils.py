"""
This file contains functions that are used to perform data augmentation.
"""
import cv2
import jpeg4py as jpeg
import numpy as np
import torch
from loguru import logger
# from common.transforms import perspective_projection
from scipy.spatial import Delaunay
from skimage.transform import resize, rotate


# https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False
):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image(
    cvimg,
    bbox,
    scale,
    rot,
    out_shape,
    interpl_strategy,
    gauss_kernel=5,
    gauss_sigma=8.0,
):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot
    )

    # anti-aliasing
    blur = cv2.GaussianBlur(img, (gauss_kernel, gauss_kernel), gauss_sigma)
    img_patch = cv2.warpAffine(
        blur, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy
    )
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        inv=True,
    )

    return img_patch, trans, inv_trans


def augm_params(is_train, flip_prob, noise_factor, rot_factor, scale_factor):
    """Get augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if is_train:
        # We flip with probability 1/2
        if np.random.uniform() <= flip_prob:
            flip = 1
            assert False, "Flipping not supported"

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot = min(
            2 * rot_factor,
            max(
                -2 * rot_factor,
                np.random.randn() * rot_factor,
            ),
        )

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(
            1 + scale_factor,
            max(
                1 - scale_factor,
                np.random.randn() * scale_factor + 1,
            ),
        )
        # but it is zero with probability 3/5
        if np.random.uniform() <= 0.6:
            rot = 0

    augm_dict = {}
    augm_dict["flip"] = flip
    augm_dict["pn"] = pn
    augm_dict["rot"] = rot
    augm_dict["sc"] = sc
    return augm_dict


def rgb_processing(is_train, rgb_img, center, bbox_dim, augm_dict, img_res):
    rot = augm_dict["rot"]
    flip = augm_dict["flip"]
    sc = augm_dict["sc"]
    pn = augm_dict["pn"]
    scale = sc * bbox_dim

    crop_dim = int(scale * 200)
    # faster cropping!!
    rgb_img = generate_patch_image(
        rgb_img,
        [center[0], center[1], crop_dim, crop_dim],
        1.0,
        rot,
        [img_res, img_res],
        cv2.INTER_CUBIC,
    )[0]

    # flip the image
    if flip:
        rgb_img = flip_img(rgb_img)

    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
    return rgb_img


def transform_kp2d(kp2d, bbox):
    # bbox: (cx, cy, scale) in the original image space
    # scale is normalized
    assert isinstance(kp2d, np.ndarray)
    assert len(kp2d.shape) == 2
    cx, cy, scale = bbox
    s = 200 * scale  # to px
    cap_dim = 1000  # px
    factor = cap_dim / (1.5 * s)
    kp2d_cropped = np.copy(kp2d)
    kp2d_cropped[:, 0] -= cx - 1.5 / 2 * s
    kp2d_cropped[:, 1] -= cy - 1.5 / 2 * s
    kp2d_cropped[:, 0] *= factor
    kp2d_cropped[:, 1] *= factor
    return kp2d_cropped


def j2d_processing(kp, center, bbox_dim, augm_dict, img_res):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    scale = augm_dict["sc"] * bbox_dim
    rot = augm_dict["rot"]
    flip = augm_dict["flip"]

    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, 0:2] = transform(
            kp[i, 0:2] + 1,
            center,
            scale,
            [img_res, img_res],
            rot=rot,
        )
    # convert to normalized coordinates
    kp = normalize_kp2d_np(kp, img_res)
    # flip the x coordinates
    if flip:
        kp = flip_kp(kp)
    kp = kp.astype("float32")
    return kp


def j3d_processing(S, augm_dict):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    rot = augm_dict["rot"]
    flip = augm_dict["flip"]
    # in-plane rotation
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = -rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    S[:, :-1] = np.einsum("ij,kj->ki", rot_mat, S[:, :-1])
    # flip the x coordinates
    if flip:
        S = flip_kp(S)
    S = S.astype("float32")
    return S


def pose_processing(pose, augm_dict):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    rot = augm_dict["rot"]
    flip = augm_dict["flip"]
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], rot)
    # flip the pose parameters
    if flip:
        pose = flip_pose(pose)
    # (72),float
    pose = pose.astype("float32")
    return pose


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]

    if not rot == 0:
        # Remove padding

        new_img = rotate(new_img, rot)  # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # resize image
    new_img = resize(new_img, tuple(res))  # scipy.misc.imresize(new_img, res)
    return new_img


def crop_cv2(img, center, scale, res, rot=0):
    c_x, c_y = center
    c_x, c_y = int(round(c_x)), int(round(c_y))
    patch_width, patch_height = int(round(res[0])), int(round(res[1]))
    bb_width = bb_height = int(round(scale * 200.0))

    trans = gen_trans_from_patch_cv(
        c_x,
        c_y,
        bb_width,
        bb_height,
        patch_width,
        patch_height,
        scale=1.0,
        rot=rot,
        inv=False,
    )

    crop_img = cv2.warpAffine(
        img,
        trans,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return crop_img


def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(center, scale, crop_scale_factor, axis="all"):
    """
    center: bbox center [x,y]
    scale: bbox height / 200
    crop_scale_factor: amount of cropping to be applied
    axis: axis which cropping will be applied
        "x": center the y axis and get random crops in x
        "y": center the x axis and get random crops in y
        "all": randomly crop from all locations
    """
    orig_size = int(scale * 200.0)
    ul = (center - (orig_size / 2.0)).astype(int)

    crop_size = int(orig_size * crop_scale_factor)

    if axis == "all":
        h_start = np.random.rand()
        w_start = np.random.rand()
    elif axis == "x":
        h_start = np.random.rand()
        w_start = 0.5
    elif axis == "y":
        h_start = 0.5
        w_start = np.random.rand()
    else:
        raise ValueError(f"axis {axis} is undefined!")

    x1, y1, x2, y2 = get_random_crop_coords(
        height=orig_size,
        width=orig_size,
        crop_height=crop_size,
        crop_width=crop_size,
        h_start=h_start,
        w_start=w_start,
    )
    scale = (y2 - y1) / 200.0
    center = ul + np.array([(y1 + y2) / 2, (x1 + x2) / 2])
    return center, scale


def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = resize(
        img, crop_shape
    )  # , interp='nearest') # scipy.misc.imresize(img, crop_shape, interp='nearest')
    new_img[old_y[0] : old_y[1], old_x[0] : old_x[1]] = img[
        new_y[0] : new_y[1], new_x[0] : new_x[1]
    ]
    return new_img


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img


def flip_kp(kp):
    """Flip keypoints."""
    # flipped_parts = constants.J49_FLIP_PERM
    # kp = kp[flipped_parts]
    kp[:, 0] = -kp[:, 0]
    return kp


def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    raise NotImplementedError
    # flipped_parts = constants.SMPL_POSE_FLIP_PERM
    # pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def denormalize_images(images):
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(
        1, 3, 1, 1
    )
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(
        1, 3, 1, 1
    )
    return images


def read_img(img_fn, dummy_shape):
    try:
        cv_img = _read_img(img_fn)
    except:
        logger.warning(f"Unable to load {img_fn}")
        cv_img = np.zeros(dummy_shape, dtype=np.float32)
        return cv_img, False
    return cv_img, True


def _read_img(img_fn):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def normalize_kp2d_np(kp2d: np.ndarray, img_res):
    assert kp2d.shape[1] == 3
    kp2d_normalized = kp2d.copy()
    kp2d_normalized[:, :2] = 2.0 * kp2d[:, :2] / img_res - 1.0
    return kp2d_normalized


def unnormalize_2d_kp(kp_2d_np: np.ndarray, res):
    assert kp_2d_np.shape[1] == 3
    kp_2d = np.copy(kp_2d_np)
    kp_2d[:, :2] = 0.5 * res * (kp_2d[:, :2] + 1)
    return kp_2d


def normalize_kp2d(kp2d: torch.Tensor, img_res):
    assert len(kp2d.shape) == 3
    kp2d_normalized = kp2d.clone()
    kp2d_normalized[:, :, :2] = 2.0 * kp2d[:, :, :2] / img_res - 1.0
    return kp2d_normalized


def unormalize_kp2d(kp2d_normalized: torch.Tensor, img_res):
    assert len(kp2d_normalized.shape) == 3
    assert kp2d_normalized.shape[2] == 2
    kp2d = kp2d_normalized.clone()
    kp2d = 0.5 * img_res * (kp2d + 1)
    return kp2d


def image_dim_to_bbox(width, height):
    assert isinstance(width, (float, int))
    assert isinstance(height, (float, int))
    """
    0----1
    |    |
    2----3
    """
    x0 = 0.0
    y0 = 0

    x1 = width - 1.0
    y1 = 0

    x2 = 0
    y2 = height - 1.0

    x3 = width - 1.0
    y3 = height - 1.0

    bbox = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
    return bbox


def crop_kp2d(kp2d, cx, cy, dim):
    assert isinstance(kp2d, np.ndarray)
    assert len(kp2d.shape) == 2
    kp2d_cropped = np.copy(kp2d)
    kp2d_cropped[:, 0] -= cx - dim / 2
    kp2d_cropped[:, 1] -= cy - dim / 2
    return kp2d_cropped


def get_wp_intrix(fixed_focal: float, img_res):
    # consruct weak perspective on patch
    camera_center = np.array([img_res // 2, img_res // 2])
    intrx = torch.zeros([3, 3])
    intrx[0, 0] = fixed_focal
    intrx[1, 1] = fixed_focal
    intrx[2, 2] = 1.0
    intrx[0, -1] = camera_center[0]
    intrx[1, -1] = camera_center[1]
    return intrx


def get_aug_intrix(
    intrx, fixed_focal: float, img_res, use_gt_k, bbox_cx, bbox_cy, scale
):

    """
    This function returns camera intrinsics under scaling.
    If use_gt_k, the GT K is used, but scaled based on the amount of scaling in the patch.
    Else, we construct an intrinsic camera with a fixed focal length and fixed camera center.
    """

    if not use_gt_k:
        # consruct weak perspective on patch
        intrx = get_wp_intrix(fixed_focal, img_res)
    else:
        # update the GT intrinsics (full image space)
        # such that it matches the scale of the patch

        dim = scale * 200.0  # bbox size
        k_scale = float(img_res) / dim  # resized_dim / bbox_size in full image space
        """
        # x1 and y1: top-left corner of bbox
        intrinsics after data augmentation
        fx' = k*fx
        fy' = k*fy
        cx' = k*(cx - x1)
        cy' = k*(cy - y1)
        """
        intrx[0, 0] *= k_scale  # k*fx
        intrx[1, 1] *= k_scale  # k*fy
        intrx[0, 2] -= bbox_cx - dim / 2.0
        intrx[1, 2] -= bbox_cy - dim / 2.0
        intrx[0, 2] *= k_scale
        intrx[1, 2] *= k_scale
    return intrx


def compute_joint_valid(
    joints2d,
    speedup,
    is_egocam,
    full_width,
    full_height,
    cx_loose,
    cy_loose,
    dim_loose,
    center,
    bbox_dim,
    augm_dict,
    img_res,
):
    """
    A joint is valid if it is visible (even after data augmentation).
    If a pixel is dark where a joint is, the joint is not valid.
    """
    # create bbox that matches the four corners of the original image
    bbox_full = image_dim_to_bbox(full_width, full_height)
    if speedup and not is_egocam:
        # if input image is loosely croped, need to move the origin
        bbox_full = crop_kp2d(bbox_full, cx_loose, cy_loose, dim_loose)

        # this is the bbox used to pre-crop the image
        loose_bbox2d = np.array(
            [
                [cx_loose - dim_loose / 2.0, cy_loose - dim_loose / 2.0],
                [cx_loose + dim_loose / 2.0, cy_loose - dim_loose / 2.0],
                [cx_loose - dim_loose / 2.0, cy_loose + dim_loose / 2.0],
                [cx_loose + dim_loose / 2.0, cy_loose + dim_loose / 2.0],
            ],
            dtype=np.float32,
        )
        loose_bbox2d = crop_kp2d(loose_bbox2d, cx_loose, cy_loose, dim_loose)

        loose_bbox2d_pad = np.ones((loose_bbox2d.shape[0], 3))
        loose_bbox2d_pad[:, :2] = loose_bbox2d
        loose_bbox2d = j2d_processing(
            loose_bbox2d_pad, center, bbox_dim, augm_dict, img_res
        )

    bbox_full_pad = np.ones((bbox_full.shape[0], 3))
    bbox_full_pad[:, :2] = bbox_full
    bbox_full = j2d_processing(bbox_full_pad, center, bbox_dim, augm_dict, img_res)

    if speedup and not is_egocam:
        # this is the loose bbox to pre-crop
        loose_bbox2d_de = unnormalize_2d_kp(loose_bbox2d, img_res)

        # whether a joint is within the pre-crop bbox
        jts_valid_loose = in_hull(joints2d[:, :2], loose_bbox2d_de[:, :2])

    # this is the bbox of the original image size
    bbox_full_de = unnormalize_2d_kp(bbox_full, img_res)
    jts_valid_bbox = in_hull(joints2d[:, :2], bbox_full_de[:, :2])

    # this is to test if the joint is within the patch (e.g., 224)
    jts_valid_patch = np.prod(
        np.logical_and(0 < joints2d[:, :2], joints2d[:, :2] < img_res), axis=1
    ).astype(bool)

    jts_valid = jts_valid_bbox * jts_valid_patch
    if speedup and not is_egocam:
        jts_valid *= jts_valid_loose
    return jts_valid
