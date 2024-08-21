import math

import cv2
import numpy as np
import torch

from common.torch_utils import all_comb


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction="mean"):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)

    re_per_joint = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1))
    re = re_per_joint.mean(axis=-1)
    if reduction == "mean":
        re = re.mean()
    elif reduction == "sum":
        re = re.sum()
    return re, re_per_joint


def compute_pck_mano(gt_dist, pred_dist, valid, dummy, alpha):
    is_valid = np.prod(valid, axis=1).astype(bool)

    diff = np.absolute(gt_dist - pred_dist)

    pck = (diff < alpha).astype(np.float32)  # 2cm

    pck_examples = pck.sum(axis=1)

    num_verts = diff.shape[1]

    pck_examples /= num_verts

    pck_examples *= 100
    pck_examples[np.logical_not(is_valid)] = float("nan")

    return pck_examples


def compute_pck_obj(gt_dist, pred_dist, valid, vlen, alpha):
    is_valid = valid.sum(axis=1).astype(np.uint64) == vlen
    diff = np.absolute(gt_dist - pred_dist)

    pck = (diff < alpha).astype(np.float32)  # 2cm

    pck_examples = pck.sum(axis=1)

    pck_examples /= vlen
    pck_examples *= 100
    pck_examples[np.logical_not(is_valid)] = float("nan")

    return pck_examples


def compute_v2v_dist_no_reduce(v3d_cam_gt, v3d_cam_pred, is_valid):
    assert isinstance(v3d_cam_gt, list)
    assert isinstance(v3d_cam_pred, list)
    assert len(v3d_cam_gt) == len(v3d_cam_pred)
    assert len(v3d_cam_gt) == len(is_valid)
    v2v = []
    for v_gt, v_pred, valid in zip(v3d_cam_gt, v3d_cam_pred, is_valid):
        if valid:
            dist = ((v_gt - v_pred) ** 2).sum(dim=1).sqrt().cpu().numpy()  # meter
        else:
            dist = None
        v2v.append(dist)
    return v2v


def compute_v2v_dist(v3d_cam_gt, v3d_cam_pred, is_valid):
    assert isinstance(v3d_cam_gt, list)
    assert isinstance(v3d_cam_pred, list)
    assert len(v3d_cam_gt) == len(v3d_cam_pred)
    assert len(v3d_cam_gt) == len(is_valid)
    v2v = []
    for v_gt, v_pred, valid in zip(v3d_cam_gt, v3d_cam_pred, is_valid):
        if valid:
            dist = float(((v_gt - v_pred) ** 2).sum(dim=1).sqrt().mean())  # meter
        else:
            dist = float("nan")
        v2v.append(dist)
    v2v = np.array(v2v)
    return v2v


def compute_diameter(v3d, num_samples):
    np.random.seed(1)
    if num_samples is not None:
        rand_idx = np.random.permutation(v3d.shape[0])[:num_samples]
        pts = v3d[rand_idx]
    else:
        pts = v3d
    pts_comb = all_comb(pts, pts)
    dist = ((pts_comb[:, :3] - pts_comb[:, 3:]) ** 2).sum(dim=1).sqrt()
    diameter = dist.max()
    return float(diameter)


def compute_joint3d_error(joints3d_cam_gt, joints3d_cam_pred, valid_jts):
    valid_jts = valid_jts.view(-1)
    assert joints3d_cam_gt.shape == joints3d_cam_pred.shape
    assert joints3d_cam_gt.shape[0] == valid_jts.shape[0]
    dist = ((joints3d_cam_gt - joints3d_cam_pred) ** 2).sum(dim=2).sqrt()
    invalid_idx = torch.nonzero((1 - valid_jts).long()).view(-1)
    dist[invalid_idx, :] = float("nan")
    dist = dist.cpu().numpy()
    return dist


def compute_joint2d_error(joints2d_gt, joints2d_pred, valid_jts, img_res):
    dist = ((joints2d_gt - joints2d_pred) ** 2).sum(dim=2).sqrt()
    valid_idx = torch.nonzero(valid_jts.long()).view(-1)
    # percentage of the patch
    dist = dist[valid_idx].cpu().numpy() / img_res * 100
    return dist


def compute_mrrpe(root_r_gt, root_l_gt, root_r_pred, root_l_pred, is_valid):
    rel_vec_gt = root_l_gt - root_r_gt
    rel_vec_pred = root_l_pred - root_r_pred

    invalid_idx = torch.nonzero((1 - is_valid).long()).view(-1)
    mrrpe = ((rel_vec_pred - rel_vec_gt) ** 2).sum(dim=1).sqrt()
    mrrpe[invalid_idx] = float("nan")
    mrrpe = mrrpe.cpu().numpy()
    return mrrpe


def compute_iou_metrics(
    bbox3d_cam_gt_top,
    bbox3d_cam_pred_top,
    bbox3d_cam_gt_bottom,
    bbox3d_cam_pred_bottom,
    is_valid,
):
    iou_top = []
    for bbox_gt, bbox_pred, curr_valid in zip(
        bbox3d_cam_gt_top, bbox3d_cam_pred_top, is_valid
    ):
        if int(curr_valid) == 0:
            iou_top.append(float("nan"))
        else:
            iou = iou_3d_torch(bbox_gt * 1000, bbox_pred * 1000, "cuda:0", 100)
            iou_top.append(iou)
    iou_top = np.array(iou_top)

    iou_bottom = []
    for bbox_gt, bbox_pred, curr_valid in zip(
        bbox3d_cam_gt_bottom, bbox3d_cam_pred_bottom, is_valid
    ):
        if int(curr_valid) == 0:
            iou_bottom.append(float("nan"))
        else:
            iou = iou_3d_torch(bbox_gt * 1000, bbox_pred * 1000, "cuda:0", 100)
            iou_bottom.append(iou)
    iou_bottom = np.array(iou_bottom)
    return iou_top, iou_bottom


def compute_arti_deg_error(pred_radian, gt_radian):
    assert pred_radian.shape == gt_radian.shape

    # articulation error in degree
    pred_degree = pred_radian / math.pi * 180  # degree
    gt_degree = gt_radian / math.pi * 180  # degree
    err_deg = torch.abs(pred_degree - gt_degree).tolist()
    return np.array(err_deg, dtype=np.float32)


def pts_inside_box_torch(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)

    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - bbox[4, :].view(1, 3)
    p1 = torch.matmul(up, u1.view((3, 1)))
    p2 = torch.matmul(up, u2.view((3, 1)))
    p3 = torch.matmul(up, u3.view((3, 1)))

    p1 = torch.logical_and(p1 > 0, p1 < torch.dot(u1, u1))
    p2 = torch.logical_and(p2 > 0, p2 < torch.dot(u2, u2))
    p3 = torch.logical_and(p3 > 0, p3 < torch.dot(u3, u3))
    return torch.logical_and(torch.logical_and(p1, p2), p3)


def iou_3d_torch(bbox1, bbox2, dev, nres):

    """
    This function compute the 3D IoU between two bounding boxes.
    Bounding box corners should be defined in this order:
            2 -------- 1
           /|         /|
          3 -------- 0 .
          | |        | |
          . 6 -------- 5
          |/         |/
          7 -------- 4
    """
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert bbox1.shape == bbox2.shape
    assert bbox1.shape == (8, 3)

    # bbox that wraps both boxes
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)

    # uniform sample in the space
    xs = torch.linspace(bmin[0], bmax[0], nres).to(dev)
    ys = torch.linspace(bmin[1], bmax[1], nres).to(dev)
    zs = torch.linspace(bmin[2], bmax[2], nres).to(dev)
    pts = torch.meshgrid(xs, ys, zs)
    pts = torch.stack(pts, dim=-1).view(-1, 3)  # (N, 3)
    bbox1_tensor = torch.FloatTensor(bbox1).to(dev)
    bbox2_tensor = torch.FloatTensor(bbox2).to(dev)

    # num points inside a box
    flag1 = pts_inside_box_torch(pts, bbox1_tensor)
    flag2 = pts_inside_box_torch(pts, bbox2_tensor)
    intersect = torch.sum(torch.logical_and(flag1, flag2))
    union = torch.sum(torch.logical_or(flag1, flag2))

    xs = xs.cpu()
    ys = ys.cpu()
    zs = zs.cpu()
    bbox1_tensor = bbox1_tensor.cpu()
    bbox2_tensor = bbox2_tensor.cpu()
    if union == 0:
        # did not sample within the union box
        return float("nan")
    else:
        return float(intersect) / float(union)


def segm_iou(pred, target, n_classes, tol, background_cls=0):
    """
    Compute segmentation iou for a segmentation mask.
    pred: (dim, dim)
    target: (dim, dim)
    n_classes: including the background class
    tol: how many entries to ignore for union for noisy target map
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert pred.shape == target.shape
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        if cls == background_cls:
            continue
        pred_cls = pred == cls
        target_cls = target == cls
        target_inds = target_cls.nonzero(as_tuple=True)[0]
        intersection = pred_cls[target_inds].long().sum()
        union = pred_cls.long().sum() + target_cls.long().sum() - intersection

        if union < tol:
            # If there is no ground truth, do not include in evaluation
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return float(np.nanmean(np.array(ious)))


SMPL_OR_JOINTS = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])


def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 24, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 24, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]
    pred_mat = pred_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.mean(np.array(angles))
