import numpy as np
import cv2
import pickle
from collections import namedtuple
from scipy import interpolate
from vlfeat.utils import rgb2gray
from cyvlfeat.sift.phow import phow
from utils.load_WUSTL_transformation import load_transformation
LocResult = namedtuple(
    'LocResult', ['success', 'num_inliers', 'inlier_ratio', 'T'])
loc_failure = LocResult(False, 0, 0, None)

def do_pnp(kpts, lms, query_K, query_dist, reproj_error=10, min_inliers=3):
    kpts = kpts.astype(np.float32).reshape((-1, 1, 2))
    lms = lms.astype(np.float32).reshape((-1, 1, 3))
    query_K = np.array(query_K)

    success, R_vec, t, inliers = cv2.solvePnPRansac(
        lms, kpts, query_K, np.array([query_dist, 0, 0, 0]),
        iterationsCount=5000, reprojectionError=reproj_error,
        flags=cv2.SOLVEPNP_P3P)

    if success:
        inliers = inliers[:, 0]
        num_inliers = len(inliers)
        inlier_ratio = len(inliers) / len(kpts)
        success &= num_inliers >= min_inliers

        ret, R_vec, t = cv2.solvePnP(
                lms[inliers], kpts[inliers], query_K,
                np.array([query_dist, 0, 0, 0]), rvec=R_vec, tvec=t,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ret

        query_T_w = np.eye(4)
        query_T_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        query_T_w[:3, 3] = t[:, 0]
        w_T_query = np.linalg.inv(query_T_w)
        # w_T_query = query_T_w

        ret = LocResult(success, num_inliers, inlier_ratio, w_T_query)
    else:
        inliers = np.empty((0,), np.int32)
        ret = loc_failure

    return ret, inliers

def proj_point2img(rgb, xyz, camera_parm, pose):

    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])


    proj_rgb = np.empty((H, W, 3))
    proj_rgb[:] = np.nan

    proj_xyz = np.empty((H, W, 3))
    proj_xyz[:] = np.nan

    proj_mat = np.matmul(camera_parm, pose[:3, :])
    H_xyz = np.concatenate((xyz.T, np.ones((1, len(xyz)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)


    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    rgb = rgb[front_idx, :].T
    xyz = xyz[front_idx, :].T
    uv_norm = uv_norm[front_idx]

    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    rgb = rgb[:, visible_idx]
    xyz = xyz[:, visible_idx]
    uv_norm = uv_norm[visible_idx]


    uv = np.floor(uv).astype(np.uint16)

    sort = np.argsort(uv_norm)[::-1]
    proj_rgb[uv[1, sort], uv[0, sort], :] = rgb[:, sort].T
    proj_xyz[uv[1, sort], uv[0, sort], :] = xyz[:, sort].T

    # check_norm = np.ones((H, W)) * uv_norm.max()
    # for idx in range(len(uv.T)):
    #     if uv_norm[idx] <= check_norm[uv[1, idx], uv[0, idx]]:
    #         proj_rgb[uv[1, idx], uv[0, idx], 0] = rgb[0, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 1] = rgb[1, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 2] = rgb[2, idx]
    #
    #         proj_xyz[uv[1, idx], uv[0, idx], 0] = xyz[0, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 1] = xyz[1, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 2] = xyz[2, idx]
    #
    #         check_norm[uv[1, idx], uv[0, idx]] = uv_norm[idx]

    proj_rgb = proj_rgb.astype(np.uint8)

    return proj_rgb, proj_xyz


def proj_point2img_v2(main_rgb, main_xyz, sub_rgb, sub_xyz, camera_parm, pose):

    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])


    proj_rgb = np.empty((H, W, 3))
    proj_rgb[:] = np.nan

    proj_xyz = np.empty((H, W, 3))
    proj_xyz[:] = np.nan

    proj_mat = np.matmul(camera_parm, pose[:3, :])

    """Sub pointclouds projection"""

    H_xyz = np.concatenate((sub_xyz.T, np.ones((1, len(sub_xyz)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)


    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    rgb = sub_rgb[front_idx, :].T
    xyz = sub_xyz[front_idx, :].T
    uv_norm = uv_norm[front_idx]

    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    rgb = rgb[:, visible_idx]
    xyz = xyz[:, visible_idx]
    uv_norm = uv_norm[visible_idx]


    uv = np.floor(uv).astype(np.uint16)

    sort = np.argsort(uv_norm)[::-1]
    proj_rgb[uv[1, sort], uv[0, sort], :] = rgb[:, sort].T
    proj_xyz[uv[1, sort], uv[0, sort], :] = xyz[:, sort].T


    # check_norm = np.ones((H, W)) * uv_norm.max()
    # for idx in range(len(uv.T)):
    #     if uv_norm[idx] <= check_norm[uv[1, idx], uv[0, idx]]:
    #         proj_rgb[uv[1, idx], uv[0, idx], 0] = rgb[0, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 1] = rgb[1, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 2] = rgb[2, idx]
    #
    #         proj_xyz[uv[1, idx], uv[0, idx], 0] = xyz[0, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 1] = xyz[1, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 2] = xyz[2, idx]
    #
    #         check_norm[uv[1, idx], uv[0, idx]] = uv_norm[idx]

    proj_rgb = proj_rgb.astype(np.uint8)


    """Main pointclouds projection"""

    H_xyz = np.concatenate((main_xyz.T, np.ones((1, len(main_xyz)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    rgb = main_rgb[front_idx, :].T
    xyz = main_xyz[front_idx, :].T
    uv_norm = uv_norm[front_idx]

    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1, :] < H)
    uv = uv[:, visible_idx]
    rgb = rgb[:, visible_idx]
    xyz = xyz[:, visible_idx]
    uv_norm = uv_norm[visible_idx]

    uv = np.floor(uv).astype(np.uint16)

    sort = np.argsort(uv_norm)[::-1]
    proj_rgb[uv[1, sort], uv[0, sort], :] = rgb[:,sort].T
    proj_xyz[uv[1, sort], uv[0, sort], :] = xyz[:,sort].T
    #
    # check_norm = np.ones((H, W)) * uv_norm.max()
    # for idx in range(len(uv.T)):
    #     if uv_norm[idx] <= check_norm[uv[1, idx], uv[0, idx]]:
    #         proj_rgb[uv[1, idx], uv[0, idx], 0] = rgb[0, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 1] = rgb[1, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 2] = rgb[2, idx]
    #
    #         proj_xyz[uv[1, idx], uv[0, idx], 0] = xyz[0, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 1] = xyz[1, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 2] = xyz[2, idx]
    #
    #         check_norm[uv[1, idx], uv[0, idx]] = uv_norm[idx]

    proj_rgb = proj_rgb.astype(np.uint8)

    return proj_rgb, proj_xyz

def seq_proj_point2img(vis_scan_idx, align_list, scan_list, camera_parm, pose):


    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])

    proj_mat = np.matmul(camera_parm, pose[:3, :])


    proj_rgb = np.empty((H, W, 3))
    proj_rgb[:] = np.nan

    proj_xyz = np.empty((H, W, 3))
    proj_xyz[:] = np.nan


    check_norm = np.ones((H, W)) * np.inf
    for scan_idx in vis_scan_idx[1:]:
        align_pth = align_list[scan_idx]
        _, P_after = load_transformation(align_pth)

        scan_pth = scan_list[scan_idx]
        scan_mat = np.load(scan_pth)
        scan_xyz = scan_mat[:, :3]

        H_xyz = np.concatenate((scan_xyz.T, np.ones((1, len(scan_xyz)))), axis=0)
        aligned_xyz = np.matmul(P_after, H_xyz)

        xyz = np.divide(aligned_xyz[:3, :], aligned_xyz[3, :]).T
        rgb = scan_mat[:, 3:]

        uv = np.matmul(proj_mat, aligned_xyz)
        uv_norm = uv[2, :]
        uv = np.divide(uv[:2, :], uv_norm)


        front_idx = uv_norm > 0
        uv = uv[:, front_idx]
        rgb = rgb[front_idx, :].T
        xyz = xyz[front_idx, :].T
        uv_norm = uv_norm[front_idx]

        visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
        uv = uv[:, visible_idx]
        rgb = rgb[:, visible_idx]
        xyz = xyz[:, visible_idx]
        uv_norm = uv_norm[visible_idx]

        uv = np.floor(uv).astype(np.uint16)



        sort = np.argsort(uv_norm)[::-1]
        proj_rgb[uv[1, sort], uv[0, sort], :] = rgb[:, sort].T
        proj_xyz[uv[1, sort], uv[0, sort], :] = xyz[:, sort].T
        # for idx in range(len(uv.T)):
        #     if uv_norm[idx] <= check_norm[uv[1, idx], uv[0, idx]]:
        #         proj_rgb[uv[1, idx], uv[0, idx], 0] = rgb[0, idx]
        #         proj_rgb[uv[1, idx], uv[0, idx], 1] = rgb[1, idx]
        #         proj_rgb[uv[1, idx], uv[0, idx], 2] = rgb[2, idx]
        #
        #         proj_xyz[uv[1, idx], uv[0, idx], 0] = xyz[0, idx]
        #         proj_xyz[uv[1, idx], uv[0, idx], 1] = xyz[1, idx]
        #         proj_xyz[uv[1, idx], uv[0, idx], 2] = xyz[2, idx]
        #
        #         check_norm[uv[1, idx], uv[0, idx]] = uv_norm[idx]

    """Main pointclouds projection"""

    main_idx = vis_scan_idx[0]

    align_pth = align_list[main_idx]
    _, P_after = load_transformation(align_pth)

    scan_pth = scan_list[main_idx]
    scan_mat = np.load(scan_pth)
    scan_xyz = scan_mat[:, :3]

    H_xyz = np.concatenate((scan_xyz.T, np.ones((1, len(scan_xyz)))), axis=0)
    aligned_xyz = np.matmul(P_after, H_xyz)

    xyz = np.divide(aligned_xyz[:3, :], aligned_xyz[3, :]).T
    rgb = scan_mat[:, 3:]

    uv = np.matmul(proj_mat, aligned_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    rgb = rgb[front_idx, :].T
    xyz = xyz[front_idx, :].T
    uv_norm = uv_norm[front_idx]

    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1, :] < H)
    uv = uv[:, visible_idx]
    rgb = rgb[:, visible_idx]
    xyz = xyz[:, visible_idx]
    uv_norm = uv_norm[visible_idx]

    uv = np.floor(uv).astype(np.uint16)

    sort = np.argsort(uv_norm)[::-1]
    proj_rgb[uv[1, sort], uv[0, sort], :] = rgb[:, sort].T
    proj_xyz[uv[1, sort], uv[0, sort], :] = xyz[:, sort].T


    # check_norm = np.ones((H, W)) * uv_norm.max()
    # for idx in range(len(uv.T)):
    #     if uv_norm[idx] <= check_norm[uv[1, idx], uv[0, idx]]:
    #         proj_rgb[uv[1, idx], uv[0, idx], 0] = rgb[0, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 1] = rgb[1, idx]
    #         proj_rgb[uv[1, idx], uv[0, idx], 2] = rgb[2, idx]
    #
    #         proj_xyz[uv[1, idx], uv[0, idx], 0] = xyz[0, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 1] = xyz[1, idx]
    #         proj_xyz[uv[1, idx], uv[0, idx], 2] = xyz[2, idx]
    #
    #         check_norm[uv[1, idx], uv[0, idx]] = uv_norm[idx]

    proj_rgb = proj_rgb.astype(np.uint8)


    return proj_rgb, proj_xyz


def compute_errormap(q_img, syn_img, syn_xyz):
    # rgb_flag = np.all(syn_xyz, axis=2)
    rgb_flag = np.all(~np.isnan(syn_xyz), axis=2)

    # if np.any(rgb_flag):
    if (rgb_flag.sum()/rgb_flag.size) > 0.05:
        # gray_img = np.array(Image.fromarray(q_img).convert("L"), dtype=np.double)
        # I_q = image_normalization(gray_img, rgb_flag)
        #
        # gray_syn = np.array(Image.fromarray(syn_img).convert("L"), dtype=np.double)
        # gray_syn[~rgb_flag] = np.nan
        # I_syn = image_normalization(inpaint_nans(gray_syn), rgb_flag)
        #
        # fq, dq = vl_phow(I_q, sizes=8, step=4)

        gray_img = rgb2gray(q_img)
        I_q = image_normalization(gray_img, rgb_flag)

        gray_syn = rgb2gray(syn_img)
        gray_syn[~rgb_flag] = np.nan
        I_syn = image_normalization(inpaint_nans(gray_syn), rgb_flag)


        # f_q, d_q = phow(I_q, step=4, sizes=(4, 6, 8, 10))
        # f_syn, d_syn = phow(I_syn, step=4, sizes=(4, 6, 8, 10))
        f_q, d_q = phow(I_q, step=4, sizes=(8, 8))
        f_syn, d_syn = phow(I_syn, step=4, sizes=(8, 8))


        f_linind = sub2ind(I_syn.shape, f_syn[:, 0], f_syn[:, 1])
        reshaped_flag = np.reshape(rgb_flag, -1)
        iseval = reshaped_flag[f_linind]
        # iseval = rgb_flag[f_syn[:,0].astype(np.int), f_syn[:,1].astype(np.int)]

        d_q = rootsift(d_q)
        d_syn = rootsift(d_syn)

        """ Compute error"""
        err = np.sqrt(np.sum(np.square(d_q[iseval,:] - d_syn[iseval, :]), 1))
        score = np.divide(1, np.quantile(err, 0.5))

        errmap = np.empty_like(I_syn)
        errmap[:] = np.nan

        errmap[f_syn[iseval, 0].astype(np.int), f_syn[iseval, 1].astype(np.int)] = err

        xuni = np.sort(np.unique(f_syn[:, 0]).astype(np.int))
        yuni = np.sort(np.unique(f_syn[:, 1]).astype(np.int))

        """Can modified..."""
        vis_errmap = np.empty([len(xuni), len(yuni)])
        vis_errmap[:] = np.nan
        for ni, i in enumerate(xuni):
            for nj, j in enumerate(yuni):
                vis_errmap[ni, nj] = errmap[i, j]
    else:
        score = 0
        vis_errmap = np.zeros([200, 200])

    return score, vis_errmap



def jumbak2(ori, ks=None, erode_first=True):
    if ks is None:
        ks = [7, 7]
    rest = 0 if erode_first else 1

    for i, k in enumerate(ks):
        if i%2 == rest:
            kernel1 = np.ones((k, k), np.uint8)
            ero = cv2.erode(ori, kernel1, iterations=1)
            dil = cv2.dilate(ero, kernel1, iterations=1)
            ori = dil
        else:
            kernel1 = np.ones((k, k), np.uint8)
            dil = cv2.dilate(ori, kernel1, iterations=1)
            ero = cv2.erode(dil, kernel1, iterations=1)
            ori = ero
    return ori


def compute_errormap_morph_mavg_v2(q_img, syn_img, syn_xyz):
    # rgb_flag = np.all(syn_xyz, axis=2)
    rgb_flag1 = np.all(~np.isnan(syn_xyz), axis=2)
    rgb_flag = jumbak2(rgb_flag1.astype(np.uint8))
    rgb_flag = rgb_flag.astype(bool)
    # if np.any(rgb_flag):
    if (rgb_flag.sum()/rgb_flag.size) > 0.05:

        # image inpainting with nearest neighbor
        dif = rgb_flag1.astype(np.int16) - rgb_flag
        fm = np.argwhere(dif == -1)  # where to fill in
        om = np.argwhere(rgb_flag1)

        # detour memory error in multiprocessing
        sep = 15
        num_each = om.shape[0]//sep
        tot = fm.shape[0]
        idx = np.arange(tot)
        arg_cat = np.zeros([sep, tot])
        dist_cat = np.zeros([sep, tot])
        for i in np.arange(sep):
            om_i = om[num_each*i : num_each*(i+1), :]
            # dist = fm[None, :, :] - om_i[:, None, :]
            dist = np.linalg.norm(fm[None, :, :] - om_i[:, None, :], axis=2)
            arg_cat[i, :] = np.argmin(dist, axis=0)
            dist_cat[i, :] = dist[arg_cat[i, :].astype(np.uint32), idx]
        if om.shape[0]%sep >0: # last one
            om_i = om[num_each * (i+1):, :]
            dist = np.linalg.norm(fm[None, :, :] - om_i[:, None, :], axis=2)
            arg_cat[i, :] = np.argmin(dist, axis=0)
            dist_cat[i, :] = dist[arg_cat[i, :].astype(np.uint32), idx]

        # fw = np.argmin(np.linalg.norm(dist_cat, axis=2), axis=0)  # fill with idx
        fw = np.argmin(dist_cat, axis=0)  # fill with idx
        fw = arg_cat[fw, idx].astype(np.uint32)
        syn_img[fm[:, 0], fm[:, 1]] = syn_img[om[fw][:, 0], om[fw][:, 1]]
            # fw = np.argmin(np.linalg.norm(dist, axis=2), axis=0)  # fill with idx
            # syn_img[fm[:, 0], fm[:, 1]] = syn_img[om_i[fw][:, 0], om_i[fw][:, 1]]


        gray_img = rgb2gray(q_img)
        I_q = image_normalization(gray_img, rgb_flag)

        gray_syn = rgb2gray(syn_img)
        gray_syn[~rgb_flag] = np.nan
        I_syn = image_normalization(inpaint_nans(gray_syn), rgb_flag)


        # f_q, d_q = phow(I_q, step=4, sizes=(4, 6, 8, 10))
        # f_syn, d_syn = phow(I_syn, step=4, sizes=(4, 6, 8, 10))

        f_q, d_q = phow(I_q, step=4, sizes=(8, 8))
        f_syn, d_syn = phow(I_syn, step=4, sizes=(8, 8))
        # f_q, d_q = phow(I_q, step=1, sizes=(8, 16))
        # f_syn, d_syn = phow(I_syn, step=1, sizes=(8, 16))


        f_linind = sub2ind(I_syn.shape, f_syn[:, 0], f_syn[:, 1])
        reshaped_flag = np.reshape(rgb_flag, -1)
        iseval = reshaped_flag[f_linind]
        # iseval = rgb_flag[f_syn[:,0].astype(np.int), f_syn[:,1].astype(np.int)]

        d_q = rootsift(d_q)
        d_syn = rootsift(d_syn)

        """ Compute error"""
        err = np.sqrt(np.sum(np.square(d_q[iseval,:] - d_syn[iseval, :]), 1))

        median_err = np.quantile(err, 0.5)
        score_idx = err<median_err
        score = np.divide(1, np.average(err[score_idx]))
        # score = np.divide(1, np.quantile(err, 0.5))

        errmap = np.empty_like(I_syn)
        errmap[:] = np.nan

        errmap[f_syn[iseval, 0].astype(np.int), f_syn[iseval, 1].astype(np.int)] = err

        xuni = np.sort(np.unique(f_syn[:, 0]).astype(np.int))
        yuni = np.sort(np.unique(f_syn[:, 1]).astype(np.int))

        """Can modified..."""
        vis_errmap = np.empty([len(xuni), len(yuni)])
        vis_errmap[:] = np.nan
        for ni, i in enumerate(xuni):
            for nj, j in enumerate(yuni):
                vis_errmap[ni, nj] = errmap[i, j]
    else:
        score = 0
        vis_errmap = np.zeros([200, 200])

    return score, vis_errmap


def image_normalization(gray_img, rgb_flag):

    img_channel = np.reshape(gray_img, -1)
    img_flag_channel = np.reshape(rgb_flag, -1)

    img_mean = np.mean(img_channel[img_flag_channel])

    img_centered = img_channel - img_mean
    img_centered[~img_flag_channel] = 0

    img_std = np.std(img_centered[img_flag_channel], ddof=1, axis=0)
    img_normalized = np.divide(img_centered, img_std)
    img_normalized = np.reshape(img_normalized, [gray_img.shape[0], gray_img.shape[1]])

    return img_normalized

def inpaint_nans(img):

    valid_mask = ~np.isnan(img)
    coords = np.array(np.nonzero(valid_mask)).T
    values = img[valid_mask]

    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(img.shape))).reshape(img.shape)

    # mask = np.isnan(img)
    # filled = inpaint.inpaint_biharmonic(img, mask)

    return filled

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind.astype(int)

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def rootsift(x):
    x = np.array(x, dtype=np.single)
    l1_norm = np.sqrt(np.divide(x, np.sum(np.abs(x), 0)+(1e-12)))
    return l1_norm

def scan2imgfeat_projection(pose, feat_pth, camera_parm, num_kpts=4096):

    with open(feat_pth[0], 'rb') as handle:
        scan_feat = pickle.load(handle)
        scan_pts = scan_feat['ptcloud']
        scan_desc = scan_feat['descriptors']
        scan_score = scan_feat['scores']

    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])


    proj_mat = np.matmul(camera_parm, pose[:3, :])
    H_xyz = np.concatenate((scan_pts.T, np.ones((1, len(scan_pts)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    desc = scan_desc[:, front_idx]
    score = scan_score[front_idx]
    ptcloud = scan_pts[front_idx, :]


    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    desc = desc[:, visible_idx]
    score = score[visible_idx]
    ptcloud = ptcloud[visible_idx, :]

    score_idx = np.argsort(score)[::-1][:num_kpts]
    uv = uv[:, score_idx]
    desc = desc[:, score_idx]
    score = score[score_idx]
    ptcloud = ptcloud[score_idx, :]

    kpts = np.floor(uv.T).astype(np.float32)

    return kpts, desc, score, ptcloud


def scan2imgfeat_projection_normal(pose, feat_pth, camera_parm, num_kpts=4096):

    with open(feat_pth[0], 'rb') as handle:
        scan_feat = pickle.load(handle)
        scan_pts = scan_feat['ptcloud']
        scan_desc = scan_feat['descriptors']
        scan_score = scan_feat['scores']
        scan_normal = scan_feat['normal']

    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])


    proj_mat = np.matmul(camera_parm, pose[:3, :])
    H_xyz = np.concatenate((scan_pts.T, np.ones((1, len(scan_pts)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    desc = scan_desc[:, front_idx]
    score = scan_score[front_idx]
    ptcloud = scan_pts[front_idx, :]
    normal = scan_normal[front_idx, :]

    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    desc = desc[:, visible_idx]
    score = score[visible_idx]
    ptcloud = ptcloud[visible_idx, :]
    normal = normal[visible_idx, :]

    c2p_vec = ptcloud - np.linalg.inv(pose)[:3,3]
    c2p_vec = c2p_vec / np.expand_dims(np.linalg.norm(c2p_vec, axis=1), -1)

    # hpr_idx = np.sum(c2p_vec*normal, axis=1) < 0.0
    hpr_idx = np.sum(c2p_vec*normal, axis=1) < 0.5
    uv = uv[:, hpr_idx]
    desc = desc[:, hpr_idx]
    score = score[hpr_idx]
    ptcloud = ptcloud[hpr_idx, :]


    score_idx = np.argsort(score)[::-1][:num_kpts]
    uv = uv[:, score_idx]
    desc = desc[:, score_idx]
    score = score[score_idx]
    ptcloud = ptcloud[score_idx, :]

    kpts = np.floor(uv.T).astype(np.float32)

    return kpts, desc, score, ptcloud



def scan2imgfeat_graph_projection(pose, pred_scan_idx, retscan_idx, feat_list, scan_graph, camera_parm):

    scan_graph_idx = scan_graph[pred_scan_idx[0]]

    vis_scan_idx = list()

    vis_scan_idx.append(pred_scan_idx[0])

    for idx in scan_graph_idx:
        if idx in retscan_idx:
            vis_scan_idx.append(idx)


    scan_pts = list()
    scan_desc = list()
    scan_score = list()

    for scan_idx in vis_scan_idx:
        feat_pth = feat_list[scan_idx]

        with open(feat_pth, 'rb') as handle:
            scan_feat = pickle.load(handle)
        scan_pts.append(scan_feat['ptcloud'])
        scan_desc.append(scan_feat['descriptors'])
        scan_score.append(scan_feat['scores'])

    scan_pts = np.concatenate(scan_pts, axis=0)
    scan_desc = np.concatenate(scan_desc, axis=1)
    scan_score = np.concatenate(scan_score)



    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])


    proj_mat = np.matmul(camera_parm, pose[:3, :])
    H_xyz = np.concatenate((scan_pts.T, np.ones((1, len(scan_pts)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    desc = scan_desc[:, front_idx]
    score = scan_score[front_idx]
    ptcloud = scan_pts[front_idx, :]


    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    desc = desc[:, visible_idx]
    score = score[visible_idx]
    ptcloud = ptcloud[visible_idx, :]


    score_idx = np.argsort(score)[::-1][:4096]
    uv = uv[:, score_idx]
    desc = desc[:, score_idx]
    score = score[score_idx]
    ptcloud = ptcloud[score_idx, :]

    kpts = np.floor(uv.T).astype(np.float32)

    return kpts, desc, score, ptcloud


def scan2imgfeat_graph_projection_v2(pose, pred_scan_idx, retscan_idx, feat_list, scan_graph, camera_parm):


    img_size = camera_parm[:2, 2] * 2 # uv coordinate
    H = int(img_size[1])
    W = int(img_size[0])

    """Main Scan feature projection..."""

    main_scan_idx = pred_scan_idx[0]
    main_scan_pth = feat_list[main_scan_idx]

    with open(main_scan_pth, 'rb') as handle:
        scan_feat = pickle.load(handle)
    main_pts = scan_feat['ptcloud']
    main_desc = scan_feat['descriptors']
    main_scores = scan_feat['scores']

    proj_mat = np.matmul(camera_parm, pose[:3, :])
    H_xyz = np.concatenate((main_pts.T, np.ones((1, len(main_pts)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    desc = main_desc[:, front_idx]
    score = main_scores[front_idx]
    ptcloud = main_pts[front_idx, :]


    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    desc = desc[:, visible_idx]
    score = score[visible_idx]
    ptcloud = ptcloud[visible_idx, :]


    score_idx = np.argsort(score)[::-1][:3000]
    uv = uv[:, score_idx]


    main_desc = desc[:, score_idx]
    main_score = score[score_idx]
    main_ptcloud = ptcloud[score_idx, :]
    main_kpts = np.floor(uv.T).astype(np.float32)

    """Sub scan feature projection..."""

    scan_graph_idx = scan_graph[main_scan_idx]
    sub_scan_idx = list()
    for idx in scan_graph_idx:
        if idx in retscan_idx:
            sub_scan_idx.append(idx)

    sub_scan_pts = list()
    sub_scan_desc = list()
    sub_scan_score = list()
    sub_scan_kpts = list()

    for scan_idx in sub_scan_idx:
        sub_feat_pth = feat_list[scan_idx]

        with open(sub_feat_pth, 'rb') as handle:
            sub_scan_feat = pickle.load(handle)
        sub_pts = sub_scan_feat['ptcloud']
        sub_desc = sub_scan_feat['descriptors']
        sub_score = sub_scan_feat['scores']
        H_xyz = np.concatenate((sub_pts.T, np.ones((1, len(sub_pts)))), axis=0)

        uv = np.matmul(proj_mat, H_xyz)
        uv_norm = uv[2, :]
        uv = np.divide(uv[:2, :], uv_norm)

        front_idx = uv_norm > 0
        uv = uv[:, front_idx]
        desc = sub_desc[:, front_idx]
        score = sub_score[front_idx]
        ptcloud = sub_pts[front_idx, :]

        visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1, :] < H)
        uv = uv[:, visible_idx]
        desc = desc[:, visible_idx]
        score = score[visible_idx]
        ptcloud = ptcloud[visible_idx, :]

        score_idx = np.argsort(score)[::-1][:1000]
        uv = uv[:, score_idx]

        sub_scan_desc.append(desc[:, score_idx])
        sub_scan_score.append(score[score_idx])
        sub_scan_pts.append(ptcloud[score_idx, :])
        sub_scan_kpts.append(np.floor(uv.T).astype(np.float32))



    sub_desc = np.concatenate(sub_scan_desc, axis=1)
    sub_score = np.concatenate(sub_scan_score)
    sub_ptcloud = np.concatenate(sub_scan_pts, axis=0)
    sub_kpts = np.concatenate(sub_scan_kpts, axis=0)


    kpts = np.concatenate([main_kpts, sub_kpts], axis=0)
    desc = np.concatenate([main_desc, sub_desc], axis=1)
    score = np.concatenate([main_score, sub_score], axis=0)
    ptcloud = np.concatenate([main_ptcloud, sub_ptcloud], axis=0)

    return kpts, desc, score, ptcloud