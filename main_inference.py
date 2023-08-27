import cv2
from PIL import Image
import numpy as np
import time
import argparse
import glob
import os
import logging
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
from itertools import repeat
import torch
import tensorflow as tf
import sys
sys.path.append('./thirdparty/netvlad_tf/python')




import thirdparty.netvlad_tf.python.netvlad_tf.nets as nets
from utils.load_WUSTL_transformation import load_transformation

from thirdparty.SuperGluePretrainedNetwork.models.matching import SuperGlue, SuperPoint
from refinements import Refinement_extended, Refinement_base
from thirdparty.SuperGluePretrainedNetwork.models.utils import read_image

from utils.utils import scan2imgfeat_projection, scan2imgfeat_projection_normal, do_pnp, loc_failure, LocResult, seq_proj_point2img, proj_point2img, compute_errormap_morph_mavg_v2
from utils.convert_format import write_inloc_format

# tf.contrib.resampler  # import C++ op
logger = logging.getLogger(__name__)
time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))


parser = argparse.ArgumentParser()
parser.add_argument('--query_dir', default='/mnt/hdd1/Dataset/InLoc_dataset/query/iphone7', help='Path to query directory')
parser.add_argument('--db_dir', default='/mnt/hdd2/Working/ICCV_TEST')
parser.add_argument('--log_dir', default='./log')

parser.add_argument('--num_thread', default=5)
parser.add_argument('--query_idx', default=5, type=int)

parser.add_argument('--num_topk_ret', default=100)
parser.add_argument('--num_topk_pe', default=20)
parser.add_argument('--num_topk_rf', default=10)


parser.add_argument('--pv_scale', default=0.125)
parser.add_argument('--fl', default=3136.0, help='4032*28/36')
parser.add_argument('--reproj_err', default=5, type=int)

parser.add_argument('--resize', type=int, nargs='+', default=[1200])
parser.add_argument('--resize_float', action='store_true', help='Resize the image after casting uint8 to float')

parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
parser.add_argument('--max_keypoints', type=int, default=3000, help='Maximum number of keypoints detected by Superpoint (-1 keeps all keypoints)')

parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.2, help='SuperGlue match threshold')

parser.add_argument('--opt_div_matching', default=True, choices=('True','False'), help='True: Divided matching, False: base matching')

args = parser.parse_args()
torch.set_grad_enabled(False)

def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - desc1 @ desc2.T)

def read_image_raw(img_path, rs_img=None):
    img = cv2.imread(img_path)[:, :, ::-1]
    if rs_img != None:
        img = cv2.resize(img, dsize=rs_img, interpolation=cv2.INTER_AREA)
    return img

def chunks(lst, n_thred):
    """Yield successive n-sized chunks from lst."""
    size = int(np.ceil(len(lst) / np.float(n_thred)))
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def convert_superglue_db_format(img_tensor, pred_q, pred_kpts, pred_desc, pred_score, device):


    pred_feat = {'keypoints': [torch.from_numpy(pred_kpts).to(device)],
                 'descriptors': [torch.from_numpy(pred_desc).to(device)],
                 'scores': [torch.from_numpy(pred_score).to(device)]}
    pred = {}
    pred = {**pred, **{k + '0': v for k, v in pred_q.items()}}
    pred = {**pred, **{k + '1': v for k, v in pred_feat.items()}}

    data = {'image0': img_tensor, 'image1': img_tensor, **pred}

    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])

    return data

def main_evaluation(query_dir, query_idx, db_dir, num_topk_ret, num_topk_pe, num_topk_rf, pv_scale, fl):

    query_list = glob.glob(os.path.join(query_dir, '*.JPG'))
    query_list.sort()
    query_pth = query_list[query_idx]

    db_nvlad_feat_pth = os.path.join(db_dir, 'netvlad_feats.npy')
    db_nvlad = np.load(db_nvlad_feat_pth)

    local_feat_list = glob.glob(os.path.join(db_dir, 'local_feats/*.pkl'))
    local_feat_list.sort()


    db_img_dir = os.path.join(db_dir, 'pth_imgs.txt')
    db_img_list = pd.read_csv(db_img_dir, header=None, sep='\n').values.squeeze()

    db_scan_dir = os.path.join(db_dir, 'pth_scans.txt')
    scan_list = pd.read_csv(db_scan_dir, header=None, sep='\n').values.squeeze()

    db_align_dir = os.path.join(db_dir, 'pth_aligns.txt')
    align_list = pd.read_csv(db_align_dir, header=None, sep='\n').values.squeeze()

    pcfeat_list = np.array(glob.glob(os.path.join(db_dir, 'pc_feats/*.pkl')), dtype=object)
    pcfeat_list.sort()
    print(">> Loading Database completed...")

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {'superpoint': {'nms_radius': args.nms_radius,
                             'keypoint_threshold': args.keypoint_threshold,
                             'max_keypoints': args.max_keypoints
                             },
              'superglue': {'weights': args.superglue,
                            'sinkhorn_iterations': args.sinkhorn_iterations,
                            'match_threshold': args.match_threshold,
                            }
              }
    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
    superglue = SuperGlue(config.get('superglue', {})).eval().to(device)

    """
    Extended Pose Correction Option
    e.g.
        refinement = Refinement_base(superglue)     : Base PC (original matching) 
        refinement = Refinement_extended(superglue) : Extended PC (Div matching)
    """
    if args.opt_div_matching == 'True':
        refinement = Refinement_extended(superglue)
    else:
        refinement = Refinement_base(superglue)



    debug_pth = os.path.join(args.log_dir, time_stamp)
    if not os.path.exists(debug_pth): os.makedirs(debug_pth)

    debug_fname = os.path.basename(query_pth).split('.')[0]
    debug_folder = os.path.join(debug_pth, debug_fname)
    if not os.path.exists(debug_folder): os.makedirs(debug_folder)
    temp_mp_dir = os.path.join(debug_folder, 'temp_mp')
    image0, inp0, scales0 = read_image(query_pth, device, args.resize, 0, args.resize_float)


    cx = image0.shape[1] / 2
    cy = image0.shape[0] / 2
    fx = fl / scales0[1]
    fy = fl / scales0[0]
    camera_parm = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


    # Load NetVlad models.
    tf.reset_default_graph()
    image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, './thirdparty/netvlad_tf/checkpoints/vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

        img = Image.open(query_pth)
        scale = max(img.size) / 640
        rsize = (np.array(img.size) / scale).astype(int)
        resized_img = img.resize(rsize)
        batch = np.expand_dims(resized_img, axis=0)
        final_poses_mpv = []

        """Global matching"""
        st_time = time.time()
        global_desc = sess.run(net_out, feed_dict={image_batch: batch})
        global_dist = compute_distance(global_desc.squeeze(), db_nvlad.T)
        nearest_global_dist = np.argsort(global_dist)
        topk_candidate = nearest_global_dist[:num_topk_ret]
        print('>> Global matching takes {:f}seconds...'.format(time.time() - st_time))



    """Pose estimation"""
    pred0 = superpoint({'image': inp0})

    clustered_frames = np.expand_dims(topk_candidate, -1)
    pred_poses = []
    num_inliers = []
    # retscan_idx = np.unique(clustered_frames // 36)

    print(' ', end='', flush=True)
    text = "Local Feature Matching"
    for place in tqdm(clustered_frames, desc=text):

        db_img_dir = db_img_list[place[0]]
        db_feat_dir = local_feat_list[place[0]]

        with open(db_feat_dir, 'rb') as handle:
            db_img_feats = pickle.load(handle)
            image1, inp1, scales1 = read_image(db_img_dir, device, args.resize, 0, args.resize_float)
            pred1 = {'keypoints': [torch.from_numpy(db_img_feats['keypoints']).to(device)],
                     'descriptors': [torch.from_numpy(db_img_feats['descriptors']).to(device)],
                     'scores': [torch.from_numpy(db_img_feats['scores']).to(device)]}


        pred_pre = {}
        pred_pre = {**pred_pre, **{k + '0': v for k, v in pred0.items()}}
        pred_pre = {**pred_pre, **{k + '1': v for k, v in pred1.items()}}
        data = {'image0': inp0, 'image1': inp1, **pred_pre}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred_pre = {**pred_pre, **superglue(data)}
        pred_pre = {k: v[0].cpu().numpy() for k, v in pred_pre.items()}

        kpts0, kpts1 = pred_pre['keypoints0'], pred_pre['keypoints1']
        matches, conf = pred_pre['matches0'], pred_pre['matching_scores0']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mkpts1_xyz = db_img_feats['pts_xyz'][matches[valid]]
        mconf = conf[valid]

        nan_valid = ~np.isnan(mkpts1_xyz).all(axis=1)
        fkpts0 = mkpts0[nan_valid]
        fkpts1 = mkpts1[nan_valid]
        fkpts1_xyz = mkpts1_xyz[nan_valid]
        fconf = mconf[nan_valid]

        "PnP"
        if len(fconf) > 3:
            result, inliers = do_pnp(fkpts0, fkpts1_xyz, camera_parm, 0.00)

        else:
            result = loc_failure
            T_w2c = np.array([[1.0, 0.0, 0.0, 1000.0],
                              [0.0, 1.0, 0.0, 1000.0],
                              [0.0, 0.0, 1.0, 1000.0],
                              [0.0, 0.0, 0.0, 1.0]])
            result = LocResult(False, result.num_inliers, result.inlier_ratio, T_w2c)

        if result.success:
            T_c2w = result.T
            T_w2c = np.linalg.inv(T_c2w)

        else:
            T_w2c = np.array([[1.0, 0.0, 0.0, 1000.0],
                              [0.0, 1.0, 0.0, 1000.0],
                              [0.0, 0.0, 1.0, 1000.0],
                              [0.0, 0.0, 0.0, 1.0]])
        num_inliers.append(result.num_inliers)
        pred_poses.append(T_w2c)

    num_inliers_sort = np.argsort(np.array(num_inliers))[::-1]
    topk_inliers = num_inliers_sort[:num_topk_pe]
    filtered_frames = clustered_frames[topk_inliers]

    """Pose Correction"""
    print(">> Pose Correction using the pre-estimated poses...")
    refine_poses = []
    num_inliers = []
    for idx in tqdm(topk_inliers):
        topk_idx = clustered_frames[idx]
        pred_pose = pred_poses[idx]
        pcfeat_pth = pcfeat_list[topk_idx // 36]

        pred_kpts, pred_desc, pred_score, pred_xyz = scan2imgfeat_projection(pred_pose, pcfeat_pth, camera_parm, num_kpts=args.max_keypoints)

        data = convert_superglue_db_format(inp0, pred0, pred_kpts, pred_desc, pred_score, device)
        mkpts0, mkpts_xyz = refinement(data, pred_xyz)


        if len(mkpts0) > 3:
            result, inliers = do_pnp(mkpts0, mkpts_xyz, camera_parm, 0.00, reproj_error=args.reproj_err)

        else:
            result = loc_failure
            T_w2c = pred_pose
            result = LocResult(False, result.num_inliers, result.inlier_ratio, T_w2c)

        if result.success:
            T_c2w = result.T
            T_w2c = np.linalg.inv(T_c2w)

        else:
            T_w2c = pred_pose

        refine_poses.append(T_w2c)
        num_inliers.append(result.num_inliers)

    num_inliers_sort = np.argsort(np.array(num_inliers))[::-1]
    topk_inliers = num_inliers_sort[:num_topk_rf]

    """Pose Verification"""
    mp_pred_pose = []
    # scan_idx = []
    align_pth = []
    scan_pth = []

    for idx in topk_inliers:
        topk_idx = filtered_frames[idx]
        mp_pred_pose.append(refine_poses[idx])
        # scan_idx.append(topk_idx // 36)
        align_pth.append(align_list[topk_idx // 36])
        scan_pth.append(scan_list[topk_idx // 36])

    rcx = cx * scales0[1] * pv_scale
    rcy = cy * scales0[0] * pv_scale
    rfx = fx * scales0[1] * pv_scale
    rfy = fy * scales0[0] * pv_scale
    rcamera_parm = np.array([[rfx, 0, rcx], [0, rfy, rcy], [0, 0, 1]])

    temp_mp_pv_dir = os.path.join(temp_mp_dir, 'pv')
    if not os.path.exists(temp_mp_pv_dir): os.makedirs(temp_mp_pv_dir)

    # num_process = multiprocessing.cpu_count()
    num_process = args.num_thread
    pool = multiprocessing.Pool(processes=num_process)

    mp_pred_pose_set = list(chunks(mp_pred_pose, num_process))
    # scan_idx_set = list(chunks(scan_idx, num_process))
    pos_idx = np.arange(len(mp_pred_pose_set))
    align_pth_set = list(chunks(align_pth, num_process))
    scan_pth_set = list(chunks(scan_pth, num_process))


    pool.starmap(pose_verification_mpv_mp,
                 zip(pos_idx, mp_pred_pose_set, align_pth_set, scan_pth_set,
                     repeat(query_pth), repeat(rcamera_parm), repeat(temp_mp_pv_dir)))

    pool.close()

    pv_results = glob.glob(os.path.join(temp_mp_pv_dir, '*.pickle'))
    pv_results.sort()

    pv_img = []
    pv_err_mpv = []
    pv_score_mpv = []

    for result_pth in pv_results:
        with open(result_pth, 'rb') as handle:
            result = pickle.load(handle)
        for idx in range(len(result['score_morph_mavg'])):
            pv_img.append(result['pv_img'][idx])
            pv_err_mpv.append(result['err_map_morph'][idx])
            pv_score_mpv.append(result['score_morph_mavg'][idx])


    debug_folder_mpv = os.path.join(debug_folder, 'mpv')
    if not os.path.exists(debug_folder_mpv): os.makedirs(debug_folder_mpv)

    pv_score_mpv = np.array(pv_score_mpv)
    top1_pv_idx_mpv = np.argmax(pv_score_mpv)
    top1_idx_mpv = topk_inliers[top1_pv_idx_mpv]
    final_pred_mpv = refine_poses[top1_idx_mpv]

    rs_x = int(cx * scales0[1] * pv_scale * 2)
    rs_y = int(cy * scales0[0] * pv_scale * 2)
    resized_query = read_image_raw(query_pth, (rs_x, rs_y))

    query_name = os.path.join(debug_folder_mpv, '00_query_img.jpg')
    cv2.imwrite(query_name, resized_query[:, :, ::-1])


    top1_name = os.path.join(debug_folder_mpv, '01_final_pose.jpg')
    cv2.imwrite(top1_name, pv_img[top1_pv_idx_mpv][:, :, ::-1])

    top1_errname = os.path.join(debug_folder_mpv, '02_final_err_{:f}.jpg'.format(pv_score_mpv[top1_pv_idx_mpv]))
    err_img = cv2.resize(pv_err_mpv[top1_pv_idx_mpv], dsize=resized_query.shape[:2][::-1])
    err_img[np.isnan(err_img)] = 0

    plt.imsave(top1_errname, err_img, cmap='jet')

    for idx, score in enumerate(pv_score_mpv):
        pv_fname = os.path.join(debug_folder_mpv, 'all')

        if not os.path.exists(pv_fname): os.makedirs(pv_fname)
        pv_name = os.path.join(pv_fname, 'top{:02d}_inlier_s{:f}.jpg'.format(idx + 1, score))
        err_name = os.path.join(pv_fname, 'top{:02d}_inlier_s{:f}_err.jpg'.format(idx + 1, score))

        pv_dfile = pv_img[idx][:, :, ::-1]
        cv2.imwrite(pv_name, pv_dfile)

        err_img = cv2.resize(pv_err_mpv[idx], dsize=resized_query.shape[:2][::-1])
        err_img[np.isnan(err_img)] = 0
        plt.imsave(err_name, err_img, cmap='jet')

    final_poses_mpv.append(final_pred_mpv)

    single_result_fname = os.path.join(debug_folder_mpv, 'pred_{:s}.txt'.format(debug_fname))
    write_inloc_format([query_list[query_idx]], [final_pred_mpv], single_result_fname)


def pose_verification_mpv_mp(pos_idx, pred_list, align_list, scan_list, query_img, camera_parm, output_dir):
    print(' ', end='', flush=True)
    text = "Progresser #{:02d}".format(pos_idx)
    result_pv = dict()
    pv_img = []

    pv_err_morph = []
    pv_morph_mavg_score = []

    for idx in tqdm(range(len(pred_list)), desc=text, position=pos_idx):

        align_pth = align_list[idx][0]
        scan_pth = scan_list[idx][0]
        pred_pose = pred_list[idx]

        _, P_after = load_transformation(align_pth)

        scan_mat = np.load(scan_pth)
        scan_xyz = scan_mat[:, :3]
        scan_rgb = scan_mat[:, 3:]

        H_xyz = np.concatenate((scan_xyz.T, np.ones((1, len(scan_xyz)))), axis=0)

        aligned_xyz = np.matmul(P_after, H_xyz)
        aligned_xyz = np.divide(aligned_xyz[:3, :], aligned_xyz[3, :]).T

        try:
            proj_rgb, proj_xyz = proj_point2img(scan_rgb, aligned_xyz, camera_parm, pred_pose)
            rs_x = int(camera_parm[0, 2] * 2)
            rs_y = int(camera_parm[1, 2] * 2)

            resized_query = read_image_raw(query_img, (rs_x, rs_y))

            score_morph_mavg, errmap_morph = compute_errormap_morph_mavg_v2(resized_query, proj_rgb, proj_xyz)


        except:
            score_morph_mavg = 0
            errmap_morph = np.zeros([200, 200])
            proj_rgb = np.zeros([200, 200, 3])

        pv_img.append(proj_rgb)
        pv_err_morph.append(errmap_morph)


        pv_morph_mavg_score.append(score_morph_mavg)


    save_pv = os.path.join(output_dir, 'core_{:02d}_pv.pickle'.format(pos_idx))
    result_pv['pv_img'] = pv_img
    result_pv['err_map_morph'] = pv_err_morph
    result_pv['score_morph_mavg'] = pv_morph_mavg_score


    with open(save_pv, 'wb') as handle:
        pickle.dump(result_pv, handle, protocol=pickle.HIGHEST_PROTOCOL)


def graph_pose_verification_v3_mpv_mp(pos_idx, pred_list, scan_idx_set, scan_list, align_list, retscan_idx, scan_graph, query_img, camera_parm, output_dir):
    print(' ', end='', flush=True)
    text = "Progresser #{:02d}".format(pos_idx)
    result_pv = dict()

    pv_img = []
    pv_err_morph = []
    pv_morph_mavg_score = []

    for idx in tqdm(range(len(pred_list)), desc=text, position=pos_idx):

        pred_pose = pred_list[idx]
        scan_idx = scan_idx_set[idx][0]

        scan_graph_idx = scan_graph[scan_idx]

        vis_scan_idx = list()
        vis_scan_idx.append(scan_idx)
        for sidx in scan_graph_idx:
            if sidx in retscan_idx:
                vis_scan_idx.append(sidx)

        try:
            proj_rgb, proj_xyz = seq_proj_point2img(vis_scan_idx, align_list, scan_list, camera_parm, pred_pose)
            rs_x = int(camera_parm[0, 2] * 2)
            rs_y = int(camera_parm[1, 2] * 2)

            resized_query = read_image_raw(query_img, (rs_x, rs_y))
            score_morph_mavg, errmap_morph = compute_errormap_morph_mavg_v2(resized_query, proj_rgb, proj_xyz)


        except:
            score_morph_mavg = 0
            errmap_morph = np.zeros([200, 200])
            proj_rgb = np.zeros([200, 200, 3])

        pv_img.append(proj_rgb)
        pv_err_morph.append(errmap_morph)
        pv_morph_mavg_score.append(score_morph_mavg)

    save_pv = os.path.join(output_dir, 'core_{:02d}_pv.pickle'.format(pos_idx))
    result_pv['pv_img'] = pv_img
    result_pv['err_map_morph'] = pv_err_morph
    result_pv['score_morph_mavg'] = pv_morph_mavg_score

    with open(save_pv, 'wb') as handle:
        pickle.dump(result_pv, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main_evaluation(args.query_dir, args.query_idx, args.db_dir, args.num_topk_ret, args.num_topk_pe, args.num_topk_rf, args.pv_scale, args.fl)