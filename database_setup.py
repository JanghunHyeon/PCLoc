from PIL import Image
import numpy as np
import argparse
import tensorflow as tf
import glob
import os
import torch
import pickle
from scipy import io
from tqdm import tqdm
import sys
sys.path.append('./thirdparty/netvlad_tf/python')
# tf.contrib.resampler  # import C++ op
import thirdparty.netvlad_tf.python.netvlad_tf.nets as nets
from thirdparty.SuperGluePretrainedNetwork.models.matching import SuperPoint
from thirdparty.SuperGluePretrainedNetwork.models.utils import read_image
from utils.load_WUSTL_transformation import load_transformation


parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='/mnt/hdd1/Dataset/InLoc_dataset', help='Path to Inloc dataset')
parser.add_argument('--save_dir', default='/mnt/hdd2/Working/ICCV_TEST', help='Path to save database features (Output)')
parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
parser.add_argument('--max_keypoints', type=int, default=3000, help='Maximum number of keypoints detected by Superpoint (-1 keeps all keypoints)')
args = parser.parse_args()

class data_loader:

    def __init__(self, data_dir):
        self.data_dir = data_dir

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.end_set = tf.errors.OutOfRangeError

    def get_dataset(self):
        def _read_image(data_dir):
            image = tf.read_file(data_dir)
            image = tf.image.decode_jpeg(image, channels=3)
            return image

        def _preprocess(image):
            # image = tf.image.resize_images(image, [480, 640], method=tf.image.ResizeMethod.BILINEAR)
            return image

        data = tf.data.Dataset.from_tensor_slices(self.data_dir)
        data = data.map(_read_image, num_parallel_calls=10)
        data = data.map(_preprocess, num_parallel_calls=10)
        dataset = tf.data.Dataset.zip({'image':data})
        with tf.device('/cpu:0'):
            tf_next = dataset.make_one_shot_iterator().get_next()

        while True:
            yield self.sess.run(tf_next)


# def main_save_netvlad_feat(db_dir, save_dir):
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#
#     tf.reset_default_graph()
#     image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
#     net_out = nets.vgg16NetvladPca(image_batch)
#     saver = tf.train.Saver()
#
#     sess = tf.Session()
#     saver.restore(sess, os.path.join(args.parm_dir, 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white'))
#
#     db_list = glob.glob(os.path.join(db_dir, '*.JPG'))
#     db_list.sort()
#
#     dataloader = data_loader(db_list)
#     dataset = dataloader.get_dataset()
#
#     pbar = tqdm(total=len(db_list))
#
#     gl_desc = []
#     while True:
#         try:
#             data = next(dataset)
#         except dataloader.end_set:
#             break
#
#         img = Image.fromarray(data['image'])
#         scale = max(img.size) / args.resize
#         rsize = (np.array(img.size)/scale).astype(int)
#
#         resized_img = img.resize(rsize)
#         batch = np.expand_dims(resized_img, axis=0)
#
#
#         descriptor = sess.run(net_out, feed_dict={image_batch:batch})
#         gl_desc.append(descriptor.squeeze())
#         pbar.update(1)
#     pbar.close()
#
#     global_feat = np.asarray(gl_desc, dtype=np.float32).transpose()
#
#     save_global_fname = os.path.join(save_dir, 'query_nvlad.npy')
#     np.save(save_global_fname, global_feat)
#
#     print(">>Save Feature Completed...")



def main_database_setup(args):

    print(">> [Database] Converting Path and Formats for Databases...")

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    db_img_list = []
    save_txtname = os.path.join(args.save_dir, 'pth_imgs.txt')
    with open(save_txtname, 'w') as f:
        bld_list = glob.glob(os.path.join(args.db_dir, 'database/cutouts/*'))
        bld_list.sort()
        for i_bld_pth in bld_list:
            i_scan_list = glob.glob(os.path.join(i_bld_pth, '*'))
            i_scan_list.sort()
            for i_scan_pth in i_scan_list:
                i_cutout_list = glob.glob(os.path.join(i_scan_pth, '*.jpg'))
                i_cutout_list.sort()
                for i_img_name in i_cutout_list:
                    db_img_list.append(i_img_name)
                    f.write(i_img_name + '\n')

    save_align_txtname = os.path.join(args.save_dir, 'pth_aligns.txt')
    with open(save_align_txtname, 'w') as f:
        for bld_pth in bld_list:
            align_mat_pth = glob.glob(os.path.join(args.db_dir, 'database/alignments', os.path.basename(bld_pth), 'transformations/*'))
            align_mat_pth.sort()
            for align_pth in align_mat_pth:
                f.write(align_pth + '\n')

    save_scan_dir = os.path.join(args.save_dir, 'scans_npy')
    save_scan_txtname = os.path.join(args.save_dir, 'pth_scans.txt')

    with open(save_scan_txtname, 'w') as f:
        bld_names = os.listdir(os.path.join(args.db_dir, 'database/cutouts'))
        bld_names.sort()

        for bld in bld_names:
            scan_list = glob.glob(os.path.join(args.db_dir, 'database/scans/{:s}/*'.format(bld)))
            scan_list.sort()
            save_scan_pth = os.path.join(save_scan_dir, bld)
            if not os.path.exists(save_scan_pth): os.makedirs(save_scan_pth)

            for scan_name in tqdm(scan_list):

                save_fname = os.path.join(save_scan_pth, os.path.basename(scan_name) + '.npy')
                f.write(save_fname + '\n')

                scan_mat = io.loadmat(scan_name)
                scan_data = scan_mat['A']
                scan_xyz = np.concatenate([scan_data[0, 0], scan_data[0, 1], scan_data[0, 2]], axis=1).astype(np.float16)
                scan_rgb = np.concatenate([scan_data[0, 4], scan_data[0, 5], scan_data[0, 6]], axis=1).astype(np.float16)

                data = np.concatenate([scan_xyz, scan_rgb], axis=1)
                np.save(save_fname, data)

    print(">> Converting Format Completed...")


    if not os.path.exists(os.path.join(args.save_dir, 'netvlad_feats.npy')):
        print(">> [Database] Global Feature Generation...")

        tf.reset_default_graph()
        image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

        net_out = nets.vgg16NetvladPca(image_batch)
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            # PATH TO GLOBAL RETRIEVAL NETWORK PARAMETER...
            saver.restore(sess, './thirdparty/netvlad_tf/checkpoints/vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

            dataloader = data_loader(db_img_list)
            dataset = dataloader.get_dataset()
            pbar = tqdm(total=len(db_img_list))
            gl_desc = []

            while True:
                try:
                    data = next(dataset)
                except dataloader.end_set:
                    break
                img = Image.fromarray(data['image'])
                scale = max(img.size) / 640
                rsize = (np.array(img.size) / scale).astype(int)

                resized_img = img.resize(rsize)
                batch = np.expand_dims(resized_img, axis=0)

                descriptor = sess.run(net_out, feed_dict={image_batch: batch})
                gl_desc.append(descriptor.squeeze())
                pbar.update(1)
            pbar.close()

        global_feat = np.asarray(gl_desc, dtype=np.float32).transpose()
        save_global_fname = os.path.join(args.save_dir, 'netvlad_feats.npy')
        np.save(save_global_fname, global_feat)

        print(">> Save Global Feature Completed...")


    local_feat_dir = os.path.join(args.save_dir, 'local_feats')
    pc_feat_dir = os.path.join(args.save_dir, 'pc_feats')
    if not os.path.exists(local_feat_dir): os.makedirs(local_feat_dir)
    if not os.path.exists(pc_feat_dir): os.makedirs(pc_feat_dir)

    print(">> [Database] Local Feature Generation...")
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {'superpoint': {'nms_radius': args.nms_radius,
                             'keypoint_threshold': args.keypoint_threshold,
                             'max_keypoints': args.max_keypoints
                             }}
    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)

    feat_idx = 0
    pc_idx = 0
    for bld_idx, bld_pth in enumerate(bld_list):
        scan_list = glob.glob(os.path.join(bld_pth, '*'))
        scan_list.sort()
        print(' ', end='', flush=True)
        text = "Database Processing [{}/{}]".format(bld_idx+1, len(bld_list))
        for scan_pth in tqdm(scan_list, desc=text):
            tr_scan_pth = glob.glob(os.path.join(args.db_dir, 'database/alignments', os.path.basename(bld_pth), 'transformations',
                                                 '*_trans_{}.txt'.format(os.path.basename(scan_pth))))[0]
            _, P_after = load_transformation(tr_scan_pth)

            cutout_list = glob.glob(os.path.join(scan_pth, '*.mat'))
            cutout_list.sort()

            scan_desc = []
            scan_score = []
            scan_xyz = []

            for cutout_pth in cutout_list:

                image0, inp0, scales0 = read_image(cutout_pth[:-4], device, [1200], 0, False)
                scan_data = io.loadmat(cutout_pth)

                xyz = scan_data['XYZcut']
                pred = superpoint({'image': inp0})

                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                keypoints = (pred['keypoints'] * scales0).astype(int)
                kpts_xyz = xyz[keypoints[:, 1], keypoints[:, 0], :]
                H_kpts = np.concatenate((kpts_xyz.T, np.ones((1, len(kpts_xyz)))), axis=0)
                align_xyz = np.matmul(P_after, H_kpts)
                align_xyz = np.divide(align_xyz[:3, :], align_xyz[3, :]).T

                nan_idx = ~np.isnan(align_xyz).all(axis=1)

                rm_keypoints = keypoints[nan_idx, :]
                rm_descrptors = pred['descriptors'][:, nan_idx]
                rm_scores = pred['scores'][nan_idx]
                rm_xyz = align_xyz[nan_idx, :]

                feat_cutout = dict()
                feat_cutout['keypoints'] = rm_keypoints
                feat_cutout['scores'] = rm_scores
                feat_cutout['descriptors'] = rm_descrptors
                feat_cutout['pts_xyz'] = rm_xyz

                save_cutout_feat_fname = os.path.join(local_feat_dir, 'local_feat_{:05}.pkl'.format(feat_idx))
                with open(save_cutout_feat_fname, 'wb') as handle:
                    pickle.dump(feat_cutout, handle, protocol=pickle.HIGHEST_PROTOCOL)
                feat_idx += 1

                scan_desc.append(rm_descrptors)
                scan_score.append(rm_scores)
                scan_xyz.append(rm_xyz)

            total_score = np.concatenate(scan_score, 0)
            total_desc = np.concatenate(scan_desc, 1)
            total_xyz = np.concatenate(scan_xyz, 0)

            pc_feat = dict()
            pc_feat['ptcloud'] = total_xyz
            pc_feat['descriptors'] = total_desc
            pc_feat['scores'] = total_score


            save_pcfeat_fname = os.path.join(pc_feat_dir, 'pcfeat_{:05}.pkl'.format(pc_idx))
            with open(save_pcfeat_fname, 'wb') as handle:
                pickle.dump(pc_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pc_idx += 1
    print(">> Save Local Feature and PC Feature Completed...")

if __name__ == '__main__':

    main_database_setup(args)
