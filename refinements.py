import torch
from thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue
import numpy as np

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred




class Refinement_extended(torch.nn.Module):

    # def __init__(self, config={}):
    #     super().__init__()
    #     self.superglue = SuperGlue(config.get('superglue', {}))
    #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, network={}):
        super().__init__()
        self.superglue = network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, data, ptcloud):

        data_cpu = {k: v[0].cpu().numpy() for k, v in data.items()}
        kpts0, kpts1 = data_cpu['keypoints0'], data_cpu['keypoints1']
        desc0, desc1 = data_cpu['descriptors0'], data_cpu['descriptors1']
        scores0, scores1 = data_cpu['scores0'], data_cpu['scores1']


        """"Original feature matching... """

        result_full = {**data, **self.superglue(data)}
        pred_full = {k: v[0].cpu().numpy() for k, v in result_full.items()}
        kpts0_full, kpts1_full = pred_full['keypoints0'], pred_full['keypoints1']
        matches_full, conf_full = pred_full['matches0'], pred_full['matching_scores0']

        valid_full = matches_full > -1
        mkpts0_full = kpts0_full[valid_full]
        mkpts1_full = kpts1_full[matches_full[valid_full]]
        mconf_full = conf_full[valid_full]
        mkpts1_xyz_full = ptcloud[matches_full[valid_full], :]





        """"Top-side feature matching... """

        img0_up_idx = kpts0[:, 1] < (data_cpu['image0'].shape[1] / 2)
        kpts0_up = kpts0[img0_up_idx, :]
        desc0_up = desc0[:, img0_up_idx]
        scores0_up = scores0[img0_up_idx]

        img1_up_idx = kpts1[:, 1] < (data_cpu['image1'].shape[1] / 2)

        kpts1_up = kpts1[img1_up_idx, :]
        desc1_up = desc1[:, img1_up_idx]
        scores1_up = scores1[img1_up_idx]
        ptcloud_up = ptcloud[img1_up_idx, :]

        data_up = {'image0': data['image0'],
                   'image1': data['image1'],

                   'keypoints0': [torch.from_numpy(kpts0_up).to(self.device)],
                   'descriptors0': [torch.from_numpy(desc0_up).to(self.device)],
                   'scores0': [torch.from_numpy(scores0_up).to(self.device)],

                   'keypoints1': [torch.from_numpy(kpts1_up).to(self.device)],
                   'descriptors1': [torch.from_numpy(desc1_up).to(self.device)],
                   'scores1': [torch.from_numpy(scores1_up).to(self.device)]}

        for k in data_up:
            if isinstance(data_up[k], (list, tuple)):
                data_up[k] = torch.stack(data_up[k])

        result_up = {**data_up, **self.superglue(data_up)}
        pred_up = {k: v[0].cpu().numpy() for k, v in result_up.items()}
        kpts0_up, kpts1_up = pred_up['keypoints0'], pred_up['keypoints1']
        matches_up, conf_up = pred_up['matches0'], pred_up['matching_scores0']

        valid_up = matches_up > -1
        mkpts0_up = kpts0_up[valid_up]
        mkpts1_up = kpts1_up[matches_up[valid_up]]
        mconf_up = conf_up[valid_up]
        mkpts1_xyz_up = ptcloud_up[matches_up[valid_up], :]



        # ############
        # save_matching_fname = 'up.png'
        # debug_text = ['SuperGlue',
        #               'Keypoints: {}:{}'.format(len(kpts0_up), len(kpts1_up)),
        #               'Matches: {}'.format(len(mkpts0_up))]
        # color = cm.jet(mconf_up)
        # make_matching_plot(data_cpu['image0'].squeeze(), data_cpu['image0'].squeeze(),
        #                    kpts0_up, kpts1_up,
        #                    mkpts0_up, mkpts1_up,
        #                    color, debug_text, save_matching_fname)
        # ###########



        """"Down-side feature matching... """

        img0_down_idx = kpts0[:, 1] >= (data_cpu['image0'].shape[1] / 2)
        kpts0_down = kpts0[img0_down_idx, :]
        desc0_down = desc0[:, img0_down_idx]
        scores0_down = scores0[img0_down_idx]

        img1_down_idx = kpts1[:, 1] >= (data_cpu['image1'].shape[1] / 2)

        kpts1_down = kpts1[img1_down_idx, :]
        desc1_down = desc1[:, img1_down_idx]
        scores1_down = scores1[img1_down_idx]
        ptcloud_down = ptcloud[img1_down_idx, :]

        data_down = {'image0': data['image0'],
                     'image1': data['image1'],

                     'keypoints0': [torch.from_numpy(kpts0_down).to(self.device)],
                     'descriptors0': [torch.from_numpy(desc0_down).to(self.device)],
                     'scores0': [torch.from_numpy(scores0_down).to(self.device)],

                     'keypoints1': [torch.from_numpy(kpts1_down).to(self.device)],
                     'descriptors1': [torch.from_numpy(desc1_down).to(self.device)],
                     'scores1': [torch.from_numpy(scores1_down).to(self.device)]}

        for k in data_down:
            if isinstance(data_down[k], (list, tuple)):
                data_down[k] = torch.stack(data_down[k])

        result_down = {**data_down, **self.superglue(data_down)}
        pred_down = {k: v[0].cpu().numpy() for k, v in result_down.items()}
        kpts0_down, kpts1_down = pred_down['keypoints0'], pred_down['keypoints1']
        matches_down, conf_down = pred_down['matches0'], pred_down['matching_scores0']

        valid_down = matches_down > -1
        mkpts0_down = kpts0_down[valid_down]
        mkpts1_down = kpts1_down[matches_down[valid_down]]
        mconf_down = conf_down[valid_down]
        mkpts1_xyz_down = ptcloud_down[matches_down[valid_down], :]

# # ############
#
#         save_matching_fname = 'down.png'
#         debug_text = ['SuperGlue',
#                       'Keypoints: {}:{}'.format(len(kpts0_down), len(kpts1_down)),
#                       'Matches: {}'.format(len(mkpts0_down))]
#         color = cm.jet(mconf_down)
#         make_matching_plot(data_cpu['image0'].squeeze(), data_cpu['image0'].squeeze(),
#                            kpts0_down, kpts1_down,
#                            mkpts0_down, mkpts1_down,
#                            color, debug_text,
#                            save_matching_fname)
# # ###########



        """"Left-side feature matching... """

        img0_left_idx = kpts0[:, 0] < (data_cpu['image0'].shape[2] / 2)
        kpts0_left = kpts0[img0_left_idx, :]
        desc0_left = desc0[:, img0_left_idx]
        scores0_left = scores0[img0_left_idx]

        img1_left_idx = kpts1[:, 0] < (data_cpu['image1'].shape[2] / 2)

        kpts1_left = kpts1[img1_left_idx, :]
        desc1_left = desc1[:, img1_left_idx]
        scores1_left = scores1[img1_left_idx]
        ptcloud_left = ptcloud[img1_left_idx, :]

        data_left = {'image0': data['image0'],
                     'image1': data['image1'],

                     'keypoints0': [torch.from_numpy(kpts0_left).to(self.device)],
                     'descriptors0': [torch.from_numpy(desc0_left).to(self.device)],
                     'scores0': [torch.from_numpy(scores0_left).to(self.device)],

                     'keypoints1': [torch.from_numpy(kpts1_left).to(self.device)],
                     'descriptors1': [torch.from_numpy(desc1_left).to(self.device)],
                     'scores1': [torch.from_numpy(scores1_left).to(self.device)]}

        for k in data_left:
            if isinstance(data_left[k], (list, tuple)):
                data_left[k] = torch.stack(data_left[k])

        result_left = {**data_left, **self.superglue(data_left)}
        pred_left = {k: v[0].cpu().numpy() for k, v in result_left.items()}
        kpts0_left, kpts1_left = pred_left['keypoints0'], pred_left['keypoints1']
        matches_left, conf_left = pred_left['matches0'], pred_left['matching_scores0']

        valid_left = matches_left > -1
        mkpts0_left = kpts0_left[valid_left]
        mkpts1_left = kpts1_left[matches_left[valid_left]]
        mconf_left = conf_left[valid_left]
        mkpts1_xyz_left = ptcloud_left[matches_left[valid_left], :]

        """"Left-side feature matching... """

        img0_right_idx = kpts0[:, 0] >= (data_cpu['image0'].shape[2] / 2)
        kpts0_right = kpts0[img0_right_idx, :]
        desc0_right = desc0[:, img0_right_idx]
        scores0_right = scores0[img0_right_idx]

        img1_right_idx = kpts1[:, 0] >= (data_cpu['image1'].shape[2] / 2)

        kpts1_right = kpts1[img1_right_idx, :]
        desc1_right = desc1[:, img1_right_idx]
        scores1_right = scores1[img1_right_idx]
        ptcloud_right = ptcloud[img1_right_idx, :]

        data_right = {'image0': data['image0'],
                      'image1': data['image1'],

                      'keypoints0': [torch.from_numpy(kpts0_right).to(self.device)],
                      'descriptors0': [torch.from_numpy(desc0_right).to(self.device)],
                      'scores0': [torch.from_numpy(scores0_right).to(self.device)],

                      'keypoints1': [torch.from_numpy(kpts1_right).to(self.device)],
                      'descriptors1': [torch.from_numpy(desc1_right).to(self.device)],
                      'scores1': [torch.from_numpy(scores1_right).to(self.device)]}

        for k in data_right:
            if isinstance(data_right[k], (list, tuple)):
                data_right[k] = torch.stack(data_right[k])

        result_right = {**data_right, **self.superglue(data_right)}
        pred_right = {k: v[0].cpu().numpy() for k, v in result_right.items()}
        kpts0_right, kpts1_right = pred_right['keypoints0'], pred_right['keypoints1']
        matches_right, conf_right = pred_right['matches0'], pred_right['matching_scores0']

        valid_right = matches_right > -1
        mkpts0_right = kpts0_right[valid_right]
        mkpts1_right = kpts1_right[matches_right[valid_right]]
        mconf_right = conf_right[valid_right]
        mkpts1_xyz_right = ptcloud_right[matches_right[valid_right], :]



        mkpts0 = np.concatenate([mkpts0_full,
                                 mkpts0_up, mkpts0_down,
                                 mkpts0_left, mkpts0_right], axis=0)

        mkpts1_xyz = np.concatenate([mkpts1_xyz_full,
                                     mkpts1_xyz_up, mkpts1_xyz_down,
                                     mkpts1_xyz_left, mkpts1_xyz_right], axis=0)

        return mkpts0, mkpts1_xyz


class Refinement_base(torch.nn.Module):

    def __init__(self, network={}):
        super().__init__()
        self.superglue = network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, data, ptcloud):

        """"Original feature matching... """
        result_full = {**data, **self.superglue(data)}
        pred_full = {k: v[0].cpu().numpy() for k, v in result_full.items()}
        kpts0_full, kpts1_full = pred_full['keypoints0'], pred_full['keypoints1']
        matches_full, conf_full = pred_full['matches0'], pred_full['matching_scores0']

        valid_full = matches_full > -1
        mkpts0_full = kpts0_full[valid_full]
        mkpts1_xyz_full = ptcloud[matches_full[valid_full], :]

        return mkpts0_full, mkpts1_xyz_full