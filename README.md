# Pose Correction for Highly Accurate Visual Localization in Large-scale Indoor Spaces (ICCV 2021)

### Janghun Hyeon<sup>1* </sup>, JooHyung Kim<sup>1* </sup>, Nakju Doh<sup>1, 2 </sup>

#### <sup>1</sup> Korea University, <sup>2</sup> TeeLabs
#### * Equally contributed to this work.



​     [<img src="images/paper.png" width="18%"/>](https://openaccess.thecvf.com/content/ICCV2021/html/Hyeon_Pose_Correction_for_Highly_Accurate_Visual_Localization_in_Large-Scale_Indoor_ICCV_2021_paper.html)    [<img src="images/poster.png" width=37.0%>](https://janghunhyeon.github.io/PDF_pages/PCLoc_poster.pdf)    [<img src="images/slides.png" width=39%>](https://janghunhyeon.github.io/PDF_pages/PCLoc_slide_upload.pdf) 




---

## Abstract 
Indoor visual localization is significant for various applications such as autonomous robots, augmented reality, and mixed reality. Recent advances in visual localization have demonstrated their feasibility in large-scale indoor spaces through coarse-to-fine methods that typically employ three steps: image retrieval, pose estimation, and pose selection. However, further research is needed to improve the accuracy of large-scale indoor visual localization. We demonstrate that the limitations in the previous methods can be attributed to the sparsity of image positions in the database, which causes view-differences between a query and a retrieved image from the database. In this paper, to address this problem, we propose a novel module, named pose correction, that enables re-estimation of the pose with local feature matching in a similar view by reorganizing the local features. This module enhances the accuracy of the initially estimated pose and assigns more reliable ranks. Furthermore, the proposed method achieves a new state-of-the-art performance with an accuracy of more than 90 %within 1.0 m in the challenging indoor benchmark dataset InLoc for the first time.

<p align="center">
<img src="images/methods.png" width="60%"/>
</p>


## Dependencies
* Python 3

* Pytorch >= 1.1

* Tensorflow >= 1.13

* openCV >= 3.4

* Matplotlib >= 3.1

* Numpy >= 1.18

* scipy >= 1.4.1

* open3d >= 0.7.0.0

* vlfeat >= 0.9.20

* vlfeat-ctypes >= 0.1.5

  


## Prerequisite: Model Parameters
PCLoc is based on coarse-to-fine localization, which uses NetVLAD, SuperPoint, and SuperGlue.
Thus, the model parameter should be downloaded from the original code.

#### NetVLAD: [Download Model](http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip)
Download parameter from the above URL, and unzip the file at: 
```shell
./netvlad_tf/parm/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.data-00000-of-00001
./netvlad_tf/parm/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.index
./netvlad_tf/parm/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.meta
```

#### SuperPoint and SuperGlue: [Download Model](https://github.com/magicleap/SuperGluePretrainedNetwork)

```shell
./SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth
./SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth
```

## Prerequisite: Dataset

#### **Dataset**

To test our model using the InLoc dataset, the dataset should be downloaded. 
Downloading takes a while (Dataset is about 1.0TB).
[Click here](https://github.com/HajimeTaira/InLoc_dataset) to download dataset.



#### **Database Description**

* `netvlad_feats.npy`: Global descriptors (NetVLAD) of the database images.
* `feats_xyz.pkl`: 3D coordinates corresponding to the keypoints (local features: Superpoint) of each database image.
* `scan_graph.pkl`: Connected pose graph used for the inter-pose matching and pose verification.
* `scan_feat_normal.pkl`: Local feature map used for the pose correction.
* `scans_npy.npy`: RGB-D scan data from the dataset (InLoc), which is used for pose verification.

## Contents
The provided sample code (`06_main_inference.py`) runs pose correction.
This code provides three options:

* `--opt_div_matching`: Usage of Divided Matching
* `--opt_normal_filtering`: Usage of Normal Filtering
* `--opt_scangraph_pv`: Usage of ScanGraph at Pose Verification.

#### Example: --opt_div_matching/ --opt_normal_filtering/ --opt_scangraph_pv
* False/ False/ False: Table 5 (b-1) from the paper
* True/ False/ False : Table 5 (b-2) from the paper
* True/ True/ False  : Table 5 (b-3) from the paper
* True/ False/ True  : Table 5 (b-4) from the paper
* True/ True/ True   : Table 5 (b-5) from the paper

## Results

After running the code, results are shown in the `--debug_dir`.


Example: ./log/202103241833/IMG_0738/mpv
* `00_query_img.jpg`: image used for the query.
* `01_final_pose.jpg`: rendered image at the final pose.
* `02_final_err_30.837045.jpg`: error image between the query and the rendered image.
* `pred_IMG_0738.txt`: estimated final pose.
* `all/*`: top-k candidates from the pose correction.

<p align="center">
<img src="images/result1.png" width="70%"/>
</p>

<p align="center">
<img src="images/result2.png" width="70%"/>
</p>


| Error [*m*, 10<sup>o</sup>] | DUC1 | DUC2 |
| :---:  | :---: | :---: |
| InLoc  | 40.9/ 58.1/ 70.2 | 35.9/ 54.2/ 69.5|
| HFNet  | 39.9/ 55.6/ 67.2 | 37.4/ 57.3/ 70.2|
| KAPTURE| 41.4/ 60.1/ 73.7| 47.3/ 67.2/ 73.3|
| D2Net  | 43.9/ 61.6/ 73.7 | 42.0/ 60.3/ 74.8|
| Oracle | 43.9/ 66.2/ 78.3 | 43.5/ 63.4/ 76.3|
| Sparse NCNet| 47.0/ 67.2/ 79.8| 43.5/ 64.9/ 80.2|
| RLOCS | 47.0/ 71.2/ 84.8| 58.8/ 77.9/ 80.9|
| SuperGlue | 46.5/ 65.7/ 77.8| 51.9/ 72.5/ 79.4|
| **Baseline (3,000)**| 53.0/ 76.8/ 85.9|61.8/ 80.9/ 87.0|
| **Ours (3,000)**| 59.6/ 78.3/ 89.4|**71.0**/ **93.1**/ **93.9**|
| **Ours (4,096)** | **60.6**/ **79.8**/ **90.4**|70.2/ 92.4/ 93.1|

Every evaluation was conudcted with the online viusal localization benchmark server. 
[visuallocalization.net/benchmark](https://www.visuallocalization.net/benchmark)


## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@inproceedings{hyeon2021pose,
  title={Pose Correction for Highly Accurate Visual Localization in Large-Scale Indoor Spaces},
  author={Hyeon, Janghun and Kim, Joohyung and Doh, Nakju},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15974--15983},
  year={2021}
}
```