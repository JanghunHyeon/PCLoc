from utils.convert_function import pose_to_matrix, quat_multi, rpy2quat, rotm2quat, rotm2quat2
import numpy as np
import os
# data_path: .../.../xxx.txt
# pose: Nx7


def write_inloc_format(imglist, pred_list, output_txtname):
    assert len(imglist) == len(pred_list), ">> Number of images and list are not matched..."
    camM_for_inloc = open(output_txtname, "w")

    for idx in range(len(imglist)):
        qname = os.path.basename(imglist[idx])
        pred = pred_list[idx]

        # R = np.expand_dims(pred[:3, :3], 0)
        # qut_R  = rotm2quat(R).squeeze()

        R = pred[:3, :3]
        qut_R = rotm2quat2(R)

        T = pred[:3, 3]

        """Image name"""
        camM_for_inloc.write('{:s}'.format(qname))
        """pose"""
        camM_for_inloc.write((" {:.10g}"*4).format(*qut_R))
        camM_for_inloc.write((" {:.10g}"*3).format(*T))
        camM_for_inloc.write('\n')
    camM_for_inloc.close()


def write_capture_data_q(data_path, pose, fov=60, width=1280, height=720):
    camM_for_capture = open(data_path, "w")
    camM_for_capture.write('fov=%.1f\nwidth=%d\nheight=%d\n\n'%(fov, width, height))
    viewer_T = pose_to_matrix(pose)
    for i in range(viewer_T.shape[0]):
        camM_for_capture.write('c ' + str(i) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 0, :]) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 1, :]) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 2, :]) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 3, :]) + '\n')
    camM_for_capture.close()

def write_capture_data_T(data_path, viewer_T, fov=60, width=1280, height=720):
    camM_for_capture = open(data_path, "w")
    camM_for_capture.write('mode=Camera2D\nfov=%.1f\nwidth=%d\nheight=%d\n\n'%(fov, width, height))
    for i in range(viewer_T.shape[0]):
        camM_for_capture.write('c ' + str(i) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 0, :]) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 1, :]) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 2, :]) + '\t')
        camM_for_capture.write(' '.join(['%.6f'] * 4) % tuple(viewer_T[i, 3, :]) + '\n')
    camM_for_capture.close()



def gen_multipose(pose, dist, ang, step):
    # pose: Nx7
    # dist: int
    # ang: 1xN
    # step: int
    # result_pose: ((stpe*2+1)^2 * len(ang[0]*2+1 )x7
    result_pose = [[]]
    for i in range(pose.shape[0]):
        multipose = np.tile(pose[i, 0:3], (np.square(step * 2 + 1) * (len(ang[0])*2+1), 1))
        delx = np.linspace(-dist*step, dist*step, step*2+1)
        dely = np.linspace(-dist*step, dist*step, step*2+1)
        xx, yy = np.meshgrid(delx, dely)
        xx = np.expand_dims(np.repeat(xx, (len(ang[0])*2+1), axis=1).flatten(), axis=1)
        yy = np.expand_dims(np.repeat(yy, (len(ang[0])*2+1), axis=1).flatten(), axis=1)
        zz = np.zeros(xx.shape)
        ang = np.sort(ang)
        tmp_ang = np.zeros((len(ang[0])*2+1, 3))
        tmp_ang[:, 2] = np.concatenate((-ang[:,::-1], np.zeros((1,1)), ang), axis=1)[0]
        tmp_ang = rpy2quat(tmp_ang)
        for k, delq in enumerate(tmp_ang):
            tmp_ang[k, :] = quat_multi(delq, pose[i, 3:])
        qq = np.tile(tmp_ang, (np.square(step * 2 + 1), 1))
        multipose = multipose + np.concatenate((xx, yy, zz), axis=1)
        multipose = np.concatenate((multipose, qq), axis=1)
        if i == 0:
            result_pose = multipose
        else:
            result_pose = np.append(result_pose, multipose, axis=0)

    return result_pose

if __name__ == '__main__':
    result = gen_multipose(np.asarray([[0,0,0,0.358256, 0.003890, -0.006000, 0.933596]]), 0.2, [[20]], 1)

    capture_data_path = '/home/bc/catkin_ws/bin/data/pose.txt'
    capture_pose = np.asarray([
    [-37.592139, 56.783445, -0.263026, 0.358256, 0.003890, -0.006000, 0.933596],
    [-44.184411, 20.217787, -0.112560, 0.314927, 0.003847, 0.034366, 0.948486],
    [-72.379829, 28.585556, -0.202195, 0.342204, 0.011280, -0.008912, 0.939516],
    [-69.576114, -7.621645, -0.305172, 0.371134, -0.017806, 0.001792, -0.928407],
    [-59.009226, -35.653002, -0.147956, 0.935265, -0.001209, -0.007623, -0.353863],
    [-39.929455, -14.352878, 0.561849, 0.869492, 0.351771, 0.341497, 0.060172],
    [-45.236361, -57.229190, -0.143791, 0.914590, -0.000413, -0.008111, 0.404302],
    [-36.248138, -35.018424, -0.300618, 0.905116, 0.012201, -0.007166, 0.424929],
    [13.358296, -10.062822, -0.905851, 0.933374, 0.011458, -0.019553, -0.358190],
    [-5.750321, 5.773265, -0.314629, 0.343779, 0.006919, -0.009568, 0.938976],
    [-17.210320, 18.053395, -0.281601, 0.934447, -0.003495, -0.007890, -0.355998],
    [-1.688506, 27.057929, -0.395395, 0.926526, -0.012832, -0.012025, -0.375820],
    [33.680831, 13.356121, 1.615747, 0.625791, -0.659775, -0.229407, 0.347066],
    [39.914764, -9.372124, -0.083834, 0.284506, 0.018834, 0.006089, -0.958470],
    [-11.072221, -69.037609, 2.293209, 0.457349, -0.118026, -0.078592, 0.877910]
    ])

    write_capture_data_q(capture_data_path, capture_pose)


    # [-37.304527, 56.213520, -0.215341, 0.430134, -0.000384, -0.002593, 0.902761],
    # [-45.941644, 21.153391, -0.127853, 0.353766, -0.004464, -0.001946, 0.935321],
    # [-72.318950, 28.289167, -0.137342, 0.365522, -0.006614, -0.003053, 0.930774],
    # [-68.981101, -6.499547, -0.116908, 0.359905, 0.005577, -0.001648, -0.932971],
    # [-59.100675, -36.073200, -0.183408, 0.947951, 0.001626, -0.001207, -0.318410],
    # [-39.366492, -11.420331, -0.146094, 0.871263, 0.004932, 0.003806, -0.490777],
    # [-45.521562, -57.720645, -0.092077, 0.923137, -0.004165, 0.004722, 0.384420],
    # [-36.886473, -35.373555, -0.113684, 0.923312, -0.002666, 0.004050, 0.384020],
    # [12.303634, -8.772934, -0.162830, 0.918669, -0.000263, 0.001619, -0.395025],
    # [-5.704770, 5.226792, -0.105911, 0.346882, -0.002461, 0.006209, 0.937885],
    # [-17.193660, 18.248664, -0.103156, 0.949005, -0.003604, 0.006068, -0.315181],
    # [-2.000762, 27.695075, -0.174750, 0.917546, -0.001994, 0.002122, -0.397618],
    # [27.544827, 18.976944, -0.054788, 0.948944, 0.001002, 0.008753, -0.315320],
    # [42.102125, -13.236561, 0.044886, 0.444847, -0.020702, 0.011144, 0.895298],
    # [-11.653375, -65.846291, -0.285309, 0.352657, 0.003438, 0.002833, 0.935742]
