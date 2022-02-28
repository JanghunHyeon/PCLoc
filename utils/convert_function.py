import numpy as np

def quat_multi(quatA, quatB):
    # Hamilton product

    res = np.zeros(4)
    res[0] = quatA[0] * quatB[0] - quatA[1] * quatB[1] - quatA[2] * quatB[2] - quatA[3] * quatB[3]
    res[1] = quatA[0] * quatB[1] + quatA[1] * quatB[0] + quatA[2] * quatB[3] - quatA[3] * quatB[2]
    res[2] = quatA[0] * quatB[2] - quatA[1] * quatB[3] + quatA[2] * quatB[0] + quatA[3] * quatB[1]
    res[3] = quatA[0] * quatB[3] + quatA[1] * quatB[2] - quatA[2] * quatB[1] + quatA[3] * quatB[0]

    return res

def quat2rpy(quat):
    # quat: unit quaternion. Nx4. qw >= 0. q[0] = qw
    # rpy: roll, pitch, yaw. Nx3, deg
    q0 = np.expand_dims(quat[:, 0], axis=1)
    q1 = np.expand_dims(quat[:, 1], axis=1)
    q2 = np.expand_dims(quat[:, 2], axis=1)
    q3 = np.expand_dims(quat[:, 3], axis=1)

    r31 = 2 * (np.multiply(q1, q2) + np.multiply(q0, q3))
    r32 = np.square(q0) + np.square(q1) - np.square(q2) - np.square(q3)

    r21 = 2 * (np.multiply(q0, q2) - np.multiply(q1, q3))

    r11 = 2 * (np.multiply(q0, q1) + np.multiply(q2, q3))
    r12 = np.square(q0) - np.square(q1) - np.square(q2) + np.square(q3)

    rpy = np.concatenate((np.arctan2(r11, r12), np.arcsin(r21), np.arctan2(r31, r32)), axis=1) * 180 / np.pi

    for k, tmp_rpy in enumerate(rpy):
        tmp_idx = np.where(rpy[k, :] < -180)
        rpy[k, tmp_idx] = rpy[k, tmp_idx] +360
        tmp_idx = np.where(rpy[k, :] >= 180)
        rpy[k, tmp_idx] = rpy[k, tmp_idx] - 360

    return rpy

def rpy2quat(rpy):
    # rpy: roll, pitch, yaw. Nx3, deg
    # quat: unit quaternion. Nx4. qw >= 0. q[0] = qw

    cang = np.cos(rpy / (2*180) * np.pi)
    sang = np.sin(rpy / (2*180) * np.pi)

    cang0 = np.expand_dims(cang[:, 0], axis=1)
    cang1 = np.expand_dims(cang[:, 1], axis=1)
    cang2 = np.expand_dims(cang[:, 2], axis=1)
    sang0 = np.expand_dims(sang[:, 0], axis=1)
    sang1 = np.expand_dims(sang[:, 1], axis=1)
    sang2 = np.expand_dims(sang[:, 2], axis=1)

    quat = np.zeros((rpy.shape[0], 4))
    quat = np.concatenate((np.multiply(np.multiply(cang0, cang1), cang2) + np.multiply(np.multiply(sang0, sang1), sang2),
                    np.multiply(np.multiply(sang0, cang1), cang2) - np.multiply(np.multiply(cang0, sang1), sang2),
                    np.multiply(np.multiply(cang0, sang1), cang2) + np.multiply(np.multiply(sang0, cang1), sang2),
                    np.multiply(np.multiply(cang0, cang1), sang2) - np.multiply(np.multiply(sang0, sang1), cang2)),
                   axis=1)

    quat_sign = np.tile(np.sign(quat[:, 0]).reshape(-1, 1), (1, 4))
    quat = np.multiply(quat_sign, quat)
    quat = np.divide(quat, np.tile(np.linalg.norm(quat, axis=1).reshape(-1, 1), (1, 4)))

    return quat

def rotm2quat(rotm):
    # rotm: rotation matrix. Nx3x3
    # quat: unit quaternion. Nx4. qw >= 0. q[0] = qw

    quat = np.zeros((rotm.shape[0], 4))
    for ii in range(0, rotm.shape[0]):
        tr = np.trace(rotm[ii, :, :])

        if tr > 0:
            sqrt_p1 = np.sqrt(tr + 1.0)
            quat[ii, 0] = 0.5 * sqrt_p1
            quat[ii, 1] = (rotm[ii, 2, 1] - rotm[ii, 1, 2]) / (2.0 * sqrt_p1)
            quat[ii, 2] = (rotm[ii, 0, 2] - rotm[ii, 2, 0]) / (2.0 * sqrt_p1)
            quat[ii, 3] = (rotm[ii, 1, 0] - rotm[ii, 0, 1]) / (2.0 * sqrt_p1)
        else:
            d = rotm[ii, :, :].diagonal()
            if (d[1] > d[0]) and (d[1] > d[2]): # max value at rotm[ii, 1, 1]
                sqdip1 = np.sqrt(d[1] - d[0] - d[2] + 1.0)
                quat[ii, 2] = 0.5 * sqdip1
                if sqdip1 != 0:
                    sqdip1 = 0.5 / sqdip1

                quat[ii, 0] = (rotm[ii, 0, 2] - rotm[ii, 2, 0]) * sqdip1
                quat[ii, 1] = (rotm[ii, 1, 0] + rotm[ii, 0, 1]) * sqdip1
                quat[ii, 3] = (rotm[ii, 2, 1] + rotm[ii, 1, 2]) * sqdip1
            elif d[2] > d[0]: # max value at rotm[ii, 2, 2]
                sqdip1 = np.sqrt(d[2] - d[0] - d[1] + 1.0)
                quat[ii, 3] = 0.5 * sqdip1
                if sqdip1 != 0:
                    sqdip1 = 0.5 / sqdip1

                quat[ii, 0] = (rotm[ii, 1, 0] - rotm[ii, 0, 1]) * sqdip1
                quat[ii, 1] = (rotm[ii, 0, 2] + rotm[ii, 2, 0]) * sqdip1
                quat[ii, 2] = (rotm[ii, 2, 1] + rotm[ii, 1, 2]) * sqdip1
            else: # max value at rotm[ii, 0, 0]
                sqdip1 = np.sqrt(d[0] - d[1] - d[2] + 1.0)
                quat[ii, 1] = 0.5 * sqdip1
                if sqdip1 != 0:
                    sqdip1 = 0.5 / sqdip1

                quat[ii, 0] = (rotm[ii, 2, 1] - rotm[ii, 1, 2]) * sqdip1
                quat[ii, 2] = (rotm[ii, 1, 0] + rotm[ii, 0, 1]) * sqdip1
                quat[ii, 3] = (rotm[ii, 0, 2] + rotm[ii, 2, 0]) * sqdip1


    tmp = quat[:, 0]<0
    quat[tmp, :] = -quat[tmp, :]
    quat = np.divide(quat, np.linalg.norm(quat, axis=1))

    return quat

def rotm2quat2(rotm):
    # rotm: rotation matrix. Nx3x3
    # quat: unit quaternion. Nx4. q[0] = qw

    m00 = rotm[0, 0]
    m01 = rotm[0, 1]
    m02 = rotm[0, 2]

    m10 = rotm[1, 0]
    m11 = rotm[1, 1]
    m12 = rotm[1, 2]

    m20 = rotm[2, 0]
    m21 = rotm[2, 1]
    m22 = rotm[2, 2]

    if m00+m11+m22 > 0:
        s = np.sqrt(1.0+m00+m11+m22)*2
        qw = 0.25*s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00>m11) & (m00> m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22 :
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s

    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz])

    return quat




def quat2rotm(quat):
    # quat: unit quaternion. Nx4. qw >= 0. q[0] = qw
    # rotm: rotation matrix. Nx3x3

    rotm = np.zeros((quat.shape[0], 3, 3))

    rotm[:, 0, 0] = np.square(quat[:, 0]) + np.square(quat[:, 1]) - np.square(quat[:, 2]) - np.square(quat[:, 3])
    rotm[:, 0, 1] = 2 * (np.multiply(quat[:, 1], quat[:, 2]) - np.multiply(quat[:, 0], quat[:, 3]))
    rotm[:, 0, 2] = 2 * (np.multiply(quat[:, 1], quat[:, 3]) + np.multiply(quat[:, 0], quat[:, 2]))
    rotm[:, 1, 0] = 2 * (np.multiply(quat[:, 1], quat[:, 2]) + np.multiply(quat[:, 0], quat[:, 3]))
    rotm[:, 1, 1] = np.square(quat[:, 0]) - np.square(quat[:, 1]) + np.square(quat[:, 2]) - np.square(quat[:, 3])
    rotm[:, 1, 2] = 2 * (np.multiply(quat[:, 2], quat[:, 3]) - np.multiply(quat[:, 0], quat[:, 1]))
    rotm[:, 2, 0] = 2 * (np.multiply(quat[:, 1], quat[:, 3]) - np.multiply(quat[:, 0], quat[:, 2]))
    rotm[:, 2, 1] = 2 * (np.multiply(quat[:, 2], quat[:, 3]) + np.multiply(quat[:, 0], quat[:, 1]))
    rotm[:, 2, 2] = np.square(quat[:, 0]) - np.square(quat[:, 1]) - np.square(quat[:, 2]) + np.square(quat[:, 3])

    return rotm


def matrix_to_pose(T):
    # T: transformation matrix, Nx4x4
    # p: pose, Nx7, (x, y, z, qw, q0, q1, q2)

    p = np.zeros((T.shape[0], 7))
    R = T[:, 0:3, 0:3]
    t = T[:, 0:3, 3]
    p[:, 0:3] = np.matmul(-np.transpose(R, axes=[0, 2, 1]), np.expand_dims(t, axis=2)).squeeze()
    p[:, 3:7] = rotm2quat(np.transpose(R, axes=[0, 2, 1]))

    return p


def pose_to_matrix(p):
    # p: pose, Nx7, (x, y, z, qw, q0, q1, q2)
    # T: transformation matrix, Nx4x4

    T = np.zeros((p.shape[0], 4, 4))
    T[:, 3, 3] = 1
    R = quat2rotm(p[:, 3:7])
    t = p[:, 0:3]
    T[:, 0:3, 3] = np.matmul(-np.transpose(R, axes=[0, 2, 1]), np.expand_dims(t, axis=2)).squeeze()
    T[:, 0:3, 0:3] = np.transpose(R, axes=[0, 2, 1])

    return T