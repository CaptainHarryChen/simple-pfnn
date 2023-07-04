import MoCCASimuBackend
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
import numpy as np


bvh_data_list = [
    './motion_material/kinematic_motion/long_walk_.bvh',
    './motion_material/kinematic_motion/long_walk_mirror_.bvh',
    './motion_material/kinematic_motion/long_run_.bvh',
    './motion_material/kinematic_motion/long_run_mirror_.bvh'
]

delta_frame = 20
lookahead_frame = 100
frame_time = 0.016667


def calc_rotation(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    rotation_axis = np.cross(A, B)
    if np.linalg.norm(rotation_axis) < 1e-8:
        if np.dot(A, B) < 0:
            rotation_axis = np.cross(A, [0,0,1])
        else:
            return R.identity()
    cos_angle = np.dot(A, B)
    sin_angle = np.linalg.norm(rotation_axis)
    angle = np.arctan2(sin_angle, cos_angle)
    rot = R.from_rotvec(angle * rotation_axis / sin_angle)
    return rot


def calc_phase(joint_translation, joint_orientation, threshold = 0.01, window_half_size = 20, eps = 1e-4):
    ph = []

    # lAnkle: 03    rAnkle: 06  
    left_v = np.linalg.norm(joint_translation[1:, 3] - joint_translation[:-1, 3], axis=1)
    right_v = np.linalg.norm(joint_translation[1:, 6] - joint_translation[:-1, 6], axis=1)
    compare = left_v <= right_v + eps

    for i in range(left_v.shape[0]):
        wl = max(i-window_half_size, 0)
        wr = min(i+window_half_size+1, left_v.shape[0])
        if left_v[i] <= right_v[i] + eps and left_v[i] < threshold:
            window = left_v[wl:wr]
            window = window[compare[wl:wr]]
            if np.min(window) == left_v[i]:
                if len(ph) != 0 and ph[-1][1] == 0.5:
                    ph.append((i, 1.0))
                    ph.append((i + 1, 0.0))
                else:
                    ph.append((i,0.))
        elif left_v[i] > right_v[i] and right_v[i] < threshold:
            window = right_v[wl:wr]
            window = window[np.logical_not(compare[wl:wr])]
            if np.min(window) == right_v[i]:
                ph.append((i,0.5))
    ph = np.array(ph)
    phase = np.interp(np.arange(len(joint_translation)), ph[:,0].flatten(), ph[:,1].flatten())

    # import matplotlib.pyplot as plt
    # x = range(10000)
    # y = phase[:10000]
    # y2 = left_v[:10000] * 100 + 1
    # y3 = right_v[:10000] * 100 + 1
    # plt.plot(x,y2)
    # plt.plot(x,y3)
    # plt.plot(x,y, color="red")
    # plt.show()
    return phase


def main():
    sceneloader = MoCCASimuBackend.ODESim.JsonSceneLoader()
    scene = sceneloader.load_from_file('stdhuman-scene.json')
    character = scene.character0
    joint_name = ['RootJoint'] + character.joint_info.joint_names()

    data_ph = []
    data_in = []
    data_out = []
    for bvh_file_name in bvh_data_list:
        # 读取motion数据
        motion = BVHMotion(bvh_file_name=bvh_file_name)
        motion.adjust_joint_name(joint_name)

        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        # 计算相位
        phases = calc_phase(joint_translation, joint_orientation)

        # 根节点的位置和旋转（通过腰和肩计算旋转）
        # lHip:1   rHip:4   lShoulder:11   rShoulder:14
        across = (joint_translation[:, 11] - joint_translation[:, 14]) + (joint_translation[:, 1] - joint_translation[:, 4])
        across = across / np.linalg.norm(across, axis=-1, keepdims=True)
        forward = np.cross(across, np.array([[0,1,0]]))
        forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)

        def calc_dir(x):
            return calc_rotation(np.array([0,0,1]), x).as_quat()
        root_rotation = np.apply_along_axis(calc_dir, -1, forward)
        root_position = joint_translation[:, 0]
        root_rotation_inv = R.from_quat(root_rotation).inv()

        # 划窗口并构造数据
        for i in range(1, motion.motion_length - lookahead_frame - delta_frame):
            inv_rot = root_rotation_inv[i]
            window = list(range(i + delta_frame, i + lookahead_frame + delta_frame, delta_frame))
            desired_root_pos = inv_rot.apply(root_position[window] - root_position[i])
            desired_root_dir = inv_rot.apply(forward[window])
            desired_root_pos = desired_root_pos[:, [0,2]]
            desired_root_dir = desired_root_dir[:, [0,2]]
            desired_root_dir = desired_root_dir / np.linalg.norm(desired_root_dir, axis=-1, keepdims=True)

            local_pos = joint_translation[i].copy()
            local_pos[:, 0] -= local_pos[0][0]
            local_pos[:, 2] -= local_pos[0][2]
            local_pos = inv_rot.apply(local_pos)
            local_vel = inv_rot.apply(joint_translation[i] - joint_translation[i-1])

            data_ph.append(phases[i])

            data_in.append(np.hstack([
                desired_root_pos.ravel(), # 0 ~ 9
                desired_root_dir.ravel(), # 10 ~ 19
                local_pos.ravel(), # 20 ~ 67
                local_vel.ravel(), # 68 ~ 115
                ]))

            next_root_pos = inv_rot.apply(root_position[i+1] - root_position[i])
            next_root_dir = inv_rot.apply(forward[i+1])
            next_root_pos = next_root_pos[[0,2]]
            next_root_dir = next_root_dir[[0,2]]
            next_root_dir = next_root_dir / np.linalg.norm(next_root_dir, axis=-1, keepdims=True)
            dphase = phases[i+1] - phases[i]
            if dphase < 0:
                dphase += 1
            next_local_rotation = (root_rotation_inv[i+1] * R.from_quat(joint_orientation[i+1])).as_quat()
            # next_local_rotation = joint_orientation[i+1]

            data_out.append(np.hstack([
                next_root_pos.ravel(), # 0 ~ 1
                next_root_dir.ravel(), # 2 ~ 3
                dphase, # 4 
                next_local_rotation.ravel() # 5 ~ 68
                ]))
    
    data_ph = np.array(data_ph)
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    in_mean, in_std = data_in.mean(axis=0), data_in.std(axis=0)
    # 
    in_std[0:10:2] = in_std[0:10:2].mean()
    in_std[1:11:2] = in_std[1:11:2].mean()
    in_std[10:20:2] = in_std[10:20:2].mean()
    in_std[11:21:2] = in_std[11:21:2].mean()
    in_std[20:68:2] = in_std[20:68:2].mean()
    in_std[21:69:2] = in_std[21:69:2].mean()
    in_std[69:116:2] = in_std[69:116:2].mean()
    in_std[70:117:2] = in_std[70:117:2].mean()
    data_in = (data_in - in_mean) / in_std

    out_mean, out_std = data_out.mean(axis=0), data_out.std(axis=0)
    out_std[0:2:2] = out_std[0:2:2].mean()
    out_std[1:3:2] = out_std[1:3:2].mean()
    out_std[2:4:2] = out_std[2:4:2].mean()
    out_std[3:5:2] = out_std[3:5:2].mean()
    out_std[5:69:2] = out_std[5:69:2].mean()
    out_std[6:70:2] = out_std[6:70:2].mean()
    data_out = (data_out - out_mean) / out_std

    np.savez("processed_data.npz", data_ph=data_ph, data_in=data_in, data_out=data_out
        ,in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std)


if __name__ == "__main__":
    main()
