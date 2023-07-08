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
window_size = 60
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

    # lAnkle: 08    rAnkle: 07  
    left_v = np.linalg.norm(joint_translation[1:, 8] - joint_translation[:-1, 8], axis=1)
    right_v = np.linalg.norm(joint_translation[1:, 7] - joint_translation[:-1, 7], axis=1)
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
        phase = calc_phase(joint_translation, joint_orientation)

        # 根节点的位置和旋转（通过腰和肩计算旋转）
        # lHip:4   rHip:3   lShoulder:13   rShoulder:12
        across = (joint_translation[:, 13] - joint_translation[:, 12]) + (joint_translation[:, 4] - joint_translation[:, 3])
        across = across / np.linalg.norm(across, axis=-1, keepdims=True)
        forward = np.cross(across, np.array([[0,1,0]]))
        forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)

        def calc_dir(x):
            return calc_rotation(np.array([0,0,1]), x).as_quat()
        root_rotation = np.apply_along_axis(calc_dir, -1, forward)
        root_rotation_rep = R.from_quat(np.tile(root_rotation, reps=[1, joint_translation.shape[1]]).reshape(-1,4))
        root_position = joint_translation[:, 0]

        local_positions = joint_translation.copy()
        local_positions[:, :, 0] -= local_positions[:, 0:1, 0]
        local_positions[:, :, 2] -= local_positions[:, 0:1, 2]
        local_positions = root_rotation_rep.apply(local_positions.reshape(-1, 3), inverse=True).reshape(-1, joint_translation.shape[1], 3)
        local_velocities = joint_translation[1:] - joint_translation[:-1]
        local_velocities = root_rotation_rep[:-joint_translation.shape[1]].apply(local_velocities.reshape(-1, 3), inverse=True).reshape(-1, joint_translation.shape[1], 3)
        local_rotations = R.as_euler(root_rotation_rep.inv() * R.from_quat(joint_orientation.reshape(-1, 4)), "XYZ").reshape(-1, joint_translation.shape[1], 3)

        root_velocity = R.from_quat(root_rotation[:-1]).apply(joint_translation[1:,0] - joint_translation[:-1, 0], inverse=True)
        root_rvelocity = R.as_euler(R.from_quat(root_rotation[:-1]).inv() * R.from_quat(root_rotation[1:]), "XYZ")[:,1]

        dphase = phase[1:] - phase[:-1]
        dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]

        # 划窗口并构造数据 !!!!!!!!!!!!!!!!!!!!!
        for i in range(window_size, motion.motion_length - window_size - 1):
            rootposs = R.from_quat(root_rotation[i]).apply(joint_translation[i-window_size:i+window_size:delta_frame,0] - joint_translation[i,0], inverse=True)
            rootdirs = R.from_quat(root_rotation[i]).apply(forward[i-window_size:i+window_size:delta_frame])
    
            data_ph.append(phase[i])

            data_in.append(np.hstack([
                rootposs[:, 0].ravel(), rootposs[:, 2].ravel(), # 0~5, 6~11
                rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(), # 12~17, 18~23
                local_positions[i-1].ravel(), # 24~71
                local_velocities[i-1].ravel(), # 72~119
                ]))

            rootposs_next = R.from_quat(root_rotation[i+1]).apply(joint_translation[i+1:i+window_size+1:delta_frame,0] - joint_translation[i+1,0], inverse=True)
            rootdirs_next = R.from_quat(root_rotation[i+1]).apply(forward[i+1:i+window_size+1:delta_frame])

            data_out.append(np.hstack([
                root_velocity[i,0].ravel(), root_velocity[i,2].ravel(), # 0,1
                root_rvelocity[i].ravel(), # 2
                dphase[i], # 3
                rootposs_next[0].ravel(), rootposs_next[2].ravel(), # 4~6, 7~9
                rootdirs_next[0].ravel(), rootdirs_next[2].ravel(), # 10~12, 13~15
                local_positions[i].ravel(), # 16~63
                local_velocities[i].ravel(), # 64~111
                local_rotations[i].ravel() # 112~159
                ]))
    
    data_ph = np.array(data_ph)
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    in_mean, in_std = data_in.mean(axis=0), data_in.std(axis=0)
    in_std[0:6] = in_std[0:6].mean()
    in_std[6:12] = in_std[6:12].mean()
    in_std[12:18] = in_std[12:18].mean()
    in_std[18:24] = in_std[18:24].mean()
    in_std[24:72] = in_std[24:72].mean()
    in_std[72:120] = in_std[72:120].mean()
    data_in = (data_in - in_mean) / in_std

    out_mean, out_std = data_out.mean(axis=0), data_out.std(axis=0)
    out_std[4:10] = out_std[4:10].mean()
    out_std[10:16] = out_std[10:16].mean()
    out_std[16:112] = out_std[16:112].mean()
    out_std[112:160] = out_std[112:160].mean()
    data_out = (data_out - out_mean) / out_std

    np.savez("processed_data.npz", data_ph=data_ph, data_in=data_in, data_out=data_out
        ,in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std)


if __name__ == "__main__":
    main()
