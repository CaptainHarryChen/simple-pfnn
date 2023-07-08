# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from model import SimplePFNN
from preprocess import calc_rotation
from dataloader import DataSet, DataLoader
import MoCCASimuBackend

class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
        # motion
        self.motions = []

        sceneloader = MoCCASimuBackend.ODESim.JsonSceneLoader()
        scene = sceneloader.load_from_file('stdhuman-scene.json')
        character = scene.character0
        joint_name = ['RootJoint'] + character.joint_info.joint_names()
        # 添加motion
        self.motions.append(BVHMotion(bvh_file_name='./motion_material/kinematic_motion/long_walk_.bvh'))
        self.motions[0].adjust_joint_name(joint_name)
        joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics(frame_id_list=[0], root_pos=np.array([0,0.918307,0]))

        self.window_size = 60 + 1
        
        self.cur_phase = 0.

        self.his_root_pos = np.tile(joint_translation[0][0], self.window_size).reshape(self.window_size, 3)
        self.his_root_dir = np.zeros_like(self.his_root_pos)
        self.his_root_dir[:, 2] = 1.
        
        self.cur_joint_translation = joint_translation[0].copy()
        self.cur_local_pos = joint_translation[0].copy()
        self.cur_local_pos[:, 0] -= self.cur_local_pos[0, 0]
        self.cur_local_pos[:, 2] -= self.cur_local_pos[0, 2]
        self.cur_local_vel = np.zeros_like(self.cur_local_pos)

        # 调试用，直接输出数据中的标签
        self.dataset = DataSet("processed_data.npz")
        self.cur_data_idx = 0

        self.model = SimplePFNN(120, 160)
        self.model.load("checkpoints/model_4.npz")

        with np.load("processed_data.npz") as f:
            self.in_mean = f["in_mean"]
            self.in_std = f["in_std"]
            self.out_mean = f["out_mean"]
            self.out_std = f["out_std"]

    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
                     ):
        '''
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
            当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
            desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
            desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
            desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
            desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
        Output: 输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义

        '''
        rootposs = np.concatenate([self.his_root_pos[0::20], desired_pos_list[1:3]], axis=0)
        rootdirs = np.concatenate([self.his_root_dir[0::20], R.from_quat(desired_rot_list[1:3]).apply(np.array([0,0,1]))], axis=0)

        root_position = self.his_root_pos[-1]
        root_rotation = calc_rotation(np.array([0,0,1]), rootdirs[-1])

        rootposs -= root_position
        rootdirs = root_rotation.apply(rootdirs, inverse=True)

        X = np.hstack([
                rootposs[:, 0].ravel(), rootposs[:, 2].ravel(), # 0~5, 6~11
                rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(), # 12~17, 18~23
                self.cur_local_pos.ravel(), # 24 ~ 71
                self.cur_local_vel.ravel(), # 62 ~ 119
                ]).reshape(1,-1)
        X = (X - self.in_mean) / self.in_std
        phase = np.array([[self.cur_phase]])
        
        # 调试用，直接输出数据中的标签
        # phase, X_, pred = self.dataset[self.cur_data_idx]
        # phase = phase.reshape(1,-1)
        # # print(np.linalg.norm(X-X_))
        # X = X_.reshape(1,-1)
        # _, __, pred = self.dataset[self.cur_data_idx]
        # self.cur_data_idx += 1

        # 使用模型预测
        # print(np.linalg.norm(X))
        pred = self.model(X, phase)[0]

        pred = pred * self.out_std + self.out_mean

        next_root_pos = pred[0:2]
        next_root_pos = np.array([next_root_pos[0], 0, next_root_pos[1]])
        next_root_pos = root_rotation.apply(next_root_pos)
        root_position += next_root_pos
        # root_position[[0,2]] = desired_pos_list[0][[0,2]]

        next_root_vel = pred[2]
        # print(next_root_vel)
        root_rotation = root_rotation * R.from_euler("XYZ", np.array([0,next_root_vel,0]))
        # root_rotation = R.from_quat(desired_rot_list[0])
        
        dphase = pred[3]
        self.cur_phase += dphase
        if self.cur_phase > 1:
            self.cur_phase -= 1
        
        next_local_position = pred[16:64].reshape(-1,3)
        next_local_velocity = pred[64:112].reshape(-1,3)
        next_local_rotation = R.from_euler("XYZ", pred[112:160].reshape(-1, 3))

        new_local_pos = 0.5 * (self.cur_local_pos + next_local_velocity) + 0.5 * next_local_position
        joint_translation = root_rotation.apply(new_local_pos)
        joint_translation[:, [0, 2]] += root_position[[0,2]]
        joint_orientation = (root_rotation * next_local_rotation).as_quat()

        self.cur_local_pos = joint_translation.copy()
        self.cur_local_pos[:, 0] -= self.cur_local_pos[0][0]
        self.cur_local_pos[:, 2] -= self.cur_local_pos[0][2]
        self.cur_local_pos = root_rotation.inv().apply(self.cur_local_pos)
        self.cur_local_vel = root_rotation.inv().apply(joint_translation - self.cur_joint_translation)
        self.cur_joint_translation = joint_translation

        self.his_root_pos[:-1] = self.his_root_pos[1:]
        self.his_root_pos[-1] = root_position
        self.his_root_dir[:-1] = self.his_root_dir[1:]
        self.his_root_dir[-1] = root_rotation.apply(np.array([0,0,1]))

        return self.motions[0].joint_name, joint_translation, joint_orientation
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，
        这里用一个简单的方案，将手柄的位置对齐于角色我位置
        '''
        controller_pos = character_state[1][0] 
        self.controller.set_pos(controller_pos)
    
