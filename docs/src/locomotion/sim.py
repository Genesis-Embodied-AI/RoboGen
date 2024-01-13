import os
import pybullet as p
import pybullet_data as pd
import numpy as np
from gym import spaces
import gym
from cem_policy.utils import save_env

class SimpleEnv(gym.Env):
    def __init__(self, dt=0.01, gui=True, task='', robot_name='', frameskip=10, frameskip_save=2, horizon=50):
        self.gui = gui

        self.task = task
        self.robot_name = robot_name
        self.frameskip = frameskip
        self.frameskip_save = frameskip_save
        self.horizon = horizon
        self.gain = 0.03

        self.gravity = -9.81

        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
                
        self.set_scene()
        self.setup_camera_rpy()

        self.action_low = -np.ones(self.n_joints)
        self.action_high = np.ones(self.n_joints)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_joints * 2 + 6, ), dtype=np.float32) 

        self.reset()

    def set_scene(
        self,
    ):
        p.resetSimulation(physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)

        planeId = p.loadURDF("plane.urdf", physicsClientId=self.id)

        self.urdf_ids = {
            "plane": planeId,
        }
        self.urdf_paths = {
            "plane": "plane.urdf",
        }
        self.urdf_scales = {
            "plane": 1.0,
        }

        self.init_robot()

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        self.init_state = p.saveState(physicsClientId=self.id)

        self.update_robot_ref()

    def init_robot(self):
        self.urdf_scales['robot'] = 1.0
        if self.robot_name == 'a1':
            self.urdf_paths['robot'] = os.path.join(os.path.dirname(__file__), "a1/a1.urdf")
            init_pos = [0, 0, 0.5]
            self.robot_id = p.loadURDF(self.urdf_paths['robot'], init_pos, physicsClientId=self.id)
            self.urdf_ids["robot"] = self.robot_id

            A1_DEFAULT_ABDUCTION_ANGLE = 0
            A1_DEFAULT_HIP_ANGLE = 0.9
            A1_DEFAULT_KNEE_ANGLE = -1.8
            NUM_LEGS = 4
            INIT_MOTOR_ANGLES = np.array([
                A1_DEFAULT_ABDUCTION_ANGLE,
                A1_DEFAULT_HIP_ANGLE,
                A1_DEFAULT_KNEE_ANGLE
            ] * NUM_LEGS)

            self.MOTOR_NAMES = [
                "FR_hip_joint",
                "FR_upper_joint",
                "FR_lower_joint",
                "FL_hip_joint",
                "FL_upper_joint",
                "FL_lower_joint",
                "RR_hip_joint",
                "RR_upper_joint",
                "RR_lower_joint",
                "RL_hip_joint",
                "RL_upper_joint",
                "RL_lower_joint",
            ]
        elif self.robot_name == 'atlas':
            self.urdf_paths['robot'] = os.path.join(os.path.dirname(__file__), "atlas/atlas_v4_with_multisense.urdf")
            self.robot_id = p.loadURDF(self.urdf_paths['robot'], [0,0,1.2], physicsClientId=self.id)
            self.urdf_ids["robot"] = self.robot_id

            self.MOTOR_NAMES = [
                'back_bkz',
                'back_bky',
                'back_bkx',
                'l_arm_shz',
                'l_arm_shx',
                'l_arm_ely',
                'l_arm_elx',
                'l_arm_wry',
                'l_arm_wrx',
                'l_arm_wry2',
                'neck_ry',
                'r_arm_shz',
                'r_arm_shx',
                'r_arm_ely',
                'r_arm_elx',
                'r_arm_wry',
                'r_arm_wrx',
                'r_arm_wry2',
                'l_leg_hpz',
                'l_leg_hpx',
                'l_leg_hpy',
                'l_leg_kny',
                'l_leg_aky',
                'l_leg_akx',
                'r_leg_hpz',
                'r_leg_hpx',
                'r_leg_hpy',
                'r_leg_kny',
                'r_leg_aky',
                'r_leg_akx',
            ]

            INIT_MOTOR_ANGLES = np.zeros(len(self.MOTOR_NAMES))

        elif self.robot_name == 'anymal':
            self.urdf_paths['robot'] = os.path.join(os.path.dirname(__file__), "anymal/anymal.urdf")
            self.robot_id = p.loadURDF(self.urdf_paths['robot'], [0,0,1.0], physicsClientId=self.id)
            self.urdf_ids["robot"] = self.robot_id

            self.MOTOR_NAMES = [
                'LF_HAA',
                'LF_HFE',
                'LF_KFE',
                'RF_HAA',
                'RF_HFE',
                'RF_KFE',
                'LH_HAA',
                'LH_HFE',
                'LH_KFE',
                'RH_HAA',
                'RH_HFE',
                'RH_KFE',
            ]
            INIT_MOTOR_ANGLES = np.zeros(len(self.MOTOR_NAMES))
        else:
            assert False

        self.joint_ids = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.n_joints = len(self.MOTOR_NAMES)
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.id)):
            joint_info = p.getJointInfo(self.robot_id, j, physicsClientId=self.id)
            name = joint_info[1].decode('utf-8')
            if name in self.MOTOR_NAMES:
                self.joint_ids.append(j)
                if self.robot_name == 'anymal':
                    self.joint_limits_lower.append(max(-2.0, joint_info[8]))
                    self.joint_limits_upper.append(min(2.0, joint_info[9]))
                else:
                    self.joint_limits_lower.append(joint_info[8])
                    self.joint_limits_upper.append(joint_info[9])
        self.joint_limits_lower = np.array(self.joint_limits_lower)
        self.joint_limits_upper = np.array(self.joint_limits_upper)
        for index in range(self.n_joints):
            joint_id = self.joint_ids[index]
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, INIT_MOTOR_ANGLES[index], physicsClientId=self.id)
            p.resetJointState(self.robot_id, joint_id, INIT_MOTOR_ANGLES[index], physicsClientId=self.id)


    def update_robot_ref(self):
        COM_pos, COM_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.id)
        self.COM_init_pos = np.array(COM_pos)

    def reset(self):
        p.restoreState(self.init_state, physicsClientId=self.id)
        self.time_step = 0
        self.done = False
        return self._get_obs()

    def step(self, action):
        self.time_step += 1
        self.take_step(action)
        obs = self._get_obs()
        done, info = self._get_done_info()
        reward = self._compute_reward() ### NOTE: to be implemented by gpt-4
        success = False
        self.success = success
        return obs, reward, done, info

    def step_(self, action):
        self.time_step += 1

        self.act(action)
        rgbs = []
        states = []
        for i in range(self.frameskip):
            p.stepSimulation(physicsClientId=self.id)       
            if (i+1) % self.frameskip_save == 0:
                rgb, depth = self.render()
                rgbs.append(rgb)
                states.append(save_env(self))

        obs = self._get_obs()
        done, info = self._get_done_info()
        reward = self._compute_reward() ### NOTE: to be implemented by gpt-4
        return obs, reward, done, info, rgbs, states

    def take_step(self, action):

        self.act(action)

        for _ in range(self.frameskip):
            p.stepSimulation(physicsClientId=self.id)       

    def act(self, action):
        action = np.clip(action, self.action_low, self.action_high)
        action = (action - self.action_low) / (self.action_high - self.action_low) * (self.joint_limits_upper - self.joint_limits_lower) + self.joint_limits_lower

        for index in range(self.n_joints):
            joint_id = self.joint_ids[index]
            joint_name = self.MOTOR_NAMES[index]
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, action[index], positionGain=self.gain, physicsClientId=self.id)

    def _compute_reward(self):
        return 0

    def setup_camera_rpy(self, camera_target=[0, 0, 0.3], distance=1.6, rpy=[0, -30, -30], fov=60, camera_width=640, camera_height=480):
        distance=2.0
        if self.robot_name == 'atlas':
            distance = 2.5

        self.camera_width = camera_width
        self.camera_height = camera_height
        camera_target = np.array([0, 0, 0.3])

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def render(self, mode=None):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, img, depth, segmask = p.getCameraImage(self.camera_width, self.camera_height, 
            self.view_matrix, self.projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL, 
            physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))[:, :, :3]
        depth = np.reshape(depth, (h, w))

        return img, depth

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0])
            
        cnt = 0
        for name, id in self.urdf_ids.items():
            if name == 'robot':
                pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                euler_angle = p.getEulerFromQuaternion(orient)
                obs[cnt:cnt+3] = [0, 0, pos[2]]
                obs[cnt+3:cnt+6] = euler_angle
                cnt += 6
                for joint_id in self.joint_ids:
                    joint_state = p.getJointState(id, joint_id, physicsClientId=self.id)
                    obs[cnt] = joint_state[0]
                    obs[cnt + 1] = joint_state[1]
                    cnt += 2
        return obs

    def _get_done_info(self):
        info = {}
        if not self.done:
            if self.time_step >= self.horizon:
                self.done = True
                info['timeout'] = True


        return self.done, info

    def disconnect(self):
        p.disconnect(self.id)

    def close(self):
        p.disconnect(self.id)

