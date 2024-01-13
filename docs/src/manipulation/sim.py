import numpy as np
import pybullet as p
import gym
from gym.utils import seeding
from gym import spaces
import pickle
import yaml
import os.path as osp
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from manipulation.panda import Panda
from manipulation.ur5 import UR5
from manipulation.sawyer import Sawyer
from manipulation.utils import parse_config, load_env, download_and_parse_objavarse_obj_from_yaml_config
from manipulation.gpt_reward_api import get_joint_id_from_name, get_link_id_from_name

class SimpleEnv(gym.Env):
    def __init__(self, 
                    dt=0.01, 
                    config_path=None, 
                    gui=False, 
                    frameskip=2, 
                    horizon=120, 
                    restore_state_file=None, 
                    rotation_mode='delta-axis-angle-local',
                    translation_mode='delta-translation', 
                    max_rotation=np.deg2rad(5), 
                    max_translation=0.15,
                    use_suction=True,  # whether to use a suction gripper
                    object_candidate_num=6, # how many candidate objects to sample from objaverse
                    vhacd=False, # if to perform vhacd on the object for better collision detection for pybullet
                    randomize=0, # if to randomize the scene
                    obj_id=0, # which object to choose to use from the candidates
                ):
        
        super().__init__()
        
        # Task
        self.config_path = config_path
        self.restore_state_file = restore_state_file
        self.frameskip = frameskip
        self.horizon = horizon
        self.gui = gui
        self.object_candidate_num = object_candidate_num
        self.solution_path = None        
        self.success = False # not really used, keeped for now
        self.primitive_save_path = None # to be used for saving the primitives execution results
        self.randomize = randomize
        self.obj_id = obj_id # which object to choose to use from the candidates

        # physics
        self.gravity = -9.81
        self.contact_constraint = None
        self.vhacd = vhacd
        
        # action space
        self.use_suction = use_suction
        self.rotation_mode = rotation_mode
        self.translation_mode = translation_mode
        self.max_rotation_angle = max_rotation
        self.max_translation = max_translation
        self.suction_to_obj_pose = 0
        self.suction_contact_link = None
        self.suction_obj_id = None
        self.activated = 0
        
        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)
                
        self.asset_dir = osp.join(osp.dirname(osp.realpath(__file__)), "assets/")
        hz=int(1/dt)
        p.setTimeStep(1.0 / hz, physicsClientId=self.id)

        self.seed()
        self.set_scene()
        self.setup_camera_rpy()
        self.scene_lower, self.scene_upper = self.get_scene_bounds()
        self.scene_center = (self.scene_lower + self.scene_upper) / 2
        self.scene_range = (self.scene_upper - self.scene_lower) / 2

        self.grasp_action_mag = 0.06 if not self.use_suction else 1
        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, self.grasp_action_mag])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.base_action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.num_objects = len(self.urdf_ids) - 2 # exclude plane, robot
        distractor_object_num = np.sum(list(self.is_distractor.values()))
        self.num_objects -= distractor_object_num

        ### For RL policy learning, observation space includes:
        # 1. object positions and orientations (6 * num_objects)
        # 2. object min and max bounding box (6 * num_objects)
        # 3. articulated object joint angles (num_objects * num_joints) 
        # 4. articulated object link position and orientation (num_objects * num_joints * 6) 
        # 5. robot base position (xy)
        # 6. robot end-effector position and orientation (6)
        # 7. gripper suction activated/deactivate or gripper joint angle (if not using suction gripper) (1)
        num_obs = self.num_objects * 12 # obs 1 and 2
        for name in self.urdf_types:
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]: # obs 3 and 4
                num_joints = p.getNumJoints(self.urdf_ids[name], physicsClientId=self.id) 
                num_obs += num_joints
                num_obs += 6 * num_joints
        num_obs += 2 + 6 + 1 # obs 5 6 7
        self.base_num_obs = num_obs

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs, ), dtype=np.float32) 
        self.base_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_num_obs, ), dtype=np.float32)

        self.detected_position = {} # not used for now, keep it
        
    def normalize_position(self, pos):
        if self.translation_mode == 'normalized-direct-translation':
            return (pos - self.scene_center) / self.scene_range 
        else:
            return pos

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random()

    def get_aabb(self, id):
        num_joints = p.getNumJoints(id, physicsClientId=self.id)
        min_aabbs, max_aabbs = [], []
        for link_idx in range(-1, num_joints):
            min_aabb, max_aabb = p.getAABB(id, link_idx, physicsClientId=self.id)
            min_aabbs.append(list(min_aabb))
            max_aabbs.append(list(max_aabb))
        min_aabb = np.min(np.concatenate(min_aabbs, axis=0).reshape(-1, 3), axis=0)
        max_aabb = np.max(np.concatenate(max_aabbs, axis=0).reshape(-1, 3), axis=0)
        return min_aabb, max_aabb
    
    def get_aabb_link(self, id, link_id):
        min_aabb, max_aabb = p.getAABB(id, link_id, physicsClientId=self.id)
        return np.array(min_aabb), np.array(max_aabb)

    def get_scene_bounds(self):
        min_aabbs = []
        max_aabbs = []
        for name, id in self.urdf_ids.items():
            if name == 'plane': continue
            min_aabb, max_aabb = self.get_aabb(id)
            min_aabbs.append(min_aabb)
            max_aabbs.append(max_aabb)
        
        min_aabb = np.min(np.stack(min_aabbs, axis=0).reshape(-1, 3), axis=0)
        max_aabb = np.max(np.stack(max_aabbs, axis=0).reshape(-1, 3), axis=0)
        range = max_aabb - min_aabb
        return min_aabb - 0.5 * range, max_aabb + 0.5 * range

    def clip_within_workspace(self, robot_pos, ori_pos, on_table):
        pos = ori_pos.copy()
        if not on_table:
            # If objects are too close to the robot, push them away
            x_near_low, x_near_high = robot_pos[0] - 0.3, robot_pos[0] + 0.3
            y_near_low, y_near_high = robot_pos[1] - 0.3, robot_pos[1] + 0.3

            if pos[0] > x_near_low and pos[0] < x_near_high:
                pos[0] = x_near_low if pos[0] < robot_pos[0] else x_near_high

            if pos[1] > y_near_low and pos[1] < y_near_high:
                pos[1] = y_near_low if pos[1] < robot_pos[1] else y_near_high
            return pos
        else:
            # Object is on table, should be within table's bounding box
            new_pos = pos.copy()
            new_pos[:2] = np.clip(new_pos[:2], self.table_bbox_min[:2], self.table_bbox_max[:2])
            return new_pos

    def get_robot_base_pos(self):
        robot_base_pos = [1, 1, 0.28]
        return robot_base_pos
    
    def get_robot_init_joint_angles(self):
        init_joint_angles = [0 for _ in range(len(self.robot.right_arm_joint_indices))]
        if self.robot_name == 'panda':
            init_joint_angles = [0, -1.10916842e-04,  7.33823451e-05, -5.47701370e-01, -5.94950533e-01,
                2.62857916e+00, -4.85316284e-01,  1.96042022e+00,  2.15271531e+00,
                -7.35304443e-01]
        return init_joint_angles

    def set_scene(
        self,
    ):
        ### simulation preparation
        p.resetSimulation(physicsClientId=self.id)
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)

        ### load restore state
        restore_state = None
        if self.restore_state_file is not None:
            with open(self.restore_state_file, 'rb') as f:
                restore_state = pickle.load(f)
        
        ### load plane 
        planeId = p.loadURDF(osp.join(self.asset_dir, "plane", "plane.urdf"), physicsClientId=self.id)

        ### create and load a robot
        robot_base_pos = self.load_robot(restore_state)

        ### load and parse task config (including semantically meaningful distractor objects)
        self.urdf_ids = {
            "robot": self.robot.body,
            "plane": planeId,
        }
        self.urdf_paths = {}
        self.urdf_types = {}
        self.init_positions = {}
        self.on_tables = {}
        self.simulator_sizes = {}
        self.is_distractor = {
            "robot": 0,
            "plane": 0,
        }
        urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables, \
            use_table, articulated_init_joint_angles, spatial_relationships = self.load_and_parse_config(restore_state)

        ### handle the case if there is a table
        self.load_table(use_table, restore_state)

        ### load each object from the task config
        self.load_object(urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables)

        ### adjusting object positions
        ### place the lowest point on the object to be the height where GPT specifies
        object_height = self.adjust_object_positions(robot_base_pos)

        ### resolve collisions between objects
        self.resolve_collision(robot_base_pos, object_height, spatial_relationships)

        ### handle any special relationships outputted by GPT
        self.handle_gpt_special_relationships(spatial_relationships)

        ### set all object's joint angles to the lower joint limit
        self.set_to_default_joint_angles()

        ### overwrite joint angles specified by GPT
        self.handle_gpt_joint_angle(articulated_init_joint_angles)
           
        ### record initial joint angles and positions
        self.record_initial_joint_and_pose()
        
        ### stabilize the scene
        for _ in range(500):
            p.stepSimulation(physicsClientId=self.id)

        ### restore to a state if provided
        if self.restore_state_file is not None:
            load_env(self, self.restore_state_file)

        ### Enable debug rendering
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
 
        self.init_state = p.saveState(physicsClientId=self.id)
        
        
    def load_robot(self, restore_state):
        robot_classes = {
            "panda": Panda,
            "sawyer": Sawyer,
            "ur5": UR5,
        }
        robot_names = list(robot_classes.keys())
        self.robot_name = robot_names[np.random.randint(len(robot_names))]
        if restore_state is not None and "robot_name" in restore_state:
            self.robot_name = restore_state['robot_name']
        self.robot_class = robot_classes[self.robot_name]
      
        # Create robot
        self.robot = self.robot_class()
        self.robot.init(self.asset_dir, self.id, self.np_random, fixed_base=True, use_suction=self.use_suction)
        self.agents = [self.robot]
        self.suction_id = self.robot.right_gripper_indices[0]

        # Update robot motor gains
        self.robot.motor_gains = 0.05
        self.robot.motor_forces = 100.0

        # Set robot base position & orientation, and joint angles
        robot_base_pos = self.get_robot_base_pos()
        robot_base_orient = [0, 0, 0, 1]
        self.robot_base_orient = robot_base_orient
        self.robot.set_base_pos_orient(robot_base_pos, robot_base_orient)
        init_joint_angles = self.get_robot_init_joint_angles()
        self.robot.set_joint_angles(self.robot.right_arm_joint_indices, init_joint_angles)    
        
        return robot_base_pos        
    
    def load_and_parse_config(self, restore_state):
        ### select and download objects from objaverse
        res = download_and_parse_objavarse_obj_from_yaml_config(self.config_path, candidate_num=self.object_candidate_num, vhacd=self.vhacd)
        if not res:
            print("=" * 20)
            print("some objects cannot be found in objaverse, task_build failed, now exit ...")
            print("=" * 20)
            exit()
        
        self.config = None
        while self.config is None:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        for obj in self.config:
            if "solution_path" in obj:
                self.solution_path = obj["solution_path"]
                break
        
        ### parse config
        urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, use_table, \
            articulated_init_joint_angles, spatial_relationships, distractor_config_path, urdf_movables = parse_config(self.config, 
                        use_bard=True, obj_id=self.obj_id)
        if not use_table:
            urdf_on_table = [False for _ in urdf_on_table]
        urdf_names = [x.lower() for x in urdf_names]
        for name in urdf_names:
            self.is_distractor[name] = 0
        
        ### parse distractor object config (semantically meaningful objects that are related but not used for the task)
        if distractor_config_path is not None:
            self.distractor_config_path = distractor_config_path
            res = download_and_parse_objavarse_obj_from_yaml_config(distractor_config_path, candidate_num=self.object_candidate_num, vhacd=self.vhacd)
            with open(distractor_config_path, 'r') as f:
                self.distractor_config = yaml.safe_load(f)
            distractor_urdf_paths, distractor_urdf_sizes, distractor_urdf_positions, distractor_urdf_names, distractor_urdf_types, \
                distractor_urdf_on_table, _, _, _, _, _ = \
                    parse_config(self.distractor_config, use_bard=True, obj_id=self.obj_id, use_vhacd=False)
            distractor_urdf_names = [x.lower() for x in distractor_urdf_names]
            if not use_table:
                distractor_urdf_on_table = [False for _ in distractor_urdf_on_table]
            
            for name in distractor_urdf_names:
                self.is_distractor[name] = 1
                
            distractor_movables = [True for _ in distractor_urdf_names]
            
            urdf_paths += distractor_urdf_paths
            urdf_sizes += distractor_urdf_sizes
            urdf_positions += distractor_urdf_positions
            urdf_names += distractor_urdf_names
            urdf_types += distractor_urdf_types
            urdf_on_table += distractor_urdf_on_table
            urdf_movables += distractor_movables

        if restore_state is not None:
            if "urdf_paths" in restore_state:
                self.urdf_paths = restore_state['urdf_paths']
                urdf_paths = [self.urdf_paths[name] for name in urdf_names]
            if "object_sizes" in restore_state:
                self.simulator_sizes = restore_state['object_sizes']
                urdf_sizes = [self.simulator_sizes[name] for name in urdf_names]
                
        return urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables, \
            use_table, articulated_init_joint_angles, spatial_relationships
        
    def load_table(self, use_table, restore_state):
        self.use_table = use_table
        if use_table:
            from manipulation.table_utils import table_paths, table_scales, table_poses, table_bbox_scale_down_factors
            self.table_path = table_paths[np.random.randint(len(table_paths))]
            if restore_state is not None:
                self.table_path = restore_state['table_path']

            table_scale = table_scales[self.table_path] 
            table_pos = table_poses[self.table_path]
            table_orientation = [np.pi/2, 0, 0]

            self.table = p.loadURDF(osp.join(self.asset_dir, self.table_path, "material.urdf"), physicsClientId=self.id, useFixedBase=True, 
                                    globalScaling=table_scale)
            
            if not self.randomize:
                random_orientation = p.getQuaternionFromEuler(table_orientation, physicsClientId=self.id)
            else:
                random_orientations = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
                random_orientation = p.getQuaternionFromEuler([np.pi/2, 0, random_orientations[np.random.randint(4)]], physicsClientId=self.id)

            p.resetBasePositionAndOrientation(self.table, table_pos, random_orientation, physicsClientId=self.id)
            self.table_bbox_min, self.table_bbox_max = self.get_aabb(self.table)
            
            table_range = self.table_bbox_max - self.table_bbox_min
            self.table_bbox_min[:2] += table_range[:2] * table_bbox_scale_down_factors[self.table_path]
            self.table_bbox_max[:2] -= table_range[:2] * table_bbox_scale_down_factors[self.table_path]
            self.table_height = self.table_bbox_max[2]
            p.addUserDebugLine([*self.table_bbox_min[:2], self.table_height], self.table_bbox_max, [1, 0, 0], lineWidth=10, lifeTime=0, physicsClientId=self.id)
            self.simulator_sizes["init_table"] = table_scale
            self.urdf_ids["init_table"] = self.table
            self.is_distractor['init_table'] = 0
    
    def load_object(self, urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables):
        for path, size, pos, name, type, on_table, moveable in zip(urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables):
            name = name.lower()
            # by default, all objects movable, except the urdf files
            use_fixed_base = (type == 'urdf' and not self.is_distractor[name])
            if type == 'urdf' and moveable: # if gpt specified the object is movable, then it is movable
                use_fixed_base = False
            size = min(size, 1.2)
            size = max(size, 0.1) # if the object is too small, current gripper cannot really manipulate it.
            
            x_orient = np.pi/2 if type == 'mesh' else 0 # handle different coordinate axis by objaverse and partnet-mobility
            if self.randomize or self.is_distractor[name]:
                orientation = p.getQuaternionFromEuler([x_orient, 0, self.np_random.uniform(-np.pi/3, np.pi/3)], physicsClientId=self.id)
            else:
                orientation = p.getQuaternionFromEuler([x_orient, 0, 0], physicsClientId=self.id)

            if not on_table:
                load_pos = pos
            else: # change to be table coordinate
                table_xy_range = self.table_bbox_max[:2] - self.table_bbox_min[:2]
                obj_x = self.table_bbox_min[0] + pos[0] * table_xy_range[0]
                obj_y = self.table_bbox_min[1] + pos[1] * table_xy_range[1]
                obj_z = self.table_height
                load_pos = [obj_x, obj_y, obj_z]
            id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size)

            # scale size 
            if name in self.simulator_sizes:
                p.removeBody(id, physicsClientId=self.id)
                saved_size = self.simulator_sizes[name]
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=saved_size)
            else:
                min_aabb, max_aabb = self.get_aabb(id)
                actual_size = np.linalg.norm(max_aabb - min_aabb)
                if np.abs(actual_size - size) > 0.05:
                    p.removeBody(id, physicsClientId=self.id)
                    id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size ** 2 / actual_size)
                    self.simulator_sizes[name] = size ** 2 / actual_size
                else:
                    self.simulator_sizes[name] = size

            self.urdf_ids[name] = id
            self.urdf_paths[name] = path
            self.urdf_types[name] = type
            self.init_positions[name] = np.array(load_pos)
            self.on_tables[name] = on_table

            print("Finished loading object: ", name)
    
    def adjust_object_positions(self, robot_base_pos):
        object_height = {}
        for name, id in self.urdf_ids.items():
            if name == 'robot' or name == 'plane' or name == 'init_table': continue
            min_aabb, max_aabb = self.get_aabb(id)
            min_z = min_aabb[2]
            object_height[id] = 2 * self.init_positions[name][2] - min_z
            pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
            new_pos = np.array(pos) 
            new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
            new_pos[2] = object_height[id]
            p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
            self.init_positions[name] = new_pos
        
        return object_height
        
    def resolve_collision(self, robot_base_pos, object_height, spatial_relationships):
        collision = True
        collision_cnt = 1
        while collision:
            if collision_cnt % 50 == 0: # if collision is not resolved every 50 iterations, we randomly reset object's position
                for name, id in self.urdf_ids.items():
                    if name == 'robot' or name == 'plane' or name == "init_table": continue
                    pos = self.init_positions[name]
                    _, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                    new_pos = np.array(pos) + np.random.uniform(-0.2, 0.2, size=3)
                    new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
                    new_pos[2] = object_height[id]
                    p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
                    p.stepSimulation(physicsClientId=self.id)

            push_directions = defaultdict(list) # store the push direction for each object

            # detect collisions between objects 
            detected_collision = False
            for name, id in self.urdf_ids.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': continue
                for name2, id2 in self.urdf_ids.items():
                    if name == name2 or name2 == 'robot' or name2 == 'plane' or name2 == 'init_table': continue

                    # if gpt specifies obj a and obj b should have some special relationship, then skip collision resolution
                    skip = False
                    for spatial_relationship in spatial_relationships:
                        words = spatial_relationship.lower().split(",")
                        words = [word.strip().lstrip() for word in words]
                        if name in words and name2 in words:
                            skip = True
                            break

                    if skip: continue
                    
                    contact_points = p.getClosestPoints(id, id2, 0.01, physicsClientId=self.id)
                    if len(contact_points) > 0:
                        contact_point = contact_points[0]
                        push_direction = contact_point[7]
                        push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])

                        # both are distractors or both are not, push both objects away
                        if (self.is_distractor[name] and self.is_distractor[name2]) or \
                            (not self.is_distractor[name] and not self.is_distractor[name2]):
                            push_directions[id].append(-push_direction)
                            push_directions[id2].append(push_direction)
                        # only 1 is distractor, only pushes the distractor
                        if self.is_distractor[name] and not self.is_distractor[name2]:
                            push_directions[id].append(push_direction)
                        if not self.is_distractor[name] and self.is_distractor[name2]:
                            push_directions[id2].append(-push_direction)
                        
                        detected_collision = True

            # collisions between robot and objects, only push object away
            for name, id in self.urdf_ids.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': 
                    continue

                contact_points = p.getClosestPoints(self.robot.body, id, 0.05, physicsClientId=self.id)
                if len(contact_points) > 0:
                    contact_point = contact_points[0]
                    push_direction = contact_point[7]
                    push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])
                    push_directions[id].append(-push_direction)
                    detected_collision = True

            # between table and objects that should not be placed on table
            if self.use_table:
                for name, id in self.urdf_ids.items():
                    if name == 'robot' or name == 'plane' or name == 'init_table': 
                        continue
                    if self.on_tables[name]:
                        continue

                    contact_points = p.getClosestPoints(self.robot.body, id, 0.05, physicsClientId=self.id)
                    if len(contact_points) > 0:
                        contact_point = contact_points[0]
                        push_direction = contact_point[7]
                        push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])
                        push_directions[id].append(-push_direction)
                        detected_collision = True
            
            # move objects
            push_distance = 0.1
            for id in push_directions:
                for direction in push_directions[id]:
                    pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                    new_pos = np.array(pos) + push_distance * direction    
                    new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
                    new_pos[2] = object_height[id]

                    p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
                    p.stepSimulation(physicsClientId=self.id)

            collision = detected_collision
            collision_cnt += 1

            if collision_cnt > 1000:
                break
    
    def record_initial_joint_and_pose(self):
        self.initial_joint_angle = {}
        for name in self.urdf_ids:        
            obj_id = self.urdf_ids[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            if self.urdf_types[name.lower()] == 'urdf':
                self.initial_joint_angle[name] = {}
                num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
                for joint_idx in range(num_joints):
                    joint_name = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[1].decode("utf-8")
                    joint_angle = p.getJointState(obj_id, joint_idx, physicsClientId=self.id)[0]
                    self.initial_joint_angle[name][joint_name] = joint_angle
        
        self.initial_pos = {}
        self.initial_orient = {}
        for name in self.urdf_ids:
            obj_id = self.urdf_ids[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            pos, orient = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.id)
            self.initial_pos[name] = pos
            self.initial_orient[name] = orient
        
    def set_to_default_joint_angles(self):
        for obj_name in self.urdf_ids:
            if obj_name == 'robot' or obj_name == 'plane' or obj_name == "init_table": continue
            obj_id = self.urdf_ids[obj_name]
            num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
            for joint_idx in range(num_joints):
                joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                if joint_limit_low > joint_limit_high:
                    joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                joint_val = joint_limit_low + 0.06 * (joint_limit_high - joint_limit_low)
                p.resetJointState(obj_id, joint_idx, joint_val, physicsClientId=self.id)

    def handle_gpt_special_relationships(self, spatial_relationships):
        # we support "on" and "in" for now, but this can be extended to more relationships
        for spatial_relationship in spatial_relationships:
            words = spatial_relationship.lower().split(",")
            words = [word.strip().lstrip() for word in words]
            if words[0] == "on":
                obj_a = words[1]
                obj_b = words[2]
                if len(words) == 4:
                    obj_b_link = words[3]
                    obj_b_link_id = get_link_id_from_name(self, obj_b, obj_b_link)
                else:
                    obj_b_link_id = -1
                obj_a_id, obj_b_id = self.urdf_ids[obj_a], self.urdf_ids[obj_b]
                
                obj_a_bbox_min, obj_a_bbox_max = self.get_aabb(obj_a_id)
                obj_a_size = obj_a_bbox_max - obj_a_bbox_min
                target_aabb_min, target_aabb_max = self.get_aabb_link(obj_b_id, obj_b_link_id)
                id_line = p.addUserDebugLine(target_aabb_min, target_aabb_max, [1, 0, 0], lineWidth=10, lifeTime=0, physicsClientId=self.id)
                id_point = p.addUserDebugPoints([(target_aabb_min + target_aabb_max) / 2], [[0, 0, 1]], 10, 0, physicsClientId=self.id)

                new_pos = (target_aabb_min + target_aabb_max) / 2
                new_pos[2] = target_aabb_max[2] # put obj a on top of obj b.
                new_pos[2] += obj_a_size[2] # add the height of obj a
                if not self.randomize:
                    obj_a_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0], physicsClientId=self.id)
                else:
                    random_orientations = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
                    obj_a_orientation = p.getQuaternionFromEuler([np.pi/2, 0, random_orientations[np.random.randint(4)]], physicsClientId=self.id)

                p.resetBasePositionAndOrientation(obj_a_id, new_pos, obj_a_orientation, physicsClientId=self.id)
                
                p.removeUserDebugItem(id_line, physicsClientId=self.id)
                p.removeUserDebugItem(id_point, physicsClientId=self.id)

            if words[0] == 'in':
                obj_a = words[1]
                obj_b = words[2]
                if len(words) == 4:
                    obj_b_link = words[3]
                    obj_b_link_id = get_link_id_from_name(self, obj_b, obj_b_link)
                else:
                    obj_b_link_id = -1
                obj_a_id, obj_b_id = self.urdf_ids[obj_a], self.urdf_ids[obj_b]
                
                # if after a lot of trying times, there is still collision, we should scale down the size of object A.
                cnt = 1
                collision_free = False
                obj_a_new_size = self.simulator_sizes[obj_a]
                obj_a_ori_pos, obj_a_orientation = p.getBasePositionAndOrientation(obj_a_id, physicsClientId=self.id)         
                target_aabb_min, target_aabb_max = self.get_aabb_link(obj_b_id, obj_b_link_id)

                while not collision_free:
                    if cnt % 100 == 0:
                        print("scaling down! object size is {}".format(obj_a_new_size))
                        obj_a_new_size = obj_a_new_size * 0.9
                        p.removeBody(obj_a_id, physicsClientId=self.id)
                        obj_a_id = p.loadURDF(self.urdf_paths[obj_a],
                                            basePosition=obj_a_ori_pos,
                                            baseOrientation=obj_a_orientation,
                                            physicsClientId=self.id, useFixedBase=False, globalScaling=obj_a_new_size)
                        self.urdf_ids[obj_a] = obj_a_id
                        self.simulator_sizes[obj_a] = obj_a_new_size

                    obj_a_bbox_min, obj_a_bbox_max = self.get_aabb(obj_a_id)
                    obj_a_size = obj_a_bbox_max - obj_a_bbox_min
                    id_line = p.addUserDebugLine(target_aabb_min, target_aabb_max, [1, 0, 0], lineWidth=10, lifeTime=0, physicsClientId=self.id)
                    id_point = p.addUserDebugPoints([(target_aabb_min + target_aabb_max) / 2], [[0, 0, 1]], 10, 0, physicsClientId=self.id)

                    center_pos = (target_aabb_min + target_aabb_max) / 2
                    up_pos = center_pos.copy()
                    up_pos[2] += obj_a_size[2]
                    possible_locations = [center_pos, up_pos]
                    obj_a_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0], physicsClientId=self.id)
                    for pos in possible_locations: # we try two possible locations to put obj a in obj b
                        p.resetBasePositionAndOrientation(obj_a_id, pos, obj_a_orientation, physicsClientId=self.id)
                        contact_points = p.getClosestPoints(obj_a_id, obj_b_id, 0.002, physicsClientId=self.id)

                        if len(contact_points) == 0:
                            collision_free = True
                            break
                    
                    p.removeUserDebugItem(id_line, physicsClientId=self.id)
                    p.removeUserDebugItem(id_point, physicsClientId=self.id)

                    cnt += 1
                    if cnt > 1000: # if after scaling for 10 times it still does not work, let it be. 
                        break
                        

    def handle_gpt_joint_angle(self, articulated_init_joint_angles):
        for name in articulated_init_joint_angles:
            obj_id = self.urdf_ids[name.lower()]

            for joint_name, joint_angle in articulated_init_joint_angles[name].items():
                joint_idx = get_joint_id_from_name(self, name.lower(), joint_name)
                joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                if joint_limit_low > joint_limit_high:
                    joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                if 'random' not in joint_angle:
                    joint_angle = float(joint_angle)
                    joint_angle = min(joint_angle, 0.7)
                    joint_angle = max(joint_angle, 0.06)
                    joint_angle = joint_limit_low + joint_angle * (joint_limit_high - joint_limit_low)
                else:
                    joint_angle = self.np_random.uniform(joint_limit_low, joint_limit_high)
                p.resetJointState(obj_id, joint_idx, joint_angle, physicsClientId=self.id)

    def reset(self):
        p.restoreState(self.init_state, physicsClientId=self.id)

        self.time_step = 0
        self.success = False
        if self.use_suction:
            self.deactivate_suction()

        return self._get_obs()

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)
    
    def setup_camera_rpy(self, camera_target=[0, 0, 0.3], distance=1.6, rpy=[0, -30, -30], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        if self.use_table:
            camera_target = np.array([0, 0, 0.3])
        else:
            for name in self.urdf_ids: # randomly center at an object
                if name in ['robot', 'plane', 'init_table']: continue
                obj_id = self.urdf_ids[name]
                min_aabb, max_aabb = self.get_aabb(obj_id)
                center = (min_aabb + max_aabb) / 2
                camera_target = center 
                break

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

    def take_step(self, actions, gains=None, forces=None):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple): 
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)

        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = self.base_action_space.shape[0] 
            action = np.copy(actions[action_index:action_index+agent_action_len])
            action_index += agent_action_len
            action = np.clip(action, self.action_low, self.action_high)

            translation = action[:3]
            rotation = action[3:6]
            suction = action[6]

            joint = agent.right_end_effector if 'right' in agent.controllable_joints else agent.left_end_effector
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            ik_indices = [_ for _ in range(len(agent.right_arm_ik_indices))]
            pos, orient = agent.get_pos_orient(joint)

            # eef translation
            if self.translation_mode == 'delta-translation':
                pos += translation * self.max_translation
            elif self.translation_mode == 'normalized-direct-translation':
                pos = translation * self.scene_range + self.scene_center
            elif self.translation_mode == 'direct-translation':
                pos = translation 

            # eef rotation
            if self.rotation_mode == 'euler-angle':
                rotation = rotation * np.pi
                orient = p.getQuaternionFromEuler(rotation)
            elif 'delta-axis-angle' in self.rotation_mode or 'delta-euler-angle' in self.rotation_mode:
                orient = self.apply_delta_rotation(rotation, orient)

            agent_joint_angles = agent.ik(joint, pos, orient, ik_indices, max_iterations=200, use_current_as_rest=False)
            agent.control(agent.controllable_joint_indices, agent_joint_angles, gains[i], forces[i])

            # gripper
            if not self.use_suction:
                agent.set_gripper_open_position(agent.right_gripper_indices, [suction])
            else:
                if suction >= 0: self.activate_suction()
                else: self.deactivate_suction()

        for _ in range(self.frameskip):
            p.stepSimulation(physicsClientId=self.id)                   

    def apply_delta_rotation(self, delta_rotation, orient):
        if 'delta-axis-angle' in self.rotation_mode:
            dtheta = np.linalg.norm(delta_rotation)
            if dtheta > 0:
                delta_rotation = delta_rotation / dtheta
                dtheta = dtheta * self.max_rotation_angle / np.sqrt(3)
                delta_rotation_matrix = R.from_rotvec(delta_rotation * dtheta).as_matrix()
            else:
                delta_rotation_matrix = np.eye(3)
            current_matrix = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)

            if self.rotation_mode == 'delta-axis-angle-local':
                new_rotation = current_matrix @ delta_rotation_matrix
            elif self.rotation_mode == 'delta-axis-angle-global':
                new_rotation = delta_rotation_matrix @ current_matrix
            orient = R.from_matrix(new_rotation).as_quat()
        elif self.rotation_mode == 'delta-euler-angle':
            euler_angle = delta_rotation / np.sqrt(3) * self.max_rotation_angle
            delta_quaternion = p.getQuaternionFromEuler(euler_angle)
            orient = delta_quaternion * orient
            
        return orient
    

    def activate_suction(self):
        if not self.activated:
            # assume the suction is attached to the right end effector
            suction_id = self.suction_id
            points = p.getContactPoints(bodyA=self.robot.body, linkIndexA=suction_id, physicsClientId=self.id)
            if points:
                # Handle contact between suction with a rigid object.
                contact_object_id_link_cnts = defaultdict(int)
                for point in points:
                    obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                    
                    if obj_id == self.urdf_ids['plane'] or obj_id == self.robot.body:
                        pass
                    else:
                        contact_object_id_link_cnts[(obj_id, contact_link)] += 1
                
                if len(contact_object_id_link_cnts) > 0:
                    # find the object that has the most contact points
                    obj_id, contact_link = max(contact_object_id_link_cnts.items(), key=lambda x: x[1])[0]
                    # print("contact with object: ", obj_id, contact_link)

                    body_pose = p.getLinkState(self.robot.body, suction_id, physicsClientId=self.id)
                    if contact_link >= 0:
                        obj_link_pose = p.getLinkState(obj_id, contact_link, physicsClientId=self.id)
                    else:
                        obj_link_pose = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                        world_to_body[1],
                                                        obj_link_pose[0], obj_link_pose[1])
                    
                    suction_to_obj = p.invertTransform(obj_to_body[0], obj_to_body[1])
                    
                    self.create_suction_constraint(obj_id, contact_link, suction_to_obj)
                    
                    self.activated = True
                    self.suction_obj_id = obj_id
                    self.suction_contact_link = contact_link
                    self.suction_to_obj_pose = suction_to_obj

    def create_suction_constraint(self, suction_obj_id, suction_contact_link, suction_to_obj_pose):
        suction_id = self.suction_id
        self.contact_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot.body,
            parentLinkIndex=suction_id,
            childBodyUniqueId=suction_obj_id,
            childLinkIndex=suction_contact_link,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0), 
            parentFramePosition=(0, 0, 0),
            parentFrameOrientation=(0, 0, 0),
            childFramePosition=suction_to_obj_pose[0],
            childFrameOrientation=suction_to_obj_pose[1], 
            physicsClientId=self.id)
        
        p.changeConstraint(self.contact_constraint, maxForce=5000, physicsClientId=self.id)

    def deactivate_suction(self):
        self.activated = False
        if self.contact_constraint is not None:
            p.removeConstraint(self.contact_constraint, physicsClientId=self.id)
            self.contact_constraint = None


    def step(self, action):
        self.time_step += 1        
        self.take_step(action)
        obs = self._get_obs()                
        # to handle some stupid typing error in early prompts
        try:
            reward, success = self._compute_reward() 
        except:
            reward, success = self.compute_reward()
        self.success = success
        done = self.time_step == self.horizon
        info = self._get_info()
        return obs, reward, done, info


    def _get_info(self):
        return {}

    def _get_obs(self):
        ### For RL policy learning, observation space includes:
        # 1. object positions and orientations (6 * num_objects)
        # 2. object min and max bounding box (6 * num_objects)
        # 3. articulated object joint angles (num_objects * num_joints) 
        # 4. articulated object link position and orientation (num_objects * num_joints * 6) 
        # 5. robot base position (xy)
        # 6. robot end-effector position and orientation (6)
        # 7. gripper suction activated/deactivate or gripper joint angle (if not using suction gripper) (1)
        obs = np.zeros(self.base_observation_space.shape[0])
            
        cnt = 0
        for name, id in self.urdf_ids.items():
            if name == 'plane' or name == 'robot':
                continue
            if self.is_distractor[name]:
                continue

            pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
            euler_angle = p.getEulerFromQuaternion(orient)
            obs[cnt:cnt+3] = self.normalize_position(pos)
            obs[cnt+3:cnt+6] = euler_angle
            cnt += 6

        for name, id in self.urdf_ids.items():
            if name == 'plane' or name == 'robot':
                continue
            if self.is_distractor[name]:
                continue
            min_aabb, max_aabb = self.get_aabb(id)
            obs[cnt:cnt+3] = self.normalize_position(min_aabb)
            obs[cnt+3:cnt+6] = self.normalize_position(max_aabb)
            cnt += 6

        for name in self.urdf_types:
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]:
                num_joints = p.getNumJoints(self.urdf_ids[name], physicsClientId=self.id)
                for joint_idx in range(num_joints):
                    joint_angle = p.getJointState(self.urdf_ids[name], joint_idx, physicsClientId=self.id)[0]
                    obs[cnt] = joint_angle
                    cnt += 1
                    link_pos, link_orient = p.getLinkState(self.urdf_ids[name], joint_idx, physicsClientId=self.id)[:2]
                    link_pos = self.normalize_position(link_pos)
                    link_euler_angle = p.getEulerFromQuaternion(link_orient)
                    obs[cnt:cnt+3] = link_pos
                    obs[cnt+3:cnt+6] = link_euler_angle
                    cnt += 6

        robot_base_pos, robot_base_orient = self.robot.get_base_pos_orient()
        robot_base_pos = self.normalize_position(robot_base_pos)
        obs[cnt:cnt+2] = robot_base_pos[:2]
        cnt += 2

        robot_eef_pos, robot_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        robot_eef_euler_angle = p.getEulerFromQuaternion(robot_eef_orient)
        obs[cnt:cnt+3] = self.normalize_position(robot_eef_pos)
        obs[cnt+3:cnt+6] = robot_eef_euler_angle
        cnt += 6

        if not self.use_suction:
            # get joint angle of the gripper
            left_finger_joint_angle = p.getJointState(self.robot.body, self.robot.right_gripper_indices[0], physicsClientId=self.id)[0]
            right_finger_joint_angle = p.getJointState(self.robot.body, self.robot.right_gripper_indices[1], physicsClientId=self.id)[0]
            obs[cnt] = left_finger_joint_angle
            obs[cnt+1] = right_finger_joint_angle
            cnt += 2
        else:
            obs[cnt] = int(self.activated)
            cnt += 1

        return obs

    def disconnect(self):
        p.disconnect(self.id)

    def close(self):
        p.disconnect(self.id)
    