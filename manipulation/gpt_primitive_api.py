import pybullet as p
import os
import numpy as np
import open3d as o3d
from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import get_pc_and_normal, align_gripper_z_with_normal, align_gripper_x_with_normal
from manipulation.gpt_reward_api import get_link_pc, get_bounding_box, get_link_id_from_name
from manipulation.utils import save_env, load_env

MOTION_PLANNING_TRY_TIMES=100

def get_save_path(simulator):
    return simulator.primitive_save_path


def release_grasp(simulator):
    simulator.deactivate_suction()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rgbs = []
    states = []
    for t in range(20):
        p.stepSimulation()
        rgbs.append(simulator.render()[0])
        state_save_path = os.path.join(save_path, "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        states.append(state_save_path)

    return rgbs, states

def grasp_object(simulator, object_name):
    ori_state = save_env(simulator, None)
    p.stepSimulation()
    object_name = object_name.lower()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # if the target object is already grasped.  
    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
    if points:
        for point in points:
            obj_id, contact_link = point[2], point[4]
            if obj_id == simulator.urdf_ids[object_name]:
                simulator.activate_suction()
                rgbs = []
                states = []
                for t in range(10):
                    p.stepSimulation()
                    rgbs.append(simulator.render()[0])
                    state_save_path = os.path.join(save_path, "state_{}.pkl".format(t))
                    save_env(simulator, state_save_path)
                    states.append(state_save_path)
                return rgbs, states

    rgbs, states = approach_object(simulator, object_name)
    base_t = len(rgbs)
    if base_t > 1:
        for t in range(10):
            simulator.activate_suction()
            p.stepSimulation()
            rgbs.append(simulator.render()[0])
            state_save_path = os.path.join(save_path, "state_{}.pkl".format(t + base_t))
            save_env(simulator, state_save_path)
            states.append(state_save_path)
    else:
        # directy reset the state
        load_env(simulator, state=ori_state)

    return rgbs, states

def grasp_object_link(simulator, object_name, link_name):
    ori_state = save_env(simulator, None)
    p.stepSimulation()
    object_name = object_name.lower()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # if the target object link is already grasped.  
    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
    if points:
        for point in points:
            obj_id, contact_link = point[2], point[4]
            if obj_id == simulator.urdf_ids[object_name] and contact_link == get_link_id_from_name(simulator, object_name, link_name):
                simulator.activate_suction()
                rgbs = []
                states = []
                for t in range(10):
                    p.stepSimulation()
                    rgbs.append(simulator.render()[0])
                    state_save_path = os.path.join(save_path, "state_{}.pkl".format(t))
                    save_env(simulator, state_save_path)
                    states.append(state_save_path)
                return rgbs, states


    rgbs, states = approach_object_link(simulator, object_name, link_name)
    base_t = len(rgbs)
    if base_t > 1:
        simulator.activate_suction()
        for t in range(10):
            p.stepSimulation()
            rgbs.append(simulator.render()[0])
            state_save_path = os.path.join(save_path, "state_{}.pkl".format(t + base_t))
            save_env(simulator, state_save_path)
            states.append(state_save_path)
    else:
        # directy reset the state
        load_env(simulator, state=ori_state)

    return rgbs, states

def approach_object(simulator, object_name, dynamics=False):
    save_path = get_save_path(simulator)
    ori_state = save_env(simulator, None)
    simulator.deactivate_suction()
    release_rgbs = []
    release_states = []
    release_steps = 20
    for t in range(release_steps):
        p.stepSimulation()
        rgb, depth = simulator.render()
        release_rgbs.append(rgb)
        state_save_path = os.path.join(save_path, "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        release_states.append(state_save_path)

    object_name = object_name.lower()
    it = 0
    object_name = object_name.lower()
    object_pc, object_normal = get_pc_and_normal(simulator, object_name)
    low, high = get_bounding_box(simulator, object_name)
    com = (low + high) / 2
    current_joint_angles = simulator.robot.get_joint_angles(indices=simulator.robot.right_arm_joint_indices)
    

    while True:
        random_point = object_pc[np.random.randint(0, object_pc.shape[0])]
        random_normal = object_normal[np.random.randint(0, object_normal.shape[0])]

        ### adjust the normal such that it points outwards the object.
        line = com - random_point
        if np.dot(line, random_normal) > 0:
            random_normal = -random_normal
            
        for normal in [random_normal, -random_normal]:
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, current_joint_angles)

            target_pos = random_point
            real_target_pos = target_pos + normal * 0
            if simulator.robot_name in ["panda", "sawyer"]:
                target_orientation = align_gripper_z_with_normal(-normal).as_quat()
                mp_target_pos = target_pos + normal * 0.03
            elif simulator.robot_name in ['ur5', 'fetch']:
                target_orientation = align_gripper_x_with_normal(-normal).as_quat()
                if simulator.robot_name == 'ur5':
                    mp_target_pos = target_pos + normal * 0.07
                elif simulator.robot_name == 'fetch':
                    mp_target_pos = target_pos + normal * 0.07

            all_objects = list(simulator.urdf_ids.keys())
            all_objects.remove("robot")
            obstacles = [simulator.urdf_ids[x] for x in all_objects]
            allow_collision_links = []
            res, path = motion_planning(simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links)

            if res:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                rgbs = release_rgbs
                intermediate_states = release_states
                for idx, q in enumerate(path):
                    if not dynamics:
                        simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
                        p.stepSimulation()
                    else:
                        for _ in range(3):
                            simulator.robot.control(simulator.robot.right_arm_joint_indices, q, simulator.robot.motor_gains, forces=5 * 240.)
                            p.stepSimulation()

                    rgb, depth = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path,  "state_{}.pkl".format(idx + release_steps))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                base_idx = len(intermediate_states)
                for t in range(20):
                    ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
                    ik_joints = simulator.robot.ik(simulator.robot.right_end_effector, 
                                                    real_target_pos, target_orientation, 
                                                    ik_indices=ik_indices)
                    p.setJointMotorControlArray(simulator.robot.body, jointIndices=simulator.robot.right_arm_joint_indices, 
                                                controlMode=p.POSITION_CONTROL, targetPositions=ik_joints,
                                                forces=[5*240] * len(simulator.robot.right_arm_joint_indices), physicsClientId=simulator.id)
                    p.stepSimulation()
                    rgb, depth = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path,  "state_{}.pkl".format(base_idx + t))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                    # TODO: check if there is already a collision. if so, break.
                    collision = False
                    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
                    if points:
                        # Handle contact between suction with a rigid object.
                        for point in points:
                            obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                            
                            if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body:
                                pass
                            else:
                                collision = True
                                break
                    if collision:
                        break

                return rgbs, intermediate_states
        
            it += 1
            if it > MOTION_PLANNING_TRY_TIMES:
                print("failed to execute the primitive")
                load_env(simulator, state=ori_state)
                save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
                rgbs = [simulator.render()[0]]
                state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
                return rgbs, state_files

def approach_object_link(simulator, object_name, link_name, dynamics=False):
    save_path = get_save_path(simulator)
    ori_state = save_env(simulator, None)
    simulator.deactivate_suction()
    
    release_rgbs = []
    release_states = []
    release_steps = 20
    for t in range(release_steps):
        p.stepSimulation()
        rgb, depth = simulator.render()
        release_rgbs.append(rgb)
        state_save_path = os.path.join(save_path, "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        release_states.append(state_save_path)

    object_name = object_name.lower()
    it = 0

    object_name = object_name.lower()
    link_pc = get_link_pc(simulator, object_name, link_name) 
    object_pc = link_pc
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(object_pc)
    pcd.estimate_normals()
    object_normal = np.asarray(pcd.normals)

    current_joint_angles = simulator.robot.get_joint_angles(indices=simulator.robot.right_arm_joint_indices)

    while True:
        object_name = object_name.lower()
        target_pos = link_pc[np.random.randint(0, link_pc.shape[0])]
        nearest_point_idx = np.argmin(np.linalg.norm(object_pc - target_pos.reshape(1, 3), axis=1))
        align_normal = object_normal[nearest_point_idx]

        ### adjust the normal such that it points outwards the object.
        low, high = get_bounding_box(simulator, object_name)
        com = (low + high) / 2
        line = com - target_pos
        if np.dot(line, align_normal) > 0:
            align_normal = -align_normal

        for normal in [align_normal, -align_normal]:
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, current_joint_angles)

            real_target_pos = target_pos + normal * 0
            debug_id = p.addUserDebugLine(target_pos, target_pos + normal, [1, 0, 0], 5)
            if simulator.robot_name in ["panda", "sawyer"]:
                target_orientation = align_gripper_z_with_normal(-normal).as_quat()
                mp_target_pos = target_pos + normal * 0.03
            elif simulator.robot_name in ['ur5', 'fetch']:
                target_orientation = align_gripper_x_with_normal(-normal).as_quat()
                if simulator.robot_name == 'ur5':
                    mp_target_pos = target_pos + normal * 0.07
                elif simulator.robot_name == 'fetch':
                    mp_target_pos = target_pos + normal * 0.07

            all_objects = list(simulator.urdf_ids.keys())
            all_objects.remove("robot")
            obstacles = [simulator.urdf_ids[x] for x in all_objects]
            allow_collision_links = []
            res, path = motion_planning(simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links)
            p.removeUserDebugItem(debug_id)

            if res:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                rgbs = release_rgbs
                intermediate_states = release_states
                for idx, q in enumerate(path):
                    if not dynamics:
                        simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
                    else:
                        for _ in range(3):
                            simulator.robot.control(simulator.robot.right_arm_joint_indices, q, simulator.robot.motor_gains, forces=5 * 240.)
                            p.stepSimulation()

                    p.stepSimulation()
                    rgb, depth = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path,  "state_{}.pkl".format(idx + release_steps))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                base_idx = len(intermediate_states)
                for t in range(20):

                    print("post motion planing step: ", t)
                    print("rgb image length: ", len(rgbs))
                    ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
                    ik_joints = simulator.robot.ik(simulator.robot.right_end_effector, 
                                                    real_target_pos, target_orientation, 
                                                    ik_indices=ik_indices)
                    p.setJointMotorControlArray(simulator.robot.body, jointIndices=simulator.robot.right_arm_joint_indices, 
                                                controlMode=p.POSITION_CONTROL, targetPositions=ik_joints,
                                                forces=[5*240] * len(simulator.robot.right_arm_joint_indices), physicsClientId=simulator.id)
                    p.stepSimulation()
                    rgb, depth = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path,  "state_{}.pkl".format(base_idx + t))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                    # TODO check here if there is a collision. if so, break
                    collision = False
                    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
                    if points:
                        # Handle contact between suction with a rigid object.
                        for point in points:
                            obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                            
                            if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body:
                                pass
                            else:
                                collision = True
                                break
                    if collision:
                        break

                return rgbs, intermediate_states 
            
            it += 1
            if it > MOTION_PLANNING_TRY_TIMES:
                print("failed to execute the primitive")
                load_env(simulator, state=ori_state)
                save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
                rgbs = [simulator.render()[0]]
                state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
                return rgbs, state_files