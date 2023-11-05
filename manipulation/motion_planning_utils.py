import numpy as np
import pybullet_ompl.pb_ompl as pb_ompl
import pybullet as p
import copy

def motion_planning(env, target_pos, target_orientation, planner="BITstar", obstacles=[], allow_collision_links=[], panda_slider=True):
    current_joint_angles = copy.deepcopy(env.robot.get_joint_angles(indices=env.robot.right_arm_joint_indices))
    ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body)
    ompl_robot.set_state(current_joint_angles)

    allow_collision_robot_link_pairs = []
    if env.robot_name == "sawyer":
        allow_collision_robot_link_pairs.append((5, 8))
    if env.robot_name == 'fetch':
        allow_collision_robot_link_pairs.append((3, 19))
    pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links, 
                                       allow_collision_robot_link_pairs=allow_collision_robot_link_pairs)
    pb_ompl_interface.set_planner(planner)

    # first need to compute a collision-free IK solution
    ik_lower_limits = env.robot.ik_lower_limits 
    ik_upper_limits = env.robot.ik_upper_limits 
    print("ik_lower_limits: ", ik_lower_limits)
    print("ik_upper_limits: ", ik_upper_limits)
    ik_joint_ranges = ik_upper_limits - ik_lower_limits

    it = 0
    while True:
        if it % 10 == 0:
            print("sampling target ik it: ", it)

        ik_rest_poses = np.random.uniform(ik_lower_limits, ik_upper_limits)
    
        target_joint_angle = np.array(p.calculateInverseKinematics(
            env.robot.body, env.robot.right_end_effector, 
            targetPosition=target_pos, targetOrientation=target_orientation, 
            lowerLimits=ik_lower_limits.tolist(), upperLimits=ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), 
            restPoses=ik_rest_poses.tolist(), 
            maxNumIterations=1000,
            residualThreshold=1e-4
        ))
        if np.all(target_joint_angle >= ik_lower_limits) and np.all(target_joint_angle <= ik_upper_limits) \
                and pb_ompl_interface.is_state_valid(target_joint_angle):
            break
        it += 1

        if it > 600:
            ompl_robot.set_state(current_joint_angles)
            print("failed to find a valid IK solution")
            return False, None
        
    
    # then plan using ompl
    assert len(target_joint_angle) == ompl_robot.num_dim
    for idx in range(ompl_robot.num_dim):
        print("joint: ", idx, " lower limit: ", ompl_robot.joint_bounds[idx][0], " upper limit: ", ompl_robot.joint_bounds[idx][1], " target: ", target_joint_angle[idx])
        assert (ompl_robot.joint_bounds[idx][0] <= target_joint_angle[idx]) & (target_joint_angle[idx] <= ompl_robot.joint_bounds[idx][1])

    ompl_robot.set_state(current_joint_angles)
    res, path = pb_ompl_interface.plan(target_joint_angle)
    ompl_robot.set_state(current_joint_angles)

    if not res:
        print("motion planning failed to find a path")

    return res, path

def motion_planning_joint_angle(env, target_joint_angle, planner="BITstar", obstacles=[], allow_collision_links=[], panda_slider=True):
    ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body)
    pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links)
    pb_ompl_interface.set_planner(planner)
        
    #  plan using ompl
    assert len(target_joint_angle) == ompl_robot.num_dim
    for idx in range(ompl_robot.num_dim):
        print("joint: ", idx, " lower limit: ", ompl_robot.joint_bounds[idx][0], " upper limit: ", ompl_robot.joint_bounds[idx][1], " target: ", target_joint_angle[idx])
        assert (ompl_robot.joint_bounds[idx][0] <= target_joint_angle[idx]) & (target_joint_angle[idx] <= ompl_robot.joint_bounds[idx][1])

    res, path = pb_ompl_interface.plan(target_joint_angle)
    
    if not res:
        print("motion planning failed to find a path")

    return res, path