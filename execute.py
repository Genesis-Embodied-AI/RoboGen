import yaml
import os
from RL.ray_learn import run_RL
import numpy as np
import pybullet as p
import time, datetime
import json
from manipulation.utils import save_numpy_as_gif, save_env, take_round_images, build_up_env, load_gif

def execute_primitive(task_config, solution_path, substep, last_restore_state_file, save_path, 
                      gui=False, randomize=False, obj_id=0):
    # build the env
    task_name = substep.replace(" ", "_")
    env, safe_config = build_up_env(task_config, solution_path, task_name, last_restore_state_file, 
                                    render=gui, randomize=randomize, obj_id=obj_id)
    env.primitive_save_path = save_path

    # execute the primitive
    max_retry = 1
    cnt = 0

    # we retry at most 10 times till we get a successful execution.
    while cnt < max_retry:
        env.reset()
        rgbs, states, success = env.execute()
        if success:
            break
        cnt += 1
    
    p.disconnect(env.id)

    return rgbs, states

def test_env(solution_path, time_string, substeps, action_spaces, meta_info, randomize=False, obj_id=0, gui=False, move_robot=False,):
    if not move_robot:
        save_path = os.path.join(solution_path, "blip2", time_string)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(solution_path, "teaser", time_string)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    substep = substeps[0].lstrip().rstrip()
    action_space = action_spaces[0].lstrip().rstrip()
    task_name = substep.replace(" ", "_")
    env, safe_config = build_up_env(
        task_config_path, solution_path, task_name, None, return_env_class=False, 
        action_space=action_space,
        render=gui, randomize=randomize, 
        obj_id=obj_id,
    )
    env.reset()
    center = None
    if env.use_table:
        center = np.array([0, 0, 0.4])
    else:
        for name in env.urdf_ids:
            if name in ['robot', 'plane', 'init_table']:
                continue
            if env.urdf_types[name] != "urdf":
                continue
            object_id = env.urdf_ids[name]
            min_aabb, max_aabb = env.get_aabb(object_id)
            center = (min_aabb + max_aabb) / 2
            break
    if center is None:
        center = np.array([0, 0, 0.4])

    name = None
    for obj_name in env.urdf_types:
        if env.urdf_types[obj_name] == "urdf":
            name = obj_name
            break
    if move_robot:
        from manipulation.gpt_primitive_api import approach_object
        env.primitive_save_path = save_path
        primitive_rgbs, primitive_states = approach_object(env, name)

    rgbs, depths = take_round_images(env, center=center, distance=1.6, azimuth_interval=5)
    if move_robot:
        all_rgbs = primitive_rgbs + rgbs
    else:
        all_rgbs = rgbs
    save_numpy_as_gif(np.array(all_rgbs), "{}/{}.gif".format(save_path, "construction"), fps=10)
    save_env(env, os.path.join(save_path, "env.pkl"))
    with open(os.path.join(save_path, "meta_info.json"), 'w') as f:
        json.dump(meta_info, f)
    return 

def execute(task_config_path, 
            time_string=None, resume=False, # these two are combined for resume training.
            training_algo='RL_sac', 
            gui=False, 
            randomize=False, # whether to randomize the initial state of the environment.
            use_bard=True, # whether to use the bard to verify the retrieved objects.
            use_gpt_size=True, # whether to use the size from gpt.
            use_gpt_joint_angle=True, # whether to initialize the joint angle from gpt.
            use_gpt_spatial_relationship=True, # whether to use the spatial relationship from gpt.
            run_training=True, # whether to actually train the policy or just build the environment.
            obj_id=0, # which object to use from the list of possible objects.
            use_motion_planning=True,
            use_distractor=False,
            skip=[], # which substeps to skip.
            move_robot=False, # whether to move the robot to the initial state.
            only_learn_substep=None,
            reward_learning_save_path=None,
            last_restore_state_file=None,
):

    if time_string is None:
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    meta_info = {
        "using_motion_planning": use_motion_planning,
        "using_bard": use_bard,
        "using_gpt_size": use_gpt_size,
        "using_gpt_joint_angle": use_gpt_joint_angle,
        "using_gpt_spatial_relationship": use_gpt_spatial_relationship,
        "obj_id": obj_id,
        "use_distractor": use_distractor
    }
    
    all_last_state_files = []

    with open(task_config_path, 'r') as file:
        task_config = yaml.safe_load(file)

    solution_path = None
    for obj in task_config:
        if "solution_path" in obj:
            solution_path = obj["solution_path"]
            break

    if not os.path.exists(solution_path):
        os.makedirs(solution_path, exist_ok=True)

    experiment_path = os.path.join(solution_path, "experiment")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    with open(os.path.join(experiment_path, "meta_info_{}.json".format(time_string)), 'w') as f:
        json.dump(meta_info, f)

    all_substeps = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps, 'r') as f:
        substeps = f.readlines()
    print("all substeps:\n {}".format("".join(substeps)))
    
    substep_types = os.path.join(solution_path, "substep_types.txt")
    with open(substep_types, 'r') as f:
        substep_types = f.readlines()
    print("all substep types:\n {}".format("".join(substep_types)))

    action_spaces = os.path.join(solution_path, "action_spaces.txt")
    with open(action_spaces, 'r') as f:
        action_spaces = f.readlines()
    print("all action spaces:\n {}".format("".join(action_spaces)))

    if not run_training:
        test_env(solution_path, time_string, substeps, action_spaces, meta_info, randomize=randomize, obj_id=obj_id, gui=gui, move_robot=move_robot)
        exit()

    all_rgbs = []
    for step_idx, (substep, substep_type, action_space) in enumerate(zip(substeps, substep_types, action_spaces)):
        if (skip is not None) and (step_idx < len(skip)) and int(skip[step_idx]):
            print("skip substep: ", substep)
            continue
        if only_learn_substep is not None and step_idx != only_learn_substep:
            print("skip substep: ", substep)
            continue

        substep = substep.lstrip().rstrip()
        substep_type = substep_type.lstrip().rstrip()
        action_space = action_space.lstrip().rstrip()
        print("executing for substep:\n {} {}".format(substep, substep_type))


        if substep_type == "primitive" and use_motion_planning:
            save_path = os.path.join(solution_path, "primitive_states", time_string, substep.replace(" ", "_"))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            all_files = os.listdir(save_path)
            all_pkl_files = [f for f in all_files if f.endswith(".pkl")]
            gif_path = os.path.join(save_path, "execute.gif")
            if os.path.exists(gif_path) and resume:
                print("final state already exists, skip {}".format(substep))
                sorted_pkl_files = sorted(all_pkl_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
                last_restore_state_file = os.path.join(save_path, sorted_pkl_files[-1])
                all_rgbs.extend(load_gif(gif_path))
            else:
                rgbs, states = execute_primitive(task_config_path, solution_path, substep, last_restore_state_file, save_path, 
                                                 gui=gui, randomize=randomize, obj_id=obj_id,)
                last_restore_state_file = states[-1]
                all_rgbs.extend(rgbs)
                save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(save_path, "execute"))

        
        if substep_type == "reward":
            save_path = os.path.join(solution_path, training_algo, time_string, substep.replace(" ", "_"))
            if reward_learning_save_path is not None:
                save_path = reward_learning_save_path

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            all_files = os.listdir(save_path)

            pkl_dir = os.path.join(save_path, "best_state")
            gif_path = os.path.join(save_path, "execute.gif")
            if os.path.exists(gif_path) and resume:
                all_files = os.listdir(pkl_dir)
                all_pkl_files = [f for f in all_files if f.endswith(".pkl")]
                sorted_pkl_files = sorted(all_pkl_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
                print("final state already exists, skip {}".format(substep))
                last_restore_state_file = os.path.join(pkl_dir, sorted_pkl_files[-1])
                all_rgbs.extend(load_gif(gif_path))
            else:
                algo = training_algo.split("_")[1]
                task_name = substep.replace(" ", "_")
                best_model_path, rgbs, state_files = run_RL(task_config_path, solution_path, task_name, 
                                                            last_restore_state_file, save_path=save_path, action_space=action_space,
                                                            algo=algo, render=gui, timesteps_total=1000000, 
                                                            randomize=randomize,
                                                            use_bard=use_bard,
                                                            obj_id=obj_id,
                                                            use_gpt_size=use_gpt_size,
                                                            use_gpt_joint_angle=use_gpt_joint_angle,
                                                            use_gpt_spatial_relationship=use_gpt_spatial_relationship,
                                                            use_distractor=use_distractor,
                                                            )
                last_restore_state_file = state_files[-1]
                all_rgbs.extend(rgbs)
                save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(save_path, "execute"))

            if only_learn_substep is not None:
                return

        all_last_state_files.append(str(last_restore_state_file))
        with open(os.path.join(experiment_path, "all_last_state_files_{}.txt".format(time_string)), 'w') as f:
            f.write("\n".join(all_last_state_files))

    # save the final gif
    save_path = os.path.join(solution_path)
    save_numpy_as_gif(np.array(all_rgbs), "{}/{}-{}.gif".format(save_path, "all", time_string))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config_path', type=str, default=None)
    parser.add_argument('--training_algo', type=str, default="RL_sac")
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--time_string', type=str, default=None)
    parser.add_argument('--gui', type=int, default=0) 
    parser.add_argument('--randomize', type=int, default=0) # whether to randomize roation of objects in the scene.
    parser.add_argument('--obj_id', type=int, default=0) # which object from the list of possible objects to use.
    parser.add_argument('--use_bard', type=int, default=1) # whether to use bard filtered objects.
    parser.add_argument('--use_gpt_size', type=int, default=1) # whether to use size outputted from gpt.
    parser.add_argument('--use_gpt_spatial_relationship', type=int, default=1) # whether to use gpt spatial relationship.
    parser.add_argument('--use_gpt_joint_angle', type=int, default=1) # whether to use initial joint angle output from gpt.
    parser.add_argument('--run_training', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--use_motion_planning', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--use_distractor', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--skip', nargs="+", default=[]) # if to train or just to build the scene.
    parser.add_argument('--move_robot', type=int, default=0) # if to train or just to build the scene.
    parser.add_argument('--only_learn_substep', type=int, default=None) # if to run learning for a substep.
    parser.add_argument('--reward_learning_save_path', type=str, default=None) # where to store the learning result of RL training. 
    parser.add_argument('--last_restore_state_file', type=str, default=None) # whether to start from a specific state.
    args = parser.parse_args()

    task_config_path = args.task_config_path
    execute(task_config_path, resume=args.resume, training_algo=args.training_algo, time_string=args.time_string, 
            gui=args.gui, 
            randomize=args.randomize,
            use_bard=args.use_bard,
            use_gpt_size=args.use_gpt_size,
            use_gpt_joint_angle=args.use_gpt_joint_angle,
            use_gpt_spatial_relationship=args.use_gpt_spatial_relationship,
            run_training=args.run_training,
            obj_id=args.obj_id,
            use_motion_planning=args.use_motion_planning,
            use_distractor=args.use_distractor,
            skip=args.skip,
            move_robot=args.move_robot,
            only_learn_substep=args.only_learn_substep,
            reward_learning_save_path=args.reward_learning_save_path,
            last_restore_state_file=args.last_restore_state_file
    )