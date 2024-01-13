import yaml
from execute import execute_primitive
from manipulation.utils import save_numpy_as_gif, load_gif
import subprocess
import numpy as np
import time, datetime
import os
import json

def execute_multiple_try(
            task_config_path, 
            time_string=None, resume=False, # these two are combined for resume training.
            training_algo='cem', 
            gui=False, 
            randomize=False, # whether to randomize the initial state of the environment.
            use_bard=True, # whether to use the bard to verify the retrieved objects.
            use_gpt_size=True, # whether to use the size from gpt.
            use_gpt_joint_angle=True, # whether to initialize the joint angle from gpt.
            use_gpt_spatial_relationship=True, # whether to use the spatial relationship from gpt.
            obj_id=0, # which object to use from the list of possible objects.
            use_motion_planning=True,
            use_distractor=False,
            skip=[], # which substeps to skip.
            num_try=8,
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

    last_restore_state_file = None
    all_rgbs = []
    for step_idx, (substep, substep_type, action_space) in enumerate(zip(substeps, substep_types, action_spaces)):
        if (skip is not None) and (step_idx < len(skip)) and int(skip[step_idx]):
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
                                                 gui=gui, randomize=randomize, use_bard=use_bard, obj_id=obj_id, 
                                                 use_gpt_size=use_gpt_size, use_gpt_joint_angle=use_gpt_joint_angle,
                                                 use_gpt_spatial_relationship=use_gpt_spatial_relationship,
                                                 use_distractor=use_distractor)
                last_restore_state_file = states[-1]
                all_rgbs.extend(rgbs)
                save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(save_path, "execute"))

        
        if substep_type == "reward":
            save_path = os.path.join(solution_path, training_algo, time_string, substep.replace(" ", "_"))
            # call execute.py multiple times to learn the reward
            processes = []
            for learning_try in range(num_try):
                try_save_path = os.path.join(save_path, "try_" + str(learning_try))
                if not os.path.exists(try_save_path):
                    os.makedirs(try_save_path, exist_ok=True)

                cmd = ["python", "execute.py", "--task_config_path", task_config_path, "--only_learn_substep", str(step_idx), "--reward_learning_save_path", try_save_path, 
                       "--last_restore_state_file", last_restore_state_file]
                # Spawn the subprocesses
                proc = subprocess.Popen(cmd)
                processes.append(proc)
                time.sleep(5)

            # Wait for all subprocesses to finish
            for proc in processes:
                proc.wait()

            best_return = -np.inf
            best_idx = None
            for learning_try in range(num_try):
                best_state_path = os.path.join(save_path, "try_" + str(learning_try), "best_state")
                all_return_files = [x for x in os.listdir(best_state_path) if x.endswith(".txt")]
                all_return = [float(x.split("_")[1][:-4]) for x in all_return_files]
                if len(all_return) > 0:
                    highest_return = max(all_return)
                    if highest_return > best_return:
                        best_return = highest_return
                        best_idx = learning_try

            best_state_path = os.path.join(save_path, "try_" + str(best_idx), "best_state")
            all_pkl_files = [x for x in os.listdir(best_state_path) if x.endswith(".pkl")]
            all_pkl_files = sorted(all_pkl_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            last_restore_state_file = os.path.join(best_state_path, all_pkl_files[-1])
            all_rgbs.extend(load_gif(os.path.join(best_state_path, "best.gif")))
            os.system("cp -r {} {}".format(best_state_path, save_path + "/"))
            os.system("cp -r {} {}".format(os.path.join(save_path, "try_" + str(best_idx),  "best_model"), save_path + "/"))

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
    parser.add_argument('--time_string', type=str, default=None) # which folder to use to resume training.
    parser.add_argument('--gui', type=int, default=0) 
    parser.add_argument('--randomize', type=int, default=1) # whether to randomize roation of objects in the scene.
    parser.add_argument('--obj_id', type=int, default=None) # which object from the list of possible objects to use.
    parser.add_argument('--use_bard', type=int, default=1) # whether to use bard filtered objects.
    parser.add_argument('--use_gpt_size', type=int, default=1) # whether to use size outputted from gpt.
    parser.add_argument('--use_gpt_spatial_relationship', type=int, default=1) # whether to use gpt spatial relationship.
    parser.add_argument('--use_gpt_joint_angle', type=int, default=1) # whether to use initial joint angle output from gpt.
    parser.add_argument('--run_training', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--use_motion_planning', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--use_distractor', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--skip', nargs="+", default=[]) # if to train or just to build the scene.
    parser.add_argument('--num_try', type=int, default=5) # if to train or just to build the scene.
    args = parser.parse_args()

    task_config_path = args.task_config_path
    execute_multiple_try(task_config_path, 
            resume=args.resume, 
            training_algo=args.training_algo, 
            time_string=args.time_string, 
            gui=args.gui, 
            randomize=args.randomize,
            use_bard=args.use_bard,
            use_gpt_size=args.use_gpt_size,
            use_gpt_joint_angle=args.use_gpt_joint_angle,
            use_gpt_spatial_relationship=args.use_gpt_spatial_relationship,
            obj_id=args.obj_id,
            use_motion_planning=args.use_motion_planning,
            use_distractor=args.use_distractor,
            skip=args.skip,
            num_try=args.num_try,
    )