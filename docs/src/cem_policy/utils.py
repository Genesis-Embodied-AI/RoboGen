import pickle
from moviepy.editor import ImageSequenceClip
import os
import pybullet as p

def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.mp4'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    # clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    # clip.write_gif(filename, fps=fps)

    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_videofile(filename, bitrate='50000k', fps=fps, logger=None)

    return clip

def save_env(env, save_path=None):
    object_joint_angle_dicts = {}
    object_joint_name_dicts = {}
    object_link_name_dicts = {}
    for obj_name, obj_id in env.urdf_ids.items():
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        object_joint_angle_dicts[obj_name] = []
        object_joint_name_dicts[obj_name] = []
        object_link_name_dicts[obj_name] = []
        for link_idx in range(0, num_links):
            joint_angle = p.getJointState(obj_id, link_idx, physicsClientId=env.id)[0]
            object_joint_angle_dicts[obj_name].append(joint_angle)
            joint_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[1].decode('utf-8')
            object_joint_name_dicts[obj_name].append(joint_name)
            link_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[12].decode('utf-8')
            object_link_name_dicts[obj_name].append(link_name)

    object_base_position = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_position[obj_name] = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)[0]

    object_base_orientation = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_orientation[obj_name] = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)[1]

    state = {
        'object_joint_angle_dicts': object_joint_angle_dicts,
        'object_joint_name_dicts': object_joint_name_dicts,
        'object_link_name_dicts': object_link_name_dicts,
        'object_base_position': object_base_position,
        'object_base_orientation': object_base_orientation,     
        'done': env.done,
        'time_step': env.time_step,
        'urdf_paths': env.urdf_paths,
        'urdf_scales': env.urdf_scales,
    }

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    return state


def load_env(env, load_path=None, state=None):
    # print("state is: ", state)

    if load_path is not None:
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
    ### set env to stored object position and orientation
    for obj_name, obj_id in env.urdf_ids.items():
        p.resetBasePositionAndOrientation(obj_id, state['object_base_position'][obj_name], state['object_base_orientation'][obj_name], physicsClientId=env.id)

    ### set env to stored object joint angles
    for obj_name, obj_id in env.urdf_ids.items():
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        for link_idx in range(0, num_links):
            joint_angle = state['object_joint_angle_dicts'][obj_name][link_idx]
            p.resetJointState(obj_id, link_idx, joint_angle, physicsClientId=env.id)

    env.done = state['done']
    env.time_step = state['time_step']

    if "urdf_paths" in state:
        env.urdf_paths = state["urdf_paths"]

    if "urdf_scales" in state:
        env.urdf_scales = state["urdf_scales"]

    return state
