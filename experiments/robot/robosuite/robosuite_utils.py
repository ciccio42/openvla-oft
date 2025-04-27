# from multi_task_test.eval_functions import *
import os
import json
import sys
import cv2
import copy
import multi_task_robosuite_env as mtre
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from multi_task_robosuite_env.controllers.controllers.expert_button import \
    get_expert_trajectory as button_expert
from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_block_expert
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite import load_controller_config
from collections import deque
import tensorflow as tf
from robosuite.utils.transform_utils import quat2mat, mat2euler, euler2mat, quat2axisangle, mat2quat
import time
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from robot_utils import get_action

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_names': ['bin_box_1', 'bin_box_2', 'bin_box_3', 'bin_box_4'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
        'splitted_obj_names': ['green box', 'yellow box', 'blue box', 'red box'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15]},
    },
    'nut_assembly': {
        'obj_names': ['round-nut', 'round-nut-2', 'round-nut-3'],
        'peg_names': ['peg1', 'peg2', 'peg3'],
        'splitted_obj_names': ['grey nut', 'brown nut', 'blue nut'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },
    'stack_block': {
        'obj_names': ['cubeA', 'cubeB', 'cubeC'],
    },
    'button': {
        'obj_names': ['machine1_goal1', 'machine1_goal2', 'machine1_goal3',
                      'machine2_goal1', 'machine2_goal2', 'machine2_goal3'],
        'place_names': ['machine1_goal1_final', 'machine1_goal2_final', 'machine1_goal3_final',
                        'machine2_goal1_final', 'machine2_goal2_final', 'machine2_goal3_final']
    }
}


TASK_MAP = {
    'nut_assembly':  {
        'num_variations':   9,
        'env_fn':   nut_expert,
        'agent-teacher': ('UR5e_NutAssemblyDistractor', 'Panda_NutAssemblyDistractor'),
        'render_hw': (200, 360),
        'object_set': 1,
    },
    'pick_place': {
        'num_variations':   16,
        'num_variations_per_object':   4,
        'env_fn':   place_expert,
        'agent-teacher': ('UR5e_PickPlaceDistractor', 'Panda_PickPlaceDistractor'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 2,
    },
    'stack_block': {
        'num_variations':   6,
        'env_fn':   stack_block_expert,
        'agent-teacher': ('UR5e_BlockStacking', 'Panda_BlockStacking'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 1,
    },
    'button': {
        'num_variations':   6,
        'env_fn':   button_expert,
        'agent-teacher': ('UR5e_Button', 'Panda_Button'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 1,
    },
}


def build_env_context(env_name: str, controller_path: str, variation: int, seed: int, gpu_id: int):
    # load custom controller
    controller = load_controller_config(
        custom_fpath=controller_path)
    
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    
    agent_env = env_fn(agent_name,
                       controller_type=controller,
                       task=variation,
                       ret_env=True,
                       seed=seed,
                       gpu_id=gpu_id,
                       object_set=TASK_MAP[env_name]['object_set'])
    
    return agent_env
    
    
def get_eval_fn(env_name):

    sys.path.append(os.path.join(os.path.dirname(__file__), "test"))
    if "pick_place" in env_name:
        from test.pick_place import pick_place_eval
        return pick_place_eval
    elif "nut_assembly" in env_name:
        NotImplementedError
    elif "button" in env_name:
        NotImplementedError
    elif "stack" in env_name:
        NotImplementedError
    else:
        assert NotImplementedError
        
        
def startup_env(model,env,variation_id):

    done, states, images = False, [], []
    states = deque(states, maxlen=1)
    images = deque(images, maxlen=1)  # NOTE: always use only one frame
    
    while True:
        try:
            obs = env.reset()
            cv2.imwrite("pre_set.jpg", obs['camera_front_image'])
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([-1])), axis=-1)
            obs, reward, env_done, info = env.step(current_gripper_pose)

            break
        except:
            pass

    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    
    return done, states, images, obs, traj, tasks, current_gripper_pose


def check_pick(threshold: float, obj_z: float, start_z: float, reached: bool, picked: bool):
    return picked or (reached and obj_z - start_z > threshold)


def check_reach(threshold: float, obj_distance: np.array, current_reach: bool):
    return current_reach or np.linalg.norm(
        obj_distance) < threshold


def check_bin(threshold: float, bin_pos: np.array, obj_pos: np.array, current_bin: bool):
    bin_x_low = bin_pos[0]
    bin_y_low = bin_pos[1]
    bin_x_low -= 0.16 / 2
    bin_y_low -= 0.16 / 2

    bin_x_high = bin_x_low + 0.16
    bin_y_high = bin_y_low + 0.16
    # print(bin_pos, obj_pos)
    res = False
    if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and bin_pos[2] < obj_pos[2] < bin_pos[2] + 0.1
    ):
        res = True
    return (current_bin or res)


def check_peg(peg_pos: np.array, obj_pos: np.array, current_peg: bool):

    # print(bin_pos, obj_pos)
    res = False
    if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < 0.860 + 0.05
    ):
        res = True
    return res or current_peg

def prepare_observation(obs, resize_size, gripper_closed=0):
    img = obs['camera_front_image']
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
        
    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()

    eef_pose = np.zeros(6, dtype=np.float64)
    # convert gripper orientation to end effector orientation
    eef_quat = obs['eef_quat']
    R_ee_to_gripper = np.array([[.0, -1.0, .0], 
                                    [1.0, .0, .0], 
                                    [.0, .0, 1.0]])
    eef_mat = R_ee_to_gripper @ quat2mat(eef_quat)
    eef_euler = mat2euler(eef_mat)
    eef_pose[0:3] = obs['eef_pos']
    eef_pose[3:6] = eef_euler
    eef_pose = np.array(eef_pose, dtype=np.float64)


    # Prepare observations dict
    observation = {
        "full_image": img,
        'joint_positions': obs['joint_pos'],
        'eef_pose': eef_pose,
        'gripper_closed':gripper_closed, 
    }
    
    return observation, img


def get_action_robosuite(cfg, model, obs, resize_size, gripper_closed, task_description, processor, action_head, proprio_projector, noisy_action_projector, use_film):
    # Prepare observation
    observation, img = prepare_observation(obs, resize_size, gripper_closed=gripper_closed)
    
    action = get_action( cfg = cfg,
                model = model,
                obs = observation,
                task_label = task_description,
                processor = processor,
                action_head = action_head,
                proprio_projector = proprio_projector,
                noisy_action_projector = noisy_action_projector,
                use_film = use_film)
    
    return action


def task_run_action(cfg, model, obs, resize_size, gripper_closed, env, task_description, processor, action_head, proprio_projector, noisy_action_projector, use_film):
    
    elapsed_time = 0
    start = time.time()
    action_chunk = get_action_robosuite(cfg = cfg,
                        model = model,
                        obs = obs,
                        resize_size = resize_size,
                        gripper_closed = gripper_closed,
                        task_description = task_description,
                        processor = processor,
                        action_head = action_head,
                        proprio_projector = proprio_projector,
                        noisy_action_projector = noisy_action_projector,
                        use_film = use_film)
    
    end = time.time()
    elapsed_time = end-start
    
    for action in action_chunk:
        
        # get current gripper position
        action_world = np.zeros(7)
        action_world[0:3] = obs['eef_pos'] + action[0:3]
        
        # action_rpy = action[3:6]
        # action_world[3:6] = quat2axisangle(mat2quat(euler2mat(action_rpy)))
        current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
        action_world[3:6] =current_gripper_orientation
        
        action_world[6] = 1 if action[6] > 0.99 else -1
        
        try:
            obs, reward, env_done, info = env.step(action_world)
        except:
            print("Episode finished!")
            break
        
        image_step = obs['camera_front_image']
        image_step = Image.fromarray(image_step)
        image_step.save("step.jpg")
    
    return obs, reward, info, action, env_done, elapsed_time


#### -------- ---- ---- ---- ---- ---- ---- ---- ---- ####
#### Trajectory class
#### to store observations, actions, rewards, etc.
#### -------- ---- ---- ---- ---- ---- ---- ---- ---- ####


def _compress_obs(obs):
    for key in obs.keys():
        if 'image' in key:
            if len(obs[key].shape) == 3:
                okay, im_string = cv2.imencode('.jpg', obs[key])
                assert okay, "image encoding failed!"
                obs[key] = im_string
        if 'depth_norm' in key:
            assert len(
                obs[key].shape) == 2 and obs[key].dtype == np.uint8, "assumes uint8 greyscale depth image!"
            depth_im = np.tile(obs[key][:, :, None], (1, 1, 3))
            okay, depth_string = cv2.imencode('.jpg', depth_im)
            assert okay, "depth encoding failed!"
            obs[key] = depth_string
    return obs


def _decompress_obs(obs):
    keys = ["camera_front_image"]
    for key in keys:
        if 'image' in key:
            try:
                decomp = cv2.imdecode(obs[key], cv2.IMREAD_COLOR)
                obs[key] = decomp
            except:
                pass
        if 'depth_norm' in key:
            obs[key] = cv2.imdecode(
                obs[key], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    return obs


class Trajectory:
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self.set_config_str(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(
            x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @ property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)

    def __getitem__(self, t):
        return self.get(t)

    def get(self, t, decompress=True):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"

        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        if decompress:
            obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t,
                        done=done_t, info=info_t, action=action_t)

        for k in list(ret_dict.keys()):
            if ret_dict[k] is None:
                ret_dict.pop(k)
        return ret_dict

    def change_obs(self, t, obs):
        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        self._data[t] = obs, reward_t, done_t, info_t, action_t

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @ property
    def config_str(self):
        return self._config_str

