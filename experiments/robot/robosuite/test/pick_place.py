from robosuite_utils import startup_env, check_reach, check_pick, check_bin, task_run_action, ENV_OBJECTS 
import numpy as np
from PIL import Image
from robosuite.utils.transform_utils import quat2mat, mat2euler, euler2mat, quat2axisangle, mat2quat


OBJECT_SET = 2


def pick_place_eval(cfg, model, env, variation_id, max_T, resize_size, task_description: str,
                    action_head=None, proprio_projector=None, noisy_action_projector=None,
                    processor=None, use_film=False, task_name = 'pick_place'):
    

    start_up_env_return = startup_env(
                        model=model,
                        env=env,
                        variation_id=variation_id,
                        )
    done, states, images, obs, traj, tasks, current_gripper_pose = start_up_env_return
    
    img = Image.fromarray(obs['camera_front_image'])
    img.save("first_frame.jpg")
    
    
    n_steps = 0

    object_name_target = env.objects[env.object_id].name.lower()
    obj_delta_key = object_name_target + '_to_robot0_eef_pos'
    obj_key = object_name_target + '_pos'
    start_z = obs[obj_key][2]
    
    
    print(f"Max-t {max_T}")
    tasks["reached_wrong"] = 0.0
    tasks["picked_wrong"] = 0.0
    tasks["place_wrong"] = 0.0
    tasks["place_wrong_correct_obj"] = 0.0
    tasks["place_wrong_wrong_obj"] = 0.0
    tasks["place_correct_bin_wrong_obj"] = 0.0
    elapsed_time = 0.0
    
    action = np.zeros((7,), dtype=np.float32)
    
    while not done:

        tasks['reached'] = int(check_reach(threshold=0.03,
                                           obj_distance=obs[obj_delta_key][:2],
                                           current_reach=tasks['reached']
                                           ))

        tasks['picked'] = int(check_pick(threshold=0.05,
                                         obj_z=obs[obj_key][2],
                                         start_z=start_z,
                                         reached=tasks['reached'],
                                         picked=tasks['picked']))

        for obj_id, obj_name, in enumerate(env.env.obj_names):
            if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                if check_reach(threshold=0.03,
                                obj_distance=obs[obj_name.lower() +
                                                '_to_robot0_eef_pos'],
                                current_reach=tasks.get(
                                    "reached_wrong", 0.0)
                                ):
                    tasks['reached_wrong'] = 1.0
                if check_pick(threshold=0.05,
                                obj_z=obs[obj_name.lower() + "_pos"][2],
                                start_z=start_z,
                                reached=tasks['reached_wrong'],
                                picked=tasks.get(
                                    "picked_wrong", 0.0)):
                    tasks['picked_wrong'] = 1.0

        if n_steps == 0:
            gripper_closed = 0.0
        else:
            gripper_closed = 0 if action[-1] == -1 else 1

        action_chunk,  time_action = task_run_action(cfg = cfg, 
                        model = model, 
                        obs = obs, 
                        resize_size = resize_size, 
                        gripper_closed = gripper_closed, 
                        env = env, 
                        task_description = task_description,
                        processor = processor, 
                        action_head = action_head,
                        proprio_projector = proprio_projector, 
                        noisy_action_projector = noisy_action_projector, 
                        use_film = use_film,
                        task_name = task_name
                        )
    
        for action in action_chunk:
            
            # get current gripper position
            action_world = np.zeros(7)
            if 'abs_pose' in cfg.task_suite_name:
                action_world[0:3] = action[0:3]
            else:
                action_world[0:3] = obs['eef_pos'] + action[0:3]
            
            action_rpy = action[3:6]
            action_world[3:6] = quat2axisangle(mat2quat(euler2mat(-action_rpy)))
            # current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
            #         env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            # action_world[3:6] =current_gripper_orientation
            
            print(f"action_world {action[6]}")
            if action[6] >= 0.97:
                # action_world[2] = action[2] - 0.05
                action_world[6] = 1
            else:
                action_world[6] = -1
            
            try:
                n_steps += 1
                obs, reward, env_done, info = env.step(action_world)
                
                if reward == 1:
                    print("Success!")
                elif env_done:
                    print(f"Episode finished! Env done: {env_done}")
            except:
                print("Episode finished! Raised an exception")
                done = True
                tasks['success'] = 0
                break
        
            image_step = obs['camera_front_image']
            image_step = Image.fromarray(image_step)
            image_step.save("step.jpg")

            # current_step = obs['camera_front_image']
            # img = Image.fromarray(current_step)
            # img.save(f"current_step.jpg")
            
            
            traj.append(obs, reward, done, info, action)
            elapsed_time += time_action
            
            tasks['success'] = int(reward or tasks['success'])
        
        
            # check if the object has been placed in a different bin
            if not tasks['success']:
                for i, bin_name in enumerate(ENV_OBJECTS['pick_place']['bin_names']):
                    if i != obs['target-box-id']:
                        bin_pos = obs[f"{bin_name}_pos"]
                        if check_bin(threshold=0.03,
                                        bin_pos=bin_pos,
                                        obj_pos=obs[f"{object_name_target}_pos"],
                                        current_bin=tasks.get(
                                            "place_wrong_correct_obj", 0.0)
                                        ):
                            tasks["place_wrong_correct_obj"] = 1.0

                for obj_id, obj_name, in enumerate(env.env.obj_names):
                    if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                        for i, bin_name in enumerate(ENV_OBJECTS['pick_place']['bin_names']):
                            if i != obs['target-box-id']:
                                bin_pos = obs[f"{bin_name}_pos"]
                                if check_bin(threshold=0.03,
                                                bin_pos=bin_pos,
                                                obj_pos=obs[f"{obj_name}_pos"],
                                                current_bin=tasks.get(
                                                    "place_wrong_wrong_obj", 0.0)
                                                ):
                                    tasks["place_wrong_wrong_obj"] = 1.0
                            else:
                                bin_pos = obs[f"{bin_name}_pos"]
                                if check_bin(threshold=0.03,
                                                bin_pos=bin_pos,
                                                obj_pos=obs[f"{obj_name}_pos"],
                                                current_bin=tasks.get(
                                                    "place_correct_bin_wrong_obj", 0.0)
                                                ):
                                    tasks["place_correct_bin_wrong_obj"] = 1.0

            
            if env_done or reward or n_steps > max_T:
                done = True
                break
            
    print(tasks)
    env.close()
    mean_elapsed_time = elapsed_time/n_steps
    print(f"Mean elapsed time {mean_elapsed_time}")
    
    del env
    del states
    del images
    del model

    return traj, tasks