import argparse
import glob
import pickle as pkl
import cv2
import numpy as np
import os
import debugpy

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_pkl', default="/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--30000_chkpt/rollout_pick_place")
    parser.add_argument('--output_dir', default="/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--30000_chkpt/rollout_pick_place/videos", help="Directory to save the videos")
    args = parser.parse_args()
    
    pkl_files = glob.glob(f"{args.path_to_pkl}/*.pkl")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # print("Wait for debugger to attach...")
    # debugpy.listen(('0.0.0.0', 5678))
    # debugpy.wait_for_client()

    for pkl_file in pkl_files:
        print(f"Loading {pkl_file}")
        with open(pkl_file, 'rb') as f:
            traj = pkl.load(f)

        # Check first image size
        task_description = traj[0]['obs']['task_description']
        first_img = traj[0]['obs']['camera_front_image']
        height, width, layers = first_img.shape
        
        video_name = os.path.join(args.output_dir, os.path.basename(pkl_file).replace('.pkl', '.mp4'))

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))  # 10 FPS

        for t in range(len(traj)-1):
            img = traj[t]['obs']['camera_front_image']  # Extract frame
            img = np.array(img[:,:,::-1])  # Convert RGB to BGR for OpenCV

            
            # Add task_description text at the bottom
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (0, 255, 0)  # Green text
            thickness = 1
            text_size, _ = cv2.getTextSize(task_description, font, font_scale, thickness)
            text_width, text_height = text_size

            position = (10, height - 10)  # 10 pixels from bottom

            # Optionally, draw a black rectangle behind the text for better readability
            cv2.rectangle(img, 
                          (position[0] - 5, position[1] - text_height - 5), 
                          (position[0] + text_width + 5, position[1] + 5), 
                          (0, 0, 0), 
                          -1)
            
            cv2.putText(img, task_description, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

            
            video.write(img)

        video.release()
        print(f"Saved video to {video_name}")
            
            
        
    