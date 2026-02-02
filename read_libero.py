import os
#import rlds
#import envlogger
#from envlogger.backends import rlds_utils
#from envlogger.backends import tfds_backend_writer
#from envlogger.testing import catch_env
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import json


# Set your table size in meters (e.g., 0.7m x 0.7m table)
TABLE_SIZE = (0.7, 0.7)  # height (y), width (x)
BDDL_FOLDER = "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO/libero/libero/bddl_files/libero_goal"


def heat_map(task_distribution, task_path, task_name):
    # Each px represents 0.5x0.5 cm (0.005m x 0.005m)
    px_resolution = 0.5  # in cm

    # Convert TABLE_SIZE to cm and then to pixels
    table_size_cm = np.array(TABLE_SIZE) * 100  # meters to cm
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)

    # Initialize table heatmap
    table_map = np.zeros((table_size_px[0], table_size_px[1]))  # shape: (height, width)

    # Loop through each episode's trajectory
    for episode_idx, trajectory in task_distribution.items():
        print(f"Processing episode: {episode_idx}")

        # Convert list of [x, y, z] to np.array and take only x, y
        trajectory = np.array(trajectory)[:, :2]  # shape: (T, 2)

        # Convert meters to cm and to pixels
        px_traj = (trajectory * 100 / px_resolution).astype(np.int32)

        # Translate coords: center of table is the middle of the image
        px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]  # x -> vertical axis (rows)
        px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]  # y -> horizontal axis (cols)

        # Clip to table bounds
        px_traj = px_traj[
            (px_traj[:, 0] >= 0) & (px_traj[:, 0] < table_map.shape[0]) &
            (px_traj[:, 1] >= 0) & (px_traj[:, 1] < table_map.shape[1])
        ]

        # Populate heatmap
        for x, y in px_traj:
            table_map[x, y] += 1

    # Set crop range in cm for visual focus (adjust as needed)
    y_min, y_max = -30, 30
    x_min, x_max = -35, 35
    task_title = task_name.replace("_", " ").title()

    # Convert to pixel bounds
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)

    # Crop table
    cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(f"Command: '{task_title}'")
    plt.xlabel("Y Axis (cm)")
    plt.ylabel("X Axis (cm)")

    norm = mcolors.LogNorm(vmin=1, vmax=np.max(cropped_map) if np.max(cropped_map) > 0 else 1)
    im = ax.imshow(cropped_map, cmap='plasma', origin='upper', norm=norm)
    # ax.invert_yaxis()  # <-- This flips the y-axis so (0,0) is bottom-left
    ax.invert_xaxis()  # Invert x-axis to match the coordinate system
    
    # Axis ticks (every 10 cm)
    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
    tick_labels_x = np.arange(x_min, x_max, 10)
    tick_labels_y = np.arange(y_min, y_max, 10)

    plt.xticks(ticks_x, tick_labels_x)
    plt.yticks(ticks_y, tick_labels_y)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory Density (log scale)")

    # Save
    os.makedirs(task_path, exist_ok=True)
    save_path = os.path.join(task_path, f"{task_name}_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {save_path}")

# Load dataset
recover_dataset_path = '/mnt/beegfs/a.cardamone7/datasets/modified_libero_rlds/libero_goal_no_noops/1.0.0'
builder = tfds.builder_from_directory(recover_dataset_path)
#builder = rlds_utils.maybe_recover_last_shard(builder)
print(f"Recovering dataset from {recover_dataset_path}")
loaded_dataset = builder.as_dataset(split='all', )

output_dir = os.path.join(recover_dataset_path,"episode_videos")
os.makedirs(output_dir, exist_ok=True)

# analyze_bddl_files()


# Initialize storage
all_actions = {}

# Process episodes
for e_idx, e in enumerate(loaded_dataset):
    # if e_idx >= 10:
    #     break  # Only process first 10 episodes

    print(f"\nProcessing episode {e_idx}")
    images = []

    task_str = None
    
    for t, step in enumerate(e['steps']):
        if t == 0:
            task_str = step['language_instruction'].numpy().decode('utf-8')
            if task_str not in all_actions:
                all_actions[task_str] = {}

            print(f"Task: {task_str}")
        
        if t == 0:
            all_actions[task_str][e_idx] = []

        # Extract image and state
        img = step['observation']['image'].numpy()
        pil_img = Image.fromarray(img)
        pos_state = step['observation']['state'][:3].numpy()

        all_actions[task_str][e_idx].append(pos_state)

        # Annotate image with task text
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), task_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        width, height = pil_img.size
        padding = 5
        text_position = ((width - text_width) // 2, height - text_height - padding)

        draw.rectangle(conda
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill=(0, 0, 0)
        )
        draw.text(text_position, task_str, fill=(255, 255, 255), font=font)

        images.append(np.array(pil_img))
        
    # save video for the episode
    video_dir = os.path.join(output_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{task_str.replace(' ', '_')}_ep{e_idx}.mp4")
    imageio.mimwrite(video_path, images, fps=10, codec='libx264')
    print(f"Saved video to {video_path}")
            


    


for task_str, episodes in all_actions.items():
    safe_name = task_str.replace(" ", "_").lower()
    heat_map(episodes, output_dir, safe_name)


# Axis labels
axis_labels = ['x', 'y', 'z']
action_dim = 3

# Plot combined image per task
for task_str, episodes in all_actions.items():
    print(f"\nPlotting combined trajectory figure for task: '{task_str}'")

    # Create a figure with 3 subplots (x, y, z)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Trajectory for Task: {task_str}", fontsize=16)

    for dim in range(action_dim):
        ax = axs[dim]
        for traj in episodes.values():
            traj_np = np.stack(traj)  # shape: (T, 3)
            ax.plot(traj_np[:, dim], alpha=0.6)

        ax.set_ylabel(f"{axis_labels[dim]} position")
        ax.grid(True)
    
    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for title

    # Create safe filename
    safe_task = task_str.replace(" ", "_").replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe_task}_trajectories_combined.png")
    plt.savefig(plot_path)
    plt.close()
    #plt.show()

    print(f"Saved combined trajectory plot to {plot_path}")
    
    
