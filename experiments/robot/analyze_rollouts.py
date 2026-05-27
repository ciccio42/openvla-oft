import numpy as np
import glob, os, json

PATH_TO_ROLLOUTS = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero_plus/rollouts/libero_object/change_spawn_False_train_False/run_0"

if __name__ == "__main__":
    rollouts_files = glob.glob(os.path.join(PATH_TO_ROLLOUTS, "*.npy"))
    rollouts_files = sorted(rollouts_files, key=lambda x: int(os.path.basename(x).split("episode=")[-1].split("--")[0]))
    
    for rollout_file in rollouts_files:
        episode_id = os.path.basename(rollout_file).split("episode=")[-1].split("--")[0]
        print(f"Analyzing episode {episode_id}...")
        rollout = np.load(rollout_file, allow_pickle=True)
        # print(f"Keys in rollout: {rollout.item().keys()}")
        print(f"\tCommand sequence: {rollout.item()['task_command']}")