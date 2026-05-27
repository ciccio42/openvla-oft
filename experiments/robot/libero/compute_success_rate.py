import os, glob
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Directory containing rollout files')
    args = parser.parse_args()

    run_dirs = glob.glob(args.run_dir + '/run_*')
    run_dirs.sort(key=lambda x: int(x.split('run_')[-1]))
    
    for run_dir in run_dirs:
        print(f"Processing run directory: {run_dir}")
        rollout_files = glob.glob(os.path.join(run_dir, '*npy'))
    
        rollout_files.sort(key=lambda x: int(os.path.basename(x).split('episode=')[-1].split('--')[0]))
        
        success_flags = []
        for file in rollout_files:
            print(f"Processing episode: {file.split('episode=')[-1].split('--')[0]}")    
            if 'success=True' in file:
                success_flags.append(1)
            elif 'success=False' in file:
                success_flags.append(0)

    success_rate = np.mean(success_flags)
    print(f"Success Rate: {success_rate}")
    # write file with success rate
    with open(os.path.join(args.run_dir, 'success_rate.txt'), 'w') as f:
        f.write(f"Success Rate: {success_rate}\n")