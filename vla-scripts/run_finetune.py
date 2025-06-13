import os
import subprocess
import re
import time
import argparse
import glob
import shutil


CKPT_FOLDER = "openvla-7b"
BASE_MODEL_NAME = "openvla-7b"
RUN_ID_NOTE = "parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio"
RESUME = False
RESUME_STEP = 0
RUN_ROOT_DIR = "/home/rsofnc000/checkpoint_save_folder/open_vla"
DATASET_NAME = "ur5e_pick_place"

def get_highest_epoch(root_dir, model_name):
    highest_epoch = 0
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return highest_epoch

    print(f"Searching for folders in {root_dir} with model name {model_name}")
    folders = glob.glob(os.path.join(root_dir, f"{model_name}--*_chkpt"))
    folders.sort(key=lambda x: int(re.search(r'--(\d+)_chkpt', x).group(1)), reverse=True)
    highest_epoch_folder = folders[0]
    
    highest_epoch = highest_epoch_folder.split("--")[-1].split("_chkpt")[0]
    print(f"Highest epoch folder: {highest_epoch_folder}")
    
    return highest_epoch

def run_merge_lora_weights_script(last_epoch, model_name, new_last_epoch):
    if last_epoch == -1:
        base_checkpoint = os.path.join(RUN_ROOT_DIR, "openvla-7b")
    else:
        base_checkpoint = os.path.join(RUN_ROOT_DIR, f"{model_name}--{last_epoch}_chkpt")

    new_checkpoint = os.path.join(RUN_ROOT_DIR, f"{model_name}--{new_last_epoch}_chkpt")

    # copy modeling_prismatic.py from base_checkpoint to new_checkpoint
    modeling_prismatic_path = os.path.join(base_checkpoint, "modeling_prismatic.py")
    shutil.copyfile(modeling_prismatic_path, os.path.join(new_checkpoint, "modeling_prismatic.py"))
    
    result = subprocess.run(['sbatch', 'merge_lora_weights_and_save.sh', base_checkpoint, new_checkpoint], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running merge_lora_weights_and_save.sh: {result.stderr}")
        return

    # Extract job ID from sbatch output
    job_id = None
    for line in result.stdout.split('\n'):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break
    
    while True:
        print(f"Job {job_id} submitted. Waiting for merge completion...")
        result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error checking job status: {result.stderr}")
            return

        if job_id not in result.stdout:
            print(f"Job {job_id} completed.")
            break

        time.sleep(10)  # Wait for 10 seconds before polling again


def run_bash_script(first_run=False):
    global CKPT_FOLDER, RUN_ID_NOTE, RESUME_STEP, RESUME
    BASH_ARGUMENTS = [f"{CKPT_FOLDER}", f"{RUN_ID_NOTE}", f"{RESUME}", f"{RESUME_STEP}", f"{RUN_ROOT_DIR}", f"{DATASET_NAME}"]
    
    
    
    old_resume_step = -1
   
    for i in range(2, 20):
        
        if i != 1:
            model_name = f"{BASE_MODEL_NAME}+{DATASET_NAME}+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--{RUN_ID_NOTE}"
            new_resume_step = get_highest_epoch(RUN_ROOT_DIR, model_name)    
            print(f"Highest epoch: {new_resume_step}")        
            # run merge_lora_weights_and_save.sh
            run_merge_lora_weights_script(old_resume_step, model_name, new_resume_step)
            old_resume_step = new_resume_step
            RESUME_STEP = new_resume_step
            RESUME = True
            CKPT_FOLDER=f"{model_name}--{new_resume_step}_chkpt"
            BASH_ARGUMENTS = [f"{CKPT_FOLDER}", f"{RUN_ID_NOTE}", f"{RESUME}", f"{RESUME_STEP}", f"{RUN_ROOT_DIR}", f"{DATASET_NAME}"]
            print(f"Running bash script with arguments: {BASH_ARGUMENTS}")
        
        
            
        print(f"Running bash script with arguments: {BASH_ARGUMENTS}")
        result = subprocess.run(['sbatch', 'finetune.sh'] + BASH_ARGUMENTS, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return
            
        # Extract job ID from sbatch output
        job_id = None
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                break
        
        print(f"Job {job_id} submitted. Waiting for completion...")
        # Poll the job status using squeue
        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}")
                return

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.")
                break

            time.sleep(10)  # Wait for 10 seconds before polling again
           
        

if __name__ == "__main__":
    run_bash_script()        
    