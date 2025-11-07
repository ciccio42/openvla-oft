# OpenVLA-OFT in the VLA-Benchmark

# Run train
## 1. Download Dataset
Links to datasets will be available after paper acceptance

## 2. Download pre-trained model
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openvla/openvla-7b', local_dir='[PATH-TO-CHECKPOINTS]')"
```

## 3. Run Train
**Note** </br>
Before running finetune change:
* **RUN_ROOT_DIR** to [PATH-TO-CHECKPOINTS]
* **DATASET_NAME** to desired dataset name. Possible values: ur5e_pick_place_delta_all, ur5e_pick_place_removed_spawn_regions, ur5e_pick_place_rm_central_spawn, ur5e_pick_place_rm_one_spawn, ur5e_pick_place_delta_removed_0_5_10_15, ur5e_pick_place_rm_12_13_14_15   
```bash
cd vla-scripts
nohup python run_finetune.py
```

# Run validation

## 1. Run merge_lora_weights
```bash
sbatch merge_lora_weights_and_save.sh [PATH-TO-OPENVLA-CHECKPOINT] [PATH-TO-FINETUNED-CHECKPOINT]
```

## 2. Use VLA-Benchmark
See instruction [here]()

