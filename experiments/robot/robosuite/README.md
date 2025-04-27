```bash
conda config --set channel_priority strict
conda create env -f robosuite_environment.yml -n openvla_robosuite
```

```bash
export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

git clone https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
```


```bash
cd tasks
pip install --user .
```

```bash
pip install torch torchvision torchaudio
cd ../../../openvla-oft
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```