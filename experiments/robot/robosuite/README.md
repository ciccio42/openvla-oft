Usefull [link](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)
<!-- ```bash
conda create --name openvla_robosuite python=3.9.23 -y
conda activate openvla_robosuite
pip install -r openvla_requirements.txt
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/moojink/transformers-openvla-oft.git
cd tasks/training
pip install -e .
pip install pyquaternion

conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate openvla_robosuite
conda activate openvla_robosuite
pip install setuptools==65.5.0
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c anaconda mesa-libgl-cos6-x86_64 -y
conda install -c menpo glfw3 -y
conda install libgcc -y
conda install patchelf -y
conda install -c anaconda mesa-libegl-cos6-x86_64 -y
conda install -c conda-forge gcc==12.1.0 -y
conda install -c conda-forge gxx_linux-64 -y
conda install -c conda-forge xorg-libx11 -y

pip install robosuite==1.4.1
```

```bash
export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

cd ../../
git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
```


```bash
cd tasks
pip install --user .
cd training
pip install --user .
cd ..
```
-->


```bash
conda env create -f robosuite_1_0_1.yaml 
pip install -r requirements.txt 
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/moojink/transformers-openvla-oft.git
cd tasks/training
pip install -e .
pip install pyquaternion

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

cd ../../
git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt

cd openvla-oft
pip install --user .
```