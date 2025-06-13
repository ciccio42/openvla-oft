# conda create --name openvla_robosuite python=3.9.23 -y
# conda activate openvla_robosuite
# pip install -r openvla_requirements.txt
# pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# pip install git+https://github.com/moojink/transformers-openvla-oft.git
# cd tasks/training
# pip install -e .
# pip install pyquaternion

# cd ../../
# conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# conda deactivate openvla_robosuite
# conda activate openvla_robosuite
# pip install setuptools==65.5.0
# conda install -c conda-forge glew -y
# conda install -c conda-forge mesalib -y
# conda install -c anaconda mesa-libgl-cos6-x86_64 -y
# conda install -c menpo glfw3 -y
# conda install libgcc -y
# conda install patchelf -y
# conda install -c anaconda mesa-libegl-cos6-x86_64 -y
# conda install -c conda-forge gcc==12.1.0 -y
# conda install -c conda-forge gxx_linux-64 -y
# conda install -c conda-forge xorg-libx11 -y

# pip install robosuite==1.4.1

cd tasks
pip install --user .
cd training
pip install --user .
cd ..
