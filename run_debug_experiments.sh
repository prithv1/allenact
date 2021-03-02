# Script to check if debug experiments work or not

# PointNav (debug)
sudo python main.py \
    -o storage/pointnav-robothor-depth \
    -b projects/pointnav_baselines/experiments/robothor/ \
    pointnav_robothor_depth_simpleconvgru_ddppo

# PointNav on the entire training split (RGB)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor_sim2sim_v3 pointnav_s2s_robothor_vanilla_rgb_resnet_ddppo \
    -s 12345 \
    -et test_1

# PointNav on the entire training split (RGBD)
sudo python main.py \
    -o storage/robothor-pointnav-rgbd-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor_sim2sim_v3 pointnav_s2s_robothor_vanilla_rgbd_resnet_ddppo \
    -s 12345 \
    -et test_1

# ObjectNav on the entire training split (RGB)
sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor_sim2sim_v3 objectnav_s2s_robothor_vanilla_rgb_resnet_ddppo \
    -s 12345 \
    -et test_1

# ObjectNav on the entire training split (RGBD)
sudo python main.py \
    -o storage/robothor-objectnav-rgbd-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor_sim2sim_v3 objectnav_s2s_robothor_vanilla_rgbd_resnet_ddppo \
    -s 12345 \
    -et test_1

# Evaluate ObjectNav pre-trained checkpoint
sudo python main.py \
    -o storage/robothor-objectnav-rgbd-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor_sim2sim_v3 objectnav_s2s_robothor_vanilla_rgbd_resnet_ddppo \
    -c pretrained_model_ckpts/robothor-objectnav-challenge-2021/Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO/2021-02-09_22-35-15/exp_Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO_0.2.0a_300M__stage_00__steps_000170207237.pt \
    -t 2021-02-09_22-35-15 \
    -et test_1