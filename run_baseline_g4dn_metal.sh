# PointNav Baselines

# PointNav RGB-D baseline
python main.py \
    -o storage/robothor-pointnav-rgbd-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgbd_resnet_ddppo \
    -et v1_new_pointnav_rgbd_clean \
    -c storage/robothor-pointnav-rgbd-resnetgru-dppo/checkpoints/Pointnav-S2S-RoboTHOR-RGBD-ResNet-DDPPO/2020-10-23_22-02-20/exp_Pointnav-S2S-RoboTHOR-RGBD-ResNet-DDPPO__stage_00__steps_000005000889.pt


# PointNav RGB-D baseline (v2)
python main.py \
    -o storage/robothor-pointnav-rgbd-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgbd_resnet_ddppo \
    -et v4_new_pointnav_rgbd_clean_nR_128_nU_4_nTS_300M

# PointNav RGB baseline (v2)
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgb_resnet_ddppo \
    -et v5_new_pointnav_rgb_clean_nR_128_nU_4_nTS_75M


# ObjectNav RGB (v1)
python main.py \
    -o storage/robothor-objectnav-rgb-resnetgru-dppo \
    -b projects/objectnav_baselines/experiments/robothor objectnav_s2s_robothor_rgb_resnet_ddppo \
    -et v1_new_objectnav_rgb_clean_nR_128_nU_4_nTS_300M


# ObjectNav RGB-D (v2)
python main.py \
    -o storage/robothor-pointnav-rgbd-resnetgru-dppo \
    -b projects/objectnav_baselines/experiments/robothor pointnav_s2s_robothor_rgbd_resnet_ddppo \
    -et v1_new_objectnav_rgbd_clean_nR_128_nU_4_nTS_300M