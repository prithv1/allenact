# PointNav Baselines

# PointNav RGB Models
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgb_resnet_ddppo \
    -et v1_new_pointnav_rgb_clean \
    -c pretrained_model_ckpts/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-08-31_12-13-30/exp_PointNavRobothorRGBPPO__stage_00__steps_000039031200.pt

# PointNav RGB baseline (v2)
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgb_resnet_ddppo \
    -et v2_new_pointnav_rgb_clean