# Run Self-supervised Adaptation under corruptions

# =======================================
# Action Prediction (w/o bel.)
# =======================================

# Gaussian Noise
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -et adapt_nR_30_vis_mode_v1_gaussian_noise \
    -vc Gaussian_Noise \
    -vs 5

# Pixelate
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -et adapt_nR_30_vis_mode_v2_pixelate \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -et adapt_nR_500_vis_mode_v2_pixelate \
    -vc Pixelate \
    -vs 5

# =======================================
# Action Prediction (w bel.)
# =======================================

# Gaussian Noise
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -et adapt_nR_30_bel_mode_v1_gaussian_noise \
    -vc Gaussian_Noise \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -et adapt_nR_30_bel_mode_v1_gaussian_noise_s3 \
    -vc Gaussian_Noise \
    -vs 3

# Pixelate
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -et adapt_nR_30_bel_mode_v1_pixelate \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -et adapt_nR_500_bel_mode_v1_pixelate \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -et adapt_nR_30_bel_mode_v1_pixelate_s3 \
    -vc Pixelate \
    -vs 3

# =======================================
# Rotation Prediction
# =======================================

# Gaussian Noise (s5)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_30_v1_gaussian_noise \
    -vc Gaussian_Noise \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_500_v1_gaussian_noise \
    -vc Gaussian_Noise \
    -vs 5

# Gaussian Noise (s3)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_30_v1_gaussian_noise_s3 \
    -vc Gaussian_Noise \
    -vs 3

# Pixelate (s5)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_30_v1_pixelate \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_500_v1_pixelate \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_500_rp_0.5_v2_pixelate \
    -vc Pixelate \
    -vs 5 \
    -rp 0.5

# Pixelate (s3)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -et adapt_nR_30_v1_pixelate_s3 \
    -vc Pixelate \
    -vs 3