# Evaluate checkpoints fine-tuned with self-supervised losses

# Rotation Prediction

# =========================
# Pixelate S3
# =========================

# Stage-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate_s3/2020-09-29_03-01-38/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate_s3__stage_00__steps_000020014391.pt \
    -t 2020-09-29_03-01-38 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_pixelate_pixelate_s3_st1 \
    -tsg 2 \
    -vc Pixelate \
    -vs 3

# Stage-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate_s3/2020-09-29_03-01-38/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate_s3__stage_00__steps_000020200691.pt \
    -t 2020-09-29_03-01-38 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_pixelate_pixelate_s3_st2 \
    -tsg 4 \
    -vc Pixelate \
    -vs 3

# Stage-3
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate_s3/2020-09-29_03-01-38/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate_s3__stage_00__steps_000020500841.pt \
    -t 2020-09-29_03-01-38 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_pixelate_pixelate_s3_st3 \
    -tsg 5 \
    -vc Pixelate \
    -vs 3

# =========================
# Gaussian Noise S3
# =========================

# Stage-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_gaussian_noise_s3/2020-09-29_01-35-21/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_gaussian_noise_s3__stage_00__steps_000020014391.pt \
    -t 2020-09-29_01-35-21 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_gaussian_noise_s3_st1 \
    -tsg 0 \
    -vc Gaussian_Noise \
    -vs 3

# Stage-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_gaussian_noise_s3/2020-09-29_01-35-21/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_gaussian_noise_s3__stage_00__steps_000020200691.pt \
    -t 2020-09-29_01-35-21 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_gaussian_noise_s3_st2 \
    -tsg 1 \
    -vc Gaussian_Noise \
    -vs 3

# Stage-3
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_gaussian_noise_s3/2020-09-29_01-35-21/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_gaussian_noise_s3__stage_00__steps_000020500841.pt \
    -t 2020-09-29_01-35-21 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_gaussian_noise_s3_st3 \
    -tsg 3 \
    -vc Gaussian_Noise \
    -vs 3

# =========================
# Pixelate S5
# =========================

# Stage-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate/2020-09-28_21-24-51/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate__stage_00__steps_000020014391.pt \
    -t 2020-09-28_21-24-51 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v2_pixelate_pixelate_s5_st1 \
    -tsg 0 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0

# Stage-1 (New) ~2M
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_500_rp_0.5_v2_pixelate/2020-09-30_22-48-57/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_500_rp_0.5_v2_pixelate__stage_00__steps_000021999041.pt \
    -t 2020-09-30_22-48-57 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_500_rp_0.5_v2_pixelate_s5_st1 \
    -tsg 0 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0


# Stage-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate/2020-09-28_21-24-51/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate__stage_00__steps_000020200691.pt \
    -t 2020-09-28_21-24-51 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v2_pixelate_pixelate_s5_st2 \
    -tsg 2 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0

# Stage-2 (New) ~4M
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_500_rp_0.5_v2_pixelate/2020-09-30_22-48-57/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_500_rp_0.5_v2_pixelate__stage_00__steps_000024009041.pt \
    -t 2020-09-30_22-48-57 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_500_rp_0.5_v2_pixelate_s5_st2 \
    -tsg 1 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0

# Stage-3
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate/2020-09-28_21-24-51/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate__stage_00__steps_000020500841.pt \
    -t 2020-09-28_21-24-51 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v2_pixelate_pixelate_s5_st3 \
    -tsg 3 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0

# Stage-3 (New) ~4M
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_500_rp_0.5_v2_pixelate/2020-09-30_22-48-57/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_500_rp_0.5_v2_pixelate__stage_00__steps_000026004041.pt \
    -t 2020-09-30_22-48-57 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_500_rp_0.5_v2_pixelate_s5_st3 \
    -tsg 2 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0

# Stage-4
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_pixelate/2020-09-28_21-24-51/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_pixelate__stage_00__steps_000020604341.pt \
    -t 2020-09-28_21-24-51 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v2_pixelate_pixelate_s5_st4 \
    -tsg 3 \
    -vc Pixelate \
    -vs 5 \
    -rp 1.0

# Another-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_500_v1_pixelate/2020-09-30_04-51-29/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_500_v1_pixelate__stage_00__steps_000020724041.pt \
    -t 2020-09-30_04-51-29 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_500_v1_pixelate_pixelate_s5_st4 \
    -tsg 5 \
    -vc Pixelate \
    -vs 5

# Another-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_500_v1_pixelate/2020-09-30_04-51-29/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_500_v1_pixelate__stage_00__steps_000020889041.pt \
    -t 2020-09-30_04-51-29 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_500_v1_pixelate_pixelate_s5_st5 \
    -tsg 5 \
    -vc Pixelate \
    -vs 5

# =========================
# Gaussian Noise S5
# =========================

# Stage-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_gaussian_noise/2020-09-29_00-06-58/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_gaussian_noise__stage_00__steps_000020014391.pt \
    -t 2020-09-29_01-35-21 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_gaussian_noise_s5_st1 \
    -tsg 0 \
    -vc Gaussian_Noise \
    -vs 5

# Stage-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_gaussian_noise/2020-09-29_00-06-58/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_gaussian_noise__stage_00__steps_000020200691.pt \
    -t 2020-09-29_01-35-21 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_gaussian_noise_s5_st2 \
    -tsg 1 \
    -vc Gaussian_Noise \
    -vs 5

# Stage-3
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPredictionAdapt/adapt_nR_30_v1_gaussian_noise/2020-09-29_00-06-58/exp_PointNavRobothorRGBPPORotationPredictionAdapt_adapt_nR_30_v1_gaussian_noise__stage_00__steps_000020500841.pt \
    -t 2020-09-29_01-35-21 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_nR_30_v1_gaussian_noise_s5_st3 \
    -tsg 3 \
    -vc Gaussian_Noise \
    -vs 5


# Action Prediction (w bel.)

# =========================
# Pixelate S3
# =========================

# Stage-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPredictionAdapt/adapt_nR_30_bel_mode_v1_pixelate_s3/2020-09-29_03-52-16/exp_PointNavRobothorRGBPPOActionPredictionAdapt_adapt_nR_30_bel_mode_v1_pixelate_s3__stage_00__steps_000015013273.pt \
    -t 2020-09-29_03-52-16 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_30_v1_pixelate_pixelate_s3_st1 \
    -tsg 2 \
    -vc Pixelate \
    -vs 3

# Stage-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPredictionAdapt/adapt_nR_30_bel_mode_v1_pixelate_s3/2020-09-29_03-52-16/exp_PointNavRobothorRGBPPOActionPredictionAdapt_adapt_nR_30_bel_mode_v1_pixelate_s3__stage_00__steps_000015199573.pt \
    -t 2020-09-29_03-52-16 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_30_v1_pixelate_pixelate_s3_st2 \
    -tsg 4 \
    -vc Pixelate \
    -vs 3

# Stage-3
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPredictionAdapt/adapt_nR_30_bel_mode_v1_pixelate_s3/2020-09-29_03-52-16/exp_PointNavRobothorRGBPPOActionPredictionAdapt_adapt_nR_30_bel_mode_v1_pixelate_s3__stage_00__steps_000015499723.pt \
    -t 2020-09-29_03-52-16 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_30_v1_pixelate_pixelate_s3_st3 \
    -tsg 5 \
    -vc Pixelate \
    -vs 3

# =========================
# Gaussian Noise S3
# =========================

# Stage-1

# Stage-2

# Stage-3

# =========================
# Pixelate S5
# =========================

# Stage-1
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPredictionAdapt/adapt_nR_30_bel_mode_v1_pixelate/2020-09-28_23-02-59/exp_PointNavRobothorRGBPPOActionPredictionAdapt_adapt_nR_30_bel_mode_v1_pixelate__stage_00__steps_000015013273.pt \
    -t 2020-09-28_23-02-59 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_30_v1_pixelate_pixelate_s5_st1 \
    -tsg 2 \
    -vc Pixelate \
    -vs 5

# Stage-2
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPredictionAdapt/adapt_nR_30_bel_mode_v1_pixelate/2020-09-28_23-02-59/exp_PointNavRobothorRGBPPOActionPredictionAdapt_adapt_nR_30_bel_mode_v1_pixelate__stage_00__steps_000015199573.pt \
    -t 2020-09-28_23-02-59 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_30_v1_pixelate_pixelate_s5_st2 \
    -tsg 4 \
    -vc Pixelate \
    -vs 5

# Stage-3
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPredictionAdapt/adapt_nR_30_bel_mode_v1_pixelate/2020-09-28_23-02-59/exp_PointNavRobothorRGBPPOActionPredictionAdapt_adapt_nR_30_bel_mode_v1_pixelate__stage_00__steps_000015499723.pt \
    -t 2020-09-28_23-02-59 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et adapt_bel_mode_nR_30_v1_pixelate_pixelate_s5_st3 \
    -tsg 5 \
    -vc Pixelate \
    -vs 5

# =========================
# Gaussian Noise S5
# =========================

# Stage-1

# Stage-2

# Stage-3
