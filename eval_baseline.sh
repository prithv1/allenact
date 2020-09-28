# PointNav Evaluation

#***********************************
# Clean Settings
#***********************************

# Vanilla
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-20_14-15-08/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000005001572.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-20_14-15-08/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000005001572.pt \
    -t 2020-09-20_14-15-08 \
    -tsg 0 \
    -tsd datasets/robothor-pointnav/minival_6_per_sc \
    -e \
    -r \
    -e \
    -et clean

# Action Prediction (with Belief State)
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -e \
    -et clean_bel_mode

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -e \
    -et clean_bel_mode

# Action Prediction (without Belief State)
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt \
    -r \
    -e \
    -et clean_vis_mode_v2

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt \
    -t 2020-09-18_18-10-54 \
    -tsg 0 \
    -e \
    -et clean_vis_mode \
    -r \
    -e \
    -et clean_vis_mode

# Temporal Distance Prediction
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -r \
    -e \
    -et clean

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -r \
    -e \
    -et clean

# Rotation Prediction
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt \
    -r \
    -e \
    -et clean

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt \
    -t 2020-09-21_18-14-37 \
    -tsg 0 \
    -e \
    -et clean \
    -r \
    -e \
    -et clean

#***********************************
# Corrupt Settings (Gaussian s5)
#***********************************

# Vanilla
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-20_14-15-08/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000005001572.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-20_14-15-08/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000005001572.pt \
    -t 2020-09-20_14-15-08 \
    -tsg 0 \
    -e \
    -et gaussian_noise_s5 \
    -vc Gaussian_Noise \
    -vs 5 \
    -r

# Action Prediction (with Belief State)
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -e \
    -et gaussian_noise_s5_vis_mode \
    -vc Gaussian_Noise \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -e \
    -et gaussian_noise_s5_vis_mode \
    -vc Gaussian_Noise \
    -vs 5

# Action Prediction (without Belief State)
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt \
    -r \
    -e \
    -et gaussian_noise_s5_bel_mode \
    -vc Gaussian_Noise \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt \
    -t 2020-09-14_01-24-44 \
    -tsg 0 \
    -e \
    -et gaussian_noise_s5_vis_mode \
    -vc Gaussian_Noise \
    -vs 5 \
    -r \
    -e \
    -et gaussian_noise_s5_bel_mode \
    -vc Gaussian_Noise \
    -vs 5

# Temporal Distance Prediction
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -r \
    -e \
    -et gaussian_noise_s5 \
    -vc Gaussian_Noise \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -r \
    -e \
    -et gaussian_noise_s5 \
    -vc Gaussian_Noise \
    -vs 5

# Rotation Prediction
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt \
    -r \
    -e \
    -et gaussian_noise_s5 \
    -vc Gaussian_Noise \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt \
    -t 2020-09-21_18-14-37 \
    -tsg 1 \
    -e \
    -et gaussian_noise_s5 \
    -vc Gaussian_Noise \
    -vs 5 \
    -r \
    -e \
    -et gaussian_noise_s5 \
    -vc Gaussian_Noise \
    -vs 5

#***********************************
# Corrupt Settings (Pixelate s5)
#***********************************

# Vanilla
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-20_14-15-08/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000005001572.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-20_14-15-08/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000005001572.pt \
    -t 2020-09-20_14-15-08 \
    -tsg 1 \
    -e \
    -et pixelate_s5 \
    -vc Pixelate \
    -vs 5 \
    -r

# Action Prediction (with Belief State)
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -e \
    -et pixelate_s5_vis_mode \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -e \
    -et pixelate_s5_vis_mode \
    -vc Pixelate \
    -vs 5

# Action Prediction (without Belief State)
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt \
    -r \
    -e \
    -et pixelate_s5_bel_mode \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-18_18-10-54/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000010001913.pt \
    -t 2020-09-14_01-24-44 \
    -tsg 2 \
    -e \
    -et pixelate_s5_vis_mode \
    -vc Pixelate \
    -vs 5 \
    -r \
    -e \
    -et pixelate_s5_bel_mode \
    -vc Pixelate \
    -vs 5

# Temporal Distance Prediction
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -r \
    -e \
    -et pixelate_s5 \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -r \
    -e \
    -et pixelate_s5 \
    -vc Pixelate \
    -vs 5

# Rotation Prediction
# Checkpoint -- storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt \
    -r \
    -e \
    -et pixelate_s5 \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v1_0.01/2020-09-21_18-14-37/exp_PointNavRobothorRGBPPORotationPrediction_v1_0.01__stage_00__steps_000005000722.pt \
    -t 2020-09-21_18-14-37 \
    -tsg 3 \
    -e \
    -et pixelate_s5 \
    -vc Pixelate \
    -vs 5 \
    -r \
    -e \
    -et pixelate_s5 \
    -vc Pixelate \
    -vs 5

# -------------------------------------------
# Latest Run 2 degradation evidence
# -------------------------------------------


# *****************************
# Clean
# *****************************

# Vanilla
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2_vanilla/2020-09-24_01-50-05/exp_PointNavRobothorRGBPPO_v2_vanilla__stage_00__steps_000015001569.pt \
    -t 2020-09-24_01-50-05 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_clean_vanilla \
    -tsg 0

# Action Prediction (w/o bel)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -t 2020-09-23_02-06-35 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_clean_vis_mode \
    -tsg 0

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_clean_vis_mode_loss_logs
# Loss Logs=====================================

# Action Prediction (w bel)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -t 2020-09-25_00-09-53 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_clean_bel_mode \
    -tsg 0

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_clean_bel_mode_loss_logs
# Loss Logs=====================================

# TD Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_13-15-09/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000015003114.pt \
    -t 2020-09-25_13-15-09 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_clean_mode \
    -tsg 3

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_13-15-09/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000015003114.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_clean_mode_loss_logs
# Loss Logs=====================================

# Rotation Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -t 2020-09-26_06-48-06 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v3_clean_mode \
    -tsg 3

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_clean_mode_loss_logs
# Loss Logs=====================================

# *****************************
# Gaussian Noise
# *****************************

# Vanilla
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2_vanilla/2020-09-24_01-50-05/exp_PointNavRobothorRGBPPO_v2_vanilla__stage_00__steps_000015001569.pt \
    -t 2020-09-24_01-50-05 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_gaussian_noise_vanilla \
    -tsg 1 \
    -vc Gaussian_Noise \
    -vs 5

# Action Prediction (w/o bel)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -t 2020-09-23_02-06-35 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_gaussian_noise_vis_mode \
    -tsg 1 \
    -vc Gaussian_Noise \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_gaussian_noise_vis_mode_loss_logs \
    -vc Gaussian_Noise \
    -vs 5
# Loss Logs=====================================


# Action Prediction (w bel)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -t 2020-09-25_00-09-53 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_gaussian_noise_bel_mode \
    -tsg 5 \
    -vc Gaussian_Noise \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_gaussian_noise_bel_mode_loss_logs \
    -vc Gaussian_Noise \
    -vs 5
# Loss Logs=====================================

# TD Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_13-15-09/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000015003114.pt \
    -t 2020-09-25_13-15-09 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_gaussian_noise_mode \
    -tsg 0 \
    -vc Gaussian_Noise \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_13-15-09/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000015003114.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_gaussian_noise_mode_loss_logs \
    -vc Gaussian_Noise \
    -vs 5
# Loss Logs=====================================

# Rotation Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -t 2020-09-26_06-48-06 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v3_gaussian_noise_mode \
    -tsg 5 \
    -vc Gaussian_Noise \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_gaussian_noise_mode_loss_logs \
    -vc Gaussian_Noise \
    -vs 5
# Loss Logs=====================================

# *****************************
# Pixelate
# *****************************

# Vanilla
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2_vanilla/2020-09-24_01-50-05/exp_PointNavRobothorRGBPPO_v2_vanilla__stage_00__steps_000015001569.pt \
    -t 2020-09-24_01-50-05 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_pixelate_vanilla \
    -tsg 2 \
    -vc Pixelate \
    -vs 5

# Action Prediction (w/o bel)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -t 2020-09-23_02-06-35 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_pixelate_vis_mode \
    -tsg 2 \
    -vc Pixelate \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_vis_mode/2020-09-23_02-06-35/exp_PointNavRobothorRGBPPOActionPrediction_v2_vis_mode__stage_00__steps_000015001511.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_pixelate_vis_mode_loss_logs \
    -vc Pixelate \
    -vs 5
# Loss Logs=====================================

# Action Prediction (w bel)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -t 2020-09-25_00-09-53 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_pixelate_bel_mode \
    -tsg 2 \
    -vc Pixelate \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-25_00-09-53/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000015002923.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_pixelate_bel_mode_loss_logs \
    -vc Pixelate \
    -vs 5
# Loss Logs=====================================

# TD Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_13-15-09/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000015003114.pt \
    -t 2020-09-25_13-15-09 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v2_pixelate_mode \
    -tsg 4 \
    -vc Pixelate \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_13-15-09/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000015003114.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_pixelate_mode_loss_logs \
    -vc Pixelate \
    -vs 5
# Loss Logs=====================================

# Rotation Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -t 2020-09-26_06-48-06 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -e \
    -et v3_pixelate_mode \
    -tsg 2 \
    -vc Pixelate \
    -vs 5

# Loss Logs=====================================
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_evaluation pointnav_robothor_rotation_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_06-48-06/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000020004041.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -et v2_pixelate_mode_loss_logs \
    -vc Pixelate \
    -vs 5
# Loss Logs=====================================