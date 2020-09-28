sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -b projects/tutorials objectnav_robothor_rgb_ddppo

sudo python main.py \
    -o storage/objectnav-robothor-rgb \
    -b projects/objectnav_baselines/experiments/robothor/ objectnav_robothor_rgb_resnet_ddppo

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -c storage/robothor-objectnav-rgb-resnet/checkpoints/Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO/2020-09-06_23-14-57/exp_Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO__stage_00__steps_000020003876.pt \
    -t 2020-09-06_23-14-57 \
    -b projects/tutorials objectnav_robothor_rgb_ddppo

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -c /home/prithvijitc/projects/repo_fix/embodied-rl/storage/Objectnav-RoboTHOR-rgb-ddppo-challenge-config-high-res/checkpoints/2020-05-21_10-02-30/exp_ObjectNavRobothorRGBPPO__stage_00__steps_000195032905.pt \
    -t 2020-08-07_03-40-27 \
    -b projects/tutorials objectnav_robothor_corrupt_rgb_ddppo \
    -np 15 \
    -tsg 0,1,2

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -c /home/prithvijitc/projects/repo_fix/embodied-rl/storage/Objectnav-RoboTHOR-rgb-ddppo-challenge-config-high-res/checkpoints/2020-05-21_10-02-30/exp_ObjectNavRobothorRGBPPO__stage_00__steps_000195032905.pt \
    -t 2020-08-07_03-40-27 \
    -b projects/tutorials objectnav_robothor_corrupt_rgb_ddppo \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -c storage/robothor-objectnav-rgb-resnet/checkpoints/Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO/2020-09-06_23-14-57/exp_Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO__stage_00__steps_000020003876.pt \
    -t 2020-09-06_23-14-57 \
    -b projects/tutorials objectnav_robothor_corrupt_rgb_ddppo \
    --tr_gpus [1,2,3,4,5,6] \
    --vl_gpus [0] \
    --ts_gpus [0] \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -b projects/tutorials objectnav_robothor_corrupt_rgb_ddppo \
    -trd datasets/robothor-objectnav/val

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnet \
    -b projects/tutorials objectnav_robothor_general_rgb_ddppo_action_prediction

# Pointnav

# Vanilla
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -et v2_vanilla \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug


sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -et v2 \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v2/2020-09-19_15-55-22/exp_PointNavRobothorRGBPPO_v2__stage_00__steps_000010002934.pt \
    -r


sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-08-31_10-20-29/exp_PointNavRobothorRGBPPO__stage_00__steps_000000756000.pt \
    -t 2020-09-06_23-14-57 \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -vc Pixelate \
    -vs 5

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-10_00-27-06/exp_PointNavRobothorRGBPPO__stage_00__steps_000005001722.pt \
    -r

# Evaluate Pointnav checkpoints
# PointNav RoboTHOR
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-10_03-47-19/exp_PointNavRobothorRGBPPO__stage_00__steps_000015003313.pt \
    -t 2020-09-10_03-47-19 \
    -tsg 0,1,2 \
    -np 15

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-10_03-47-19/exp_PointNavRobothorRGBPPO__stage_00__steps_000015003313.pt \
    -t 2020-09-10_03-47-19 \
    -tsg 3,4,5 \
    -np 15 \
    -vc Pixelate \
    -vs 5

# PointNav RoboTHOR (Action-prediction)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-10_14-45-53/exp_PointNavRobothorRGBPPO__stage_00__steps_000010000080.pt \
    -t 2020-09-10_14-45-53

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-12_18-37-27/exp_PointNavRobothorRGBPPO__stage_00__steps_000015002834.pt \
    -t 2020-09-12_18-37-27 \
    -tsd datasets/robothor-pointnav/minival_90_per_sc \
    -np 15 \
    -tsg 0,1,2

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-12_18-37-27/exp_PointNavRobothorRGBPPO__stage_00__steps_000015002834.pt \
    -t 2020-09-12_18-37-27 \
    -tsd datasets/robothor-pointnav/microval \
    -np 15 \
    -tsg 1 \
    -vc Pixelate \
    -vs 5

# Belief Included
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -et v2_bel_mode \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/v2_bel_mode/2020-09-23_15-04-41/exp_PointNavRobothorRGBPPOActionPrediction_v2_bel_mode__stage_00__steps_000010002098.pt \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -et v2_vis_mode \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

# Belief Excluded
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -et vis_mode \
    -r \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/vis_mode/2020-09-17_20-04-00/exp_PointNavRobothorRGBPPOActionPrediction_vis_mode__stage_00__steps_000005001653.pt \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_state_prediction_rgb_ddppo \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_td_prediction_rgb_ddppo \
    -et v2 \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_13-16-33/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000010001368.pt

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-25_08-49-42/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000010002689.pt \
    -r \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_random_prediction_rgb_ddppo \
    -et v2_0.002 \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_rotation_prediction_rgb_ddppo \
    -et v3_0.01 \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPORotationPrediction/v3_0.01/2020-09-26_02-59-33/exp_PointNavRobothorRGBPPORotationPrediction_v3_0.01__stage_00__steps_000015003422.pt \
    -r \
    -trd datasets/robothor-pointnav/debug \
    -vld datasets/robothor-pointnav/debug

# PointNav Evaluation

# Vanilla
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-10_03-47-19/exp_PointNavRobothorRGBPPO__stage_00__steps_000015003313.pt \
    -t 2020-09-10_03-47-19 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 4

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_general_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-09-10_03-47-19/exp_PointNavRobothorRGBPPO__stage_00__steps_000015003313.pt \
    -t 2020-09-10_03-47-19 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 6 \
    -vc Pixelate \
    -vs 5 \
    -et pixelate_s5

# Action Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -t 2020-09-14_01-24-44 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 0

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -t 2020-09-14_01-24-44 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 0 \
    -vc Pixelate \
    -vs 5 \
    -et pixelate_s5


# TD Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -t 2020-09-14_20-21-04 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 0

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_td_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOTDPrediction/2020-09-14_20-21-04/exp_PointNavRobothorRGBPPOTDPrediction__stage_00__steps_000005001206.pt \
    -t 2020-09-14_20-21-04 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 1 \
    -vc Pixelate \
    -vs 5 \
    -et pixelate_s5

# Adapt Action Prediction
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -vld datasets/robothor-pointnav/minival_60_per_sc \
    -vc Gaussian_Noise \
    -vs 5 \
    -et gaussian_noise_s5_nR_500

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOActionPrediction/2020-09-14_01-24-44/exp_PointNavRobothorRGBPPOActionPrediction__stage_00__steps_000015003738.pt \
    -r \
    -trd datasets/robothor-pointnav/minival_60_per_sc \
    -vld datasets/robothor-pointnav/minival_60_per_sc \
    -vc Pixelate \
    -vs 5

# (Evaluate Adapted Action Prediction)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials pointnav_robothor_adapt_action_prediction_rgb_ddppo \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPOAdaptActionPrediction/2020-09-16_18-49-35/exp_PointNavRobothorRGBPPOAdaptActionPrediction__stage_00__steps_000000005400.pt \
    -t 2020-09-16_18-49-35 \
    -tsd datasets/robothor-pointnav/minival_60_per_sc \
    -tsg 4 \
    -vc Gaussian_Noise \
    -vs 5 \
    -et gaussian_noise_s5



