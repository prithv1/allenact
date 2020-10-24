sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnetgru-dppo \
    -b projects/objectnav_baselines/experiments/robothor objectnav_s2s_robothor_rgb_resnetgru_ddppo

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnetgru-dppo \
    -b projects/objectnav_baselines/experiments/robothor objectnav_s2s_robothor_rgb_resnetgru_ddppo \
    -c /home/prithvijitc/projects/allenact_fix/fork_ala/allenact/storage/objectnav-robothor/checkpoints/Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO/2020-10-09_16-07-22/exp_Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO__stage_00__steps_000295342033.pt \
    -t 2020-10-09_16-07-22 \
    -tsg 0 \
    -e \
    -d

sudo python main.py \
    -o storage/robothor-objectnav-rgb-resnetgru-dppo \
    -b projects/objectnav_baselines/experiments/robothor objectnav_s2s_robothor_rgb_resnetgru_ddppo \
    -c /home/prithvijitc/projects/allenact_fix/fork_ala/allenact/storage/robothor-objectnav/checkpoints/Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO/2020-10-04_02-56-58/exp_Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO__stage_00__steps_000300004277.pt \
    -t 2020-10-09_16-07-22 \
    -tsg 0 \
    -e

sudo python main.py \
    -o storage/robothor-pointnav-rgbd-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgbd_resnet_ddppo

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgb_resnet_ddppo \
    -c /home/prithvijitc/projects/allenact_fix/fork_ala/allenact/pretrained_model_ckpts/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-08-31_12-13-30/exp_PointNavRobothorRGBPPO__stage_00__steps_000039031200.pt \
    -t 2020-09-24_01-50-05 \
    -tsg 0 \
    -tsd datasets/robothor-pointnav/eval_val_0.4_per_sc \
    -e \
    -vc Spatter \
    -vs 3

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-dppo \
    -b projects/pointnav_baselines/experiments/robothor pointnav_s2s_robothor_rgb_resnet_ddppo \
    -c /home/prithvijitc/projects/allenact_fix/fork_ala/allenact/pretrained_model_ckpts/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/2020-08-31_12-13-30/exp_PointNavRobothorRGBPPO__stage_00__steps_000039031200.pt