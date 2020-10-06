# To run baseline pointnav pre-training jobs
# on the p2.8x instance on AWS EC2

# Vanilla Model
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_p2_8x_large \
    -et v1_vanilla_ec2_p2_8x_large

# Rotation Prediction (updating ResNet)
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_rot_pred_rnet1_ve1_rc1_rp0.7_coeff0.01_ec2_p2_8x_large
