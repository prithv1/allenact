# To run self-supervised adaptation experiments for pointnav
# on the p2.8x instance on AWS EC2

# Separate Rotation Prediction

# Pixelate S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_ec2_p2_8x_large_pixelate_s5 \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPrediction/v1_sep_rot_pred_rnet1_ve1_rc1_coeff0.01_ec2_p2_8x_large/2020-10-07_05-44-29/exp_PointNavRobothorResNetRGBPPOSeparateRotationPrediction_v1_sep_rot_pred_rnet1_ve1_rc1_coeff0.01_ec2_p2_8x_large__stage_00__steps_000020002306.pt \
    -vc Pixelate \
    -vs 5


# Gaussian Noise S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_ec2_p2_8x_large_gaussian_noise_s5 \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPrediction/v1_sep_rot_pred_rnet1_ve1_rc1_coeff0.01_ec2_p2_8x_large/2020-10-07_05-44-29/exp_PointNavRobothorResNetRGBPPOSeparateRotationPrediction_v1_sep_rot_pred_rnet1_ve1_rc1_coeff0.01_ec2_p2_8x_large__stage_00__steps_000020002306.pt \
    -vc Gaussian_Noise \
    -vs 5
