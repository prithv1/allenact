# To run self-supervised adaptation experiments for pointnav
# on the p2.8x instance on AWS EC2

# Separate Rotation Prediction

# Adaptation--------------------------------------------------------------------------------------------

# Pixelate S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_pixelate_s5_rec_aux_perf \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt/v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_pixelate_s5/2020-10-16_13-52-28/exp_PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt_v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_pixelate_s5__stage_00__steps_000020083306.pt \
    -t 2020-10-16_13-52-28 \
    -vc Pixelate \
    -vs 5 \
    -tsg 0

python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_pixelate_s3_rec_aux_perf \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt/v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_pixelate_s3/2020-10-16_14-14-39/exp_PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt_v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_pixelate_s3__stage_00__steps_000020040106.pt \
    -t 2020-10-16_14-14-39 \
    -vc Pixelate \
    -vs 3 \
    -tsg 0

# Gaussian Noise S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_gaussian_noise_s5_rec_aux_perf \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt/v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_gaussian_noise_s5/2020-10-16_14-36-04/exp_PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt_v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_gaussian_noise_s5__stage_00__steps_000020158906.pt \
    -t 2020-10-16_14-36-04 \
    -vc Gaussian_Noise \
    -vs 5 \
    -tsg 1

python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_gaussian_noise_s3_rec_aux_perf \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt/v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_gaussian_noise_s3/2020-10-16_15-17-42/exp_PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt_v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_gaussian_noise_s3__stage_00__steps_000020088706.pt \
    -t 2020-10-16_15-17-42 \
    -vc Gaussian_Noise \
    -vs 3 \
    -tsg 1

# Low Lighting
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_lighting_s5_rec_aux_perf \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt/v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_lighting_s5/2020-10-16_15-50-59/exp_PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt_v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_lighting_s5__stage_00__steps_000020099506.pt \
    -t 2020-10-16_15-50-59 \
    -vc Lighting \
    -vs 5 \
    -tsg 2

python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_ss_adapt_sep_rotation_prediction_resnet_rgb_ddppo_p2_8x_large \
    -et v2_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_lighting_s3_rec_aux_perf \
    -c storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt/v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_lighting_s3/2020-10-16_17-02-11/exp_PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt_v1_sep_rot_pred_rnet0_ve1_rc1_coeff0.01_only_vis_ec2_g4dn12x_large_lighting_s3__stage_00__steps_000020029306.pt \
    -t 2020-10-16_17-02-11 \
    -vc Lighting \
    -vs 3 \
    -tsg 3