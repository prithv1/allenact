# Finetune a pre-trained vanilla navigation agent 
# for a few million steps under the corruptions

# Clean
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_clean_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt


# Pixelate S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_pixelate_s5_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Pixelate \
    -vs 5

# Pixelate S3
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_pixelate_s3_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Pixelate \
    -vs 3

# Gaussian Noise S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_gaussian_noise_s5_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Gaussian_Noise \
    -vs 5

# Gaussian Noise S3
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_gaussian_noise_s3_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Gaussian_Noise \
    -vs 3

# Lighting S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_lighting_s5_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Lighting \
    -vs 5

# Lighting S3
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_lighting_s3_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Lighting \
    -vs 3

# Spatter S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_spatter_s5_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Spatter \
    -vs 5

# Spatter S3
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_spatter_s3_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Spatter \
    -vs 3

# Contrast S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_contrast_s5_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Contrast \
    -vs 5

# Contrast S3
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_contrast_s3_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Contrast \
    -vs 3

# Motion Blur S5
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_motion_blur_s5_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Motion_Blur \
    -vs 5

# Motion Blur S3
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge_corr_finetune \
    -et v1_motion_blur_s3_corr_finetune \
    -c /home/ubuntu/projects/allenact/storage/robothor-pointnav-rgb-resnet/checkpoints/PointNavRobothorRGBPPO/v1_vanilla_ec2_p3_8x_large/2020-10-03_08-04-27/exp_PointNavRobothorRGBPPO_v1_vanilla_ec2_p3_8x_large__stage_00__steps_000075006000.pt \
    -vc Motion_Blur \
    -vs 3