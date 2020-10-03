# To run baseline pointnav pre-training jobs
# on the p3.8x instance on AWS EC2

python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_p3_8x_large \
    -et v1_vanilla_ec2_p3_8x_large