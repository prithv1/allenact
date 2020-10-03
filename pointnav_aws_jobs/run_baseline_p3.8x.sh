# To run baseline pointnav pre-training jobs
# on the p3.8x instance on AWS EC2

sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_p3.8x_large.py \
    -et v1_vanilla_ec2