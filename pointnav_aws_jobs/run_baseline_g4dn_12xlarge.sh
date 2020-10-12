# To run baseline pointnav pre-training jobs
# on the g4dn.12xlarge instance on AWS EC2

# Vanilla Model
python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_aws_configs pointnav_robothor_general_rgb_ddppo_g4dn_12xlarge \
    -et v1_vanilla_ec2_g4dn_12xlarge
