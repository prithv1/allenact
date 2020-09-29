# Run Self-supervised Adaptation under corruptions

# =======================================
# Action Prediction
# =======================================

# Gaussian Noise
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -et adapt_v1_gaussian_noise \
    -vc Gaussian_Noise \
    -vs 5

# Pixelate
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_action_prediction_rgb_ddppo \
    -et adapt_v1_pixelate \
    -vc Pixelate \
    -vs 5

# =======================================
# Rotation Prediction
# =======================================

# Gaussian Noise
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -et adapt_v1_gaussian_noise \
    -vc Gaussian_Noise \
    -vs 5

# Pixelate
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnet \
    -b projects/tutorials/pointnav_adaptation pointnav_robothor_ss_adapt_rotation_prediction_rgb_ddppo \
    -et adapt_v1_pixelate \
    -vc Pixelate \
    -vs 5