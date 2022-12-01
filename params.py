################  i3d ######################
frames_dir = r"/disks/disk0/huangxvfeng/dataset/AQA/frames"
frames_npy_dir = r"/disks/disk0/huangxvfeng/dataset/AQA/output_npy"
i3d_feature_dim = 2048

############### ms-tcn ##############################
tcn_save_dir = "./ckpts"
mstcn_pretrained_path = "./ckpts/mstcn01.pth"
tcn_num_stages = 4
tcn_num_layers = 10
tcn_num_f_maps = 64
tcn_num_classes = 5
tcn_features_dim = i3d_feature_dim
tcn_bz = 1
tcn_sample_rate = 1

############## resource ##############################
difficulty_dir = "./resources/difficulty_level.npy"
overall_score_dir = "./resources/overall_scores.npy"
tcn_time_dir = "./resources/tcn_time_point.npy"
train_index_dir = "./resources/training_idx.npy"
test_index_dir = "./resources/testing_idx.npy"

############  tcn format change #####################
tcn_time_points_filling_save_dir = "./tools/tcn_time"

###########  full connect layer ####################
best_base_full_connect_layer_dir = "./ckpts/base_full_connect1.pth"















