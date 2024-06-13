import configargparse


def config_parser():
    
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/stereo_blur_dataset/sailor.txt',
                        help='config file path')
    parser.add_argument("--datadir", type=str, default='./data/stereo_blur_dataset/sailor/dense',
                        help='input data directory')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    
    # train options
    parser.add_argument("--N_iters", type=int, default=300000,
                        help='the number of sharp images one blur image corresponds to')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--deblur_images", type=int, default=5,
                        help='the number of sharp images one blur image corresponds to')
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)
    parser.add_argument("--target_idx", type=int, default=15, 
                        help='target_idx')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 288')
    parser.add_argument("--linear", action='store_true', default=False,
                        help='linear or cubic spline')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--pose_lrate", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')
    parser.add_argument("--netchunk", type=int, default=1024*16, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--chunk", type=int, default=1024*16, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, otherwise 3 frame consistency')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')
    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')
    parser.add_argument("--w_gradient", type=float, default=0.01, 
                        help='w_gradient regularization weight')
    parser.add_argument("--w_pose", type=float, default=0.02, 
                        help='pose_loss regularization weight')
    parser.add_argument("--w_empty", type=float, default=0.1, 
                        help='pose_loss regularization weight')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument('--num_basis', type=int, default=6,
                        help='The number of basis for motion trajectory')
    
    # rendering options
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    
    # logging/saving options
    parser.add_argument("--i_tqdm", type=int, default=100, 
                        help='frequency of print tqdm')
    parser.add_argument("--i_print", type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_img", type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    
    # llff flags
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    
    
    return parser