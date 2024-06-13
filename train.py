import torch
import numpy as np
import os
import sys
# import tqdm
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from kornia import create_meshgrid

from config import config_parser
from load_llff_data import load_llff_data
from nerf_model import create_nerf
from spline import *
from run_nerf_helpers import *
from render import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)


def train():
    parser = config_parser()
    args = parser.parse_args()

    print('spline numbers: ', args.deblur_images)
    
    K = None
    if args.dataset_type == 'llff':
        images, poses_start, bds, \
        sharp_images, depths, \
        masks, motion_coords, \
        render_poses, ref_c2w = load_llff_data(args.datadir, args.start_frame, args.end_frame, 
                                               target_idx=args.target_idx, recenter=True, 
                                               bd_factor=.9, spherify=args.spherify, 
                                               final_height=args.final_height)
        hwf = poses_start[0, :3,- 1]
        i_test = []
        i_val = [] 
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        
        poses_start = torch.Tensor(poses_start)
        poses_end = poses_start
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses_org = poses_start.repeat(args.deblur_images, 1, 1)
        poses = poses_org[:, :, :4]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()
    
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
    
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    render_kwargs_train, render_kwargs_test, start, optimizer, \
    optimizer_se3, optimizer_basis = create_nerf(args, poses_start_se3, poses_end_se3, images.shape[0])
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_rand = args.N_rand
    N_iters = args.N_iters + 1
    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    masks = 1.0 - torch.Tensor(masks)
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda()
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])

    decay_iteration = max(args.decay_iteration, args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)
    chain_bwd = 0

    for i in trange(start, N_iters):
        chain_bwd = 1 - chain_bwd
        print('Random FROM SINGLE IMAGE')
        # Random from one image
        img_i = np.random.choice(i_train)
        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()
        
        target = images[img_i].cuda()
        depth_gt = depths[img_i].cuda()
        mask_gt = masks[img_i].cuda()
        dy_coords = torch.Tensor(motion_coords[img_i]).cuda()

        if img_i == 0:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
                                                   args.start_frame, fwd=True)
            flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
        elif img_i == num_img - 1:
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
                                                   args.start_frame, fwd=False)
            flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
        else:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
                                                   img_i, args.start_frame, 
                                                   fwd=True)
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
                                                   img_i, args.start_frame, 
                                                   fwd=False)
        flow_fwd = torch.Tensor(flow_fwd).cuda()
        fwd_mask = torch.Tensor(fwd_mask).cuda()
    
        flow_bwd = torch.Tensor(flow_bwd).cuda()
        bwd_mask = torch.Tensor(bwd_mask).cuda()

        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        spline_poses = get_pose(args, img_i, render_kwargs_train['se3']) #[deblur_images,3,4]
        ray_idx = torch.randperm(H * W)[:args.N_rand // args.deblur_images]
        if args.use_motion_mask and i < decay_iteration * 1000:
            print('HARD MINING STAGE !')
            num_extra_sample = args.num_extra_sample // args.deblur_images
            dy_ray_idx = dy_coords[:, 0] * W + dy_coords[:, 1]
            select_inds_hard = np.random.choice(dy_ray_idx.shape[0], 
                                                size=[min(dy_ray_idx.shape[0], num_extra_sample)], 
                                                replace=False)
            hard_ray_idx = dy_ray_idx[select_inds_hard].int()
            ray_idx = torch.cat([ray_idx, hard_ray_idx], 0)

        target_rgb = target.reshape(-1, H * W, 3)
        target_rgb = target_rgb[:, ray_idx]
        target_rgb = target_rgb.reshape(-1, 3)

        target_depth = depth_gt.reshape(-1, H * W)
        target_depth = target_depth[:, ray_idx]
        target_depth = target_depth.reshape(-1)

        target_mask = mask_gt.reshape(-1, H * W)
        target_mask = target_mask[:, ray_idx]
        target_mask = target_mask.reshape(-1).unsqueeze(-1)

        target_pixel = uv_grid.reshape(-1, H * W, 2)
        target_pixel = target_pixel[:, ray_idx]
        target_pixel = target_pixel.reshape(-1, 2)

        target_of_fwd = flow_fwd.reshape(-1, H * W, 2)
        target_of_fwd = target_of_fwd[:, ray_idx]
        target_of_fwd = target_of_fwd.reshape(-1, 2)
        target_fwd_mask = fwd_mask.reshape(-1, H * W)
        target_fwd_mask = target_fwd_mask[:, ray_idx]
        target_fwd_mask = target_fwd_mask.reshape(-1).unsqueeze(-1)

        target_of_bwd = flow_bwd.reshape(-1, H * W, 2)
        target_of_bwd = target_of_bwd[:, ray_idx]
        target_of_bwd = target_of_bwd.reshape(-1, 2)
        target_bwd_mask = bwd_mask.reshape(-1, H * W)
        target_bwd_mask = target_bwd_mask[:, ray_idx]
        target_bwd_mask = target_bwd_mask.reshape(-1).unsqueeze(-1)

        if args.chain_sf and i > decay_iteration * 1000 * 2:
            chain_5frames = True
        else:
            chain_5frames = False

        ret = render(args, i, img_i, chain_bwd, chain_5frames, spline_poses, ray_idx, 
                     num_img, H, W, K, training=True, **render_kwargs_train)
        
        spline_poses_post = get_pose(args, min(img_i + 1, int(num_img) - 1), 
                                     render_kwargs_train['se3'])
        spline_poses_prev = get_pose(args, max(img_i - 1, 0), 
                                     render_kwargs_train['se3'])
        
        interval = target_rgb.shape[0]
        rgb_ = 0
        depth_ = 0
        rgb_map_dy_ = 0
        rgb_map_st_ = 0
        depth_map_dy_ = 0
        rgb_map_post_dy_ = 0
        rgb_map_prev_dy_ = 0
        rgb_map_pp_dy_ = 0
        weight_map_post_ = 0
        weight_map_prev_ = 0
        weights_map_dd_ = 0
        render_of_fwd_ = 0
        render_of_bwd_ = 0

        for j in range(0, args.deblur_images):
            rgb_ += ret['rgb_map'][j * interval : (j + 1) * interval]
            rgb_map_dy_ += ret['rgb_map_dy'][j * interval : (j + 1) * interval]
            rgb_map_st_ += ret['rgb_map_st'][j * interval : (j + 1) * interval]
            rgb_map_post_dy_ += ret['rgb_map_post_dy'][j * interval : (j + 1) * interval]
            rgb_map_prev_dy_ += ret['rgb_map_prev_dy'][j * interval : (j + 1) * interval]
            weight_map_post_ += ret['prob_map_post'][j * interval : (j + 1) * interval]
            weight_map_prev_ += ret['prob_map_prev'][j * interval : (j + 1) * interval]
            weights_map_dd_ += ret['weights_map_dd'][j * interval : (j + 1) * interval]
            if chain_5frames:
                rgb_map_pp_dy_ += ret['rgb_map_pp_dy'][j * interval : (j + 1) * interval] 
            pose = spline_poses[j]
            pose_post = spline_poses_post[j]
            pose_prev = spline_poses_prev[j]
            weights_dy = ret['weights_dy'][j * interval : (j + 1) * interval]
            raw_pts_post = ret['raw_pts_post'][j * interval : (j + 1) * interval]
            raw_pts_prev = ret['raw_pts_prev'][j * interval : (j + 1) * interval]
            fwd, bwd = compute_optical_flow(pose_post, pose, pose_prev, 
                                            H, W, focal, weights_dy, 
                                            raw_pts_post, raw_pts_prev)
            
            if j == 0:
                depth_ += ret['depth_map'][j * interval : (j + 1) * interval]
                depth_map_dy_ += ret['depth_map_dy'][j * interval : (j + 1) * interval]
                render_of_fwd_ += fwd
                render_of_bwd_ += bwd
            else:
                depth_ = torch.min(depth_, ret['depth_map'][j * interval : (j + 1) * interval])
                depth_map_dy_ = torch.min(depth_map_dy_, ret['depth_map_dy'][j * interval : (j + 1) * interval])
                render_of_fwd_ = compare_flow(render_of_fwd_, fwd, target_pixel)
                render_of_bwd_ = compare_flow(render_of_bwd_, bwd, target_pixel)

        rgb_ = rgb_ / args.deblur_images
        rgb_map_dy_ = rgb_map_dy_ / args.deblur_images
        rgb_map_st_ = rgb_map_st_ / args.deblur_images
        rgb_map_post_dy_ = rgb_map_post_dy_ / args.deblur_images
        rgb_map_prev_dy_ = rgb_map_prev_dy_ / args.deblur_images
        weight_map_post_ = weight_map_post_ / args.deblur_images
        weight_map_prev_ = weight_map_prev_ / args.deblur_images
        weights_map_dd_ = weights_map_dd_ / args.deblur_images
        rgb_blur = rgb_.reshape(-1, 3)
        depth_blur = depth_.reshape(-1)
        rgb_map_dy_blur = rgb_map_dy_.reshape(-1, 3)
        rgb_map_st_blur = rgb_map_st_.reshape(-1, 3)
        depth_map_dy_blur = depth_map_dy_.reshape(-1)
        rgb_map_post_dy_blur = rgb_map_post_dy_.reshape(-1, 3)
        rgb_map_prev_dy_blur = rgb_map_prev_dy_.reshape(-1, 3)
        weight_map_post_blur = weight_map_post_.reshape(-1)
        weight_map_prev_blur = weight_map_prev_.reshape(-1)
        weights_map_dd_blur = weights_map_dd_.reshape(-1)
        render_of_fwd_blur = render_of_fwd_.reshape(-1, 2)
        render_of_bwd_blur = render_of_bwd_.reshape(-1, 2)
        if chain_5frames:
            rgb_map_pp_dy_ = rgb_map_pp_dy_ / args.deblur_images
            rgb_map_pp_dy_blur = rgb_map_pp_dy_.reshape(-1, 3)

        optimizer_se3.zero_grad()
        optimizer.zero_grad()
        optimizer_basis.zero_grad()

        prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                         + torch.mean(torch.abs(ret['raw_prob_ref2post'])))
        
        if i <= decay_iteration * 1000:
            render_loss = img2mse(rgb_map_dy_blur, target_rgb)
            render_loss += compute_mse(rgb_map_post_dy_blur, target_rgb, 
                                       weight_map_post_blur.unsqueeze(-1))
            render_loss += compute_mse(rgb_map_prev_dy_blur, target_rgb, 
                                       weight_map_prev_blur.unsqueeze(-1))
        else:
            print('only compute dynamic render loss in masked region')
            weights_map_dd = weights_map_dd_blur.unsqueeze(-1).detach()

            # dynamic rendering loss
            render_loss = compute_mse(rgb_map_dy_blur, target_rgb, 
                                      weights_map_dd)
            render_loss += compute_mse(rgb_map_post_dy_blur, target_rgb, 
                                       weight_map_post_blur.unsqueeze(-1) * weights_map_dd)
            render_loss += compute_mse(rgb_map_prev_dy_blur, target_rgb, 
                                       weight_map_prev_blur.unsqueeze(-1) * weights_map_dd)
        
        render_loss += 0.1 * compute_mse(rgb_map_st_blur[:args.N_rand // args.deblur_images, ...], 
                                         target_rgb[:args.N_rand // args.deblur_images, ...], 
                                         target_mask[:args.N_rand // args.deblur_images, ...])
        img_loss = img2mse(rgb_blur[:args.N_rand // args.deblur_images, ...], 
                           target_rgb[:args.N_rand // args.deblur_images, ...])
        render_loss += img_loss
        psnr = mse2psnr(img_loss)

        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']
        sf_cycle_loss = args.w_cycle * compute_mae(ret['raw_sf_ref2post'], 
                                                   -ret['raw_sf_post2ref'], 
                                                   weight_post.unsqueeze(-1), dim=3)
        sf_cycle_loss += args.w_cycle * compute_mae(ret['raw_sf_ref2prev'], 
                                                    -ret['raw_sf_prev2ref'], 
                                                    weight_prev.unsqueeze(-1), dim=3)
        
        render_sf_ref2prev = torch.sum(ret['weights_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
        render_sf_ref2post = torch.sum(ret['weights_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)
        
        sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                     + torch.mean(torch.abs(render_sf_ref2post))) 
        
        divsor = i // (decay_iteration * 1000)
        decay_rate = 10
        if args.decay_depth_w:
            w_depth = args.w_depth / (decay_rate ** divsor)
        else:
            w_depth = args.w_depth
        if args.decay_optical_flow_w:
            w_of = args.w_optical_flow/(decay_rate ** divsor)
        else:
            w_of = args.w_optical_flow
        
        print('w_depth ', w_depth, 'w_of ', w_of)

        depth_loss = w_depth * compute_depth_loss(depth_map_dy_blur, -target_depth)
        
        gradient_loss = args.w_gradient * compute_gradient_loss(depth_blur, -target_depth)

        if img_i == 0:
            print('only fwd flow')
            flow_loss = w_of * compute_mae(render_of_fwd_blur, target_of_fwd, target_fwd_mask)
        elif img_i == num_img - 1:
            print('only bwd flow')
            flow_loss = w_of * compute_mae(render_of_bwd_blur, target_of_bwd, target_bwd_mask)
        else:
            flow_loss = w_of * compute_mae(render_of_fwd_blur, target_of_fwd, target_fwd_mask)
            flow_loss += w_of * compute_mae(render_of_bwd_blur, target_of_bwd, target_bwd_mask)

        
        sf_sm_loss = args.w_sm * (compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                     ret['raw_pts_post'], 
                                                     H, W, focal) \
                                + compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                     ret['raw_pts_prev'], 
                                                     H, W, focal))
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                      ret['raw_pts_post'], 
                                                      ret['raw_pts_prev'], 
                                                      H, W, focal)
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                      ret['raw_pts_post'], 
                                                      ret['raw_pts_prev'], 
                                                      H, W, focal)
        if chain_bwd:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_prev'], 
                                                          ret['raw_pts_ref'], 
                                                          ret['raw_pts_pp'], 
                                                          H, W, focal)
        else:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_post'], 
                                                          ret['raw_pts_pp'], 
                                                          ret['raw_pts_ref'], 
                                                          H, W, focal)
        
        entropy_loss = args.w_entropy * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8))

        if chain_5frames:
            print('5 FRAME RENDER LOSS ADDED') 
            render_loss += compute_mse(rgb_map_pp_dy_blur, target_rgb, 
                                       weights_map_dd)
        
        w_empty = args.w_empty / (decay_rate ** divsor)
        # for j in range(0, args.deblur_images):
        #     empty_loss += w_empty * compute_empty_loss(ret['z_vals'][j * interval : (j + 1) * interval], 
        #                                                ret['sigma_dy'][j * interval : (j + 1) * interval], 
        #                                                -target_depth, 1 - target_mask, near, far)
        #     empty_loss += w_empty * compute_empty_loss(ret['z_vals'][j * interval : (j + 1) * interval], 
        #                                                ret['sigma_st'][j * interval : (j + 1) * interval], 
        #                                                -target_depth, target_mask, near, far)
        empty_loss = w_empty * compute_empty_loss(ret['z_vals'], ret['sigma_dy'], 
                                                   -target_depth.repeat(args.deblur_images), 
                                                   1 - target_mask.repeat(args.deblur_images, 1), near, far)
        empty_loss += w_empty * compute_empty_loss(ret['z_vals'], ret['sigma_st'], 
                                                   -target_depth.repeat(args.deblur_images), 
                                                   target_mask.repeat(args.deblur_images, 1), near, far)
        
        pose_first, pose_last = spline_poses[0], spline_poses[args.deblur_images - 1]
        pose_prev_last = spline_poses_prev[args.deblur_images - 1]
        pose_post_first = spline_poses_post[0]
        if img_i == 0:
            pose_loss = args.w_pose * torch.mean(torch.abs(pose_last - pose_post_first))
        elif img_i == num_img - 1:
            pose_loss = args.w_pose * torch.mean(torch.abs(pose_first - pose_prev_last))
        else:
            pose_loss = args.w_pose * (torch.mean(torch.abs(pose_last - pose_post_first)) + 
                                       torch.mean(torch.abs(pose_first - pose_prev_last)))
        
        loss = sf_reg_loss + sf_cycle_loss + render_loss + flow_loss + \
               sf_sm_loss + prob_reg_loss + depth_loss + entropy_loss + \
               gradient_loss + pose_loss + empty_loss
        print('render_loss ', render_loss.item(), 
              ' bidirection_loss ', sf_cycle_loss.item(), 
              ' sf_reg_loss ', sf_reg_loss.item())
        print('depth_loss ', depth_loss.item(), 
              ' flow_loss ', flow_loss.item(), 
              ' sf_sm_loss ', sf_sm_loss.item())
        print('prob_reg_loss ', prob_reg_loss.item(),
              ' entropy_loss ', entropy_loss.item(), 
              ' gradient_loss ', gradient_loss.item())
        print('pose_loss ', pose_loss.item(), 
              'empty_loss', empty_loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer_se3.step()
        optimizer_basis.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        new_lrate_basis = args.lrate * 0.25 * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_basis.param_groups:
            param_group['lr'] = new_lrate_basis
        
        decay_rate_pose = 0.01
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose

        if i % args.i_tqdm == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}, PSNR: {psnr.item()}")
        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/render_loss", render_loss.item(), i)
            writer.add_scalar("train/depth_loss", depth_loss.item(), i)
            writer.add_scalar("train/flow_loss", flow_loss.item(), i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)
            writer.add_scalar("train/sf_reg_loss", sf_reg_loss.item(), i)
            writer.add_scalar("train/sf_cycle_loss", sf_cycle_loss.item(), i)
            writer.add_scalar("train/sf_sm_loss", sf_sm_loss.item(), i)
            writer.add_scalar("train/entropy_loss", entropy_loss.item(), i)
            writer.add_scalar("train/gradient_loss", gradient_loss.item(), i)
            writer.add_scalar("train/pose_loss", pose_loss.item(), i)
            writer.add_scalar("train/empty_loss", empty_loss.item(), i)

        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'dynamic_model_state_dict': render_kwargs_train['dynamic_model'].state_dict(),
                    'static_model_state_dict': render_kwargs_train['static_model'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_se3_state_dict': optimizer_se3.state_dict(),
                    'optimizer_basis_state_dict': optimizer_basis.state_dict(),
                    'se3_state_dict': render_kwargs_train['se3'].state_dict(),
                    'trajectory_basis_state_dict': render_kwargs_train['trajectory_basis'].state_dict(),
                    'dynamic_model_fine_state_dict': render_kwargs_train['dynamic_model_fine'].state_dict(),
                    'static_model_fine_state_dict': render_kwargs_train['static_model_fine'].state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'dynamic_model_state_dict': render_kwargs_train['dynamic_model'].state_dict(),
                    'static_model_state_dict': render_kwargs_train['static_model'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_se3_state_dict': optimizer_se3.state_dict(),
                    'optimizer_basis_state_dict': optimizer_basis.state_dict(),
                    'se3_state_dict': render_kwargs_train['se3'].state_dict(),
                    'trajectory_basis_state_dict': render_kwargs_train['trajectory_basis'].state_dict(),
                }, path)
            print('Saved checkpoints at', path)
        
        if i % args.i_img == 0:
            target_blur = images[img_i]
            target_depth = depths[img_i] - torch.min(depths[img_i])
            if sharp_images is not None:
                target_sharp = sharp_images[img_i]
            
            with torch.no_grad():
                spline_poses = get_pose(args, img_i, render_kwargs_train['se3'])
                pose = spline_poses[args.deblur_images // 2]
                img_idx_embed = img_i * args.deblur_images + args.deblur_images // 2
                # img_idx_embed = img_idx_embed / (num_img * args.deblur_images - 1) * 2. - 1.0
                ret = render_image_test(args, i, img_idx_embed, num_img, pose, H, W, K, **render_kwargs_test)

                writer.add_image("val/rgb_map", torch.clamp(ret['rgb_map'], 0., 1.), 
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map", normalize_depth(ret['depth_map']), 
                                 global_step=i, dataformats='HW')
                writer.add_image("val/rgb_map_st", torch.clamp(ret['rgb_map_st'], 0., 1.), 
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_st", normalize_depth(ret['depth_map_st']), 
                                 global_step=i, dataformats='HW')
                writer.add_image("val/rgb_map_dy", torch.clamp(ret['rgb_map_dy'], 0., 1.), 
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_dy", normalize_depth(ret['depth_map_dy']), 
                                 global_step=i, dataformats='HW')
                writer.add_image("val/rgb_gt_blur", target_blur, 
                                 global_step=i, dataformats='HWC')
                writer.add_image("val/disp_gt", 
                                 torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.), 
                                 global_step=i, dataformats='HW')
                writer.add_image("val/weights_map_dd", ret['weights_map_dd'], 
                                 global_step=i, dataformats='HW')
                if sharp_images is not None:
                    writer.add_image("val/rgb_gt_sharp", target_sharp, 
                                     global_step=i, dataformats='HWC')
        
        global_step += 1

   

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()