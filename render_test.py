import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from config import config_parser
from load_llff_data import *
from nerf_model import create_nerf
from spline import *
from render import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)


def get_video(rgb_dir, depth_dir):
    rgb_filenames = os.listdir(rgb_dir)
    rgb_filenames.sort()
    depth_filenames = os.listdir(depth_dir)
    depth_filenames.sort()

    rgbs = []
    depths = []
    for i in range(len(rgb_filenames)):
        bgr = cv2.imread(os.path.join(rgb_dir, rgb_filenames[i]))
        b,g,r = cv2.split(bgr)
        rgb = cv2.merge([r,g,b])
        rgbs.append(rgb)

        depth_ = cv2.imread(os.path.join(depth_dir, depth_filenames[i]))
        b,g,r = cv2.split(depth_)
        depth = cv2.merge([r,g,b])
        depths.append(depth)
        
    rgbs = np.stack(rgbs, axis=0)
    depths = np.stack(depths, axis=0)
    print('RGBs: ', rgbs.shape, 'Depths: ', depths.shape)

    imageio.mimwrite(os.path.join(rgb_dir, 'video_rgb.mp4'), rgbs, fps=30, quality=8)
    imageio.mimwrite(os.path.join(depth_dir, 'video_depth.mp4'), depths, fps=30, quality=8)


def get_video_input_sharp(rgb_dir, depth_dir, deblur_images):
    rgb_filenames = os.listdir(rgb_dir)
    rgb_filenames.sort()
    depth_filenames = os.listdir(depth_dir)
    depth_filenames.sort()

    rgbs = []
    depths = []
    print('Begin render input like sharp video.')
    for i in range(len(rgb_filenames)):
        if i % deblur_images == (deblur_images // 2):
            print('idx: ', i)
            bgr = cv2.imread(os.path.join(rgb_dir, rgb_filenames[i]))
            b,g,r = cv2.split(bgr)
            rgb = cv2.merge([r,g,b])
            rgbs.append(rgb)

            depth_ = cv2.imread(os.path.join(depth_dir, depth_filenames[i]))
            b,g,r = cv2.split(depth_)
            depth = cv2.merge([r,g,b])
            depths.append(depth)
        
    rgbs = np.stack(rgbs, axis=0)
    depths = np.stack(depths, axis=0)
    print('RGBs: ', rgbs.shape, 'Depths: ', depths.shape)

    imageio.mimwrite(os.path.join(rgb_dir, 'video_rgb_input_like.mp4'), rgbs, fps=30, quality=8)
    imageio.mimwrite(os.path.join(depth_dir, 'video_depth_input_like.mp4'), depths, fps=30, quality=8)



def render_test():
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
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    num_img = float(images.shape[0])
    print('target_idx ', args.target_idx)
    target = args.target_idx
    num = args.deblur_images


    print('Render view interpolation.')
    with torch.no_grad():
        spline_poses = get_pose(args, target, render_kwargs_train['se3'])
        pose = spline_poses[num // 2]
        pose_idx = torch.cat([pose, torch.tensor(hwf).unsqueeze(-1)], -1).cpu().numpy()
        render_poses = render_wander_path(pose_idx)
        render_poses = np.array(render_poses).astype(np.float32)
        render_poses = torch.Tensor(render_poses).to(device)
        img_idx_embed = target * num + num // 2
        # img_idx_embed = img_idx_embed / (num_img * num - 1) * 2. - 1.0
        testsavedir = os.path.join(basedir, expname, 
                                   'render-view-interp-%03d' % target + \
                                   '_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)

        save_img_dir = os.path.join(testsavedir, 'images')
        save_depth_dir = os.path.join(testsavedir, 'depths')
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_depth_dir, exist_ok=True)
        for i in range(0, (render_poses.shape[0])):
            print('render view-interp: ', i)
            ret = render_image_test(args, 0, img_idx_embed, num_img, render_poses[i, :3, :4], H, W, K, **render_kwargs_test)

            depth = torch.clamp(ret['depth_map']/percentile(ret['depth_map'], 97), 0., 1.)  #1./disp
            rgb = ret['rgb_map'].cpu().numpy()#.append(ret['rgb_map_ref'].cpu().numpy())

            if save_img_dir is not None:
                rgb8 = to8b(rgb)
                depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

                start_y = (rgb8.shape[1] - W) // 2
                rgb8 = rgb8[:, start_y : start_y + W, :]
                depth8 = depth8[:, start_y : start_y + W, :]

                filename = os.path.join(save_img_dir, '{:05d}-{:03d}.jpg'.format(target, i))
                imageio.imwrite(filename, rgb8)
                filename = os.path.join(save_img_dir, '{:05d}-{:03d}.jpg'.format(target, i+60))
                imageio.imwrite(filename, rgb8)
                filename = os.path.join(save_depth_dir, '{:05d}-{:03d}.jpg'.format(target, i))
                imageio.imwrite(filename, depth8)
                filename = os.path.join(save_depth_dir, '{:05d}-{:03d}.jpg'.format(target, i+60))
                imageio.imwrite(filename, depth8)
        
        get_video(save_img_dir, save_depth_dir)
    

    print('Render time interpolation.')
    with torch.no_grad():
        img_idx_embed = torch.linspace((target - 8) * num, (target + 8) * num - 1, steps=16 * num)
        # img_idx_embed = img_idx_embed / (num_img * num - 1) * 2. - 1.0
        pose = torch.Tensor(pose).to(device)
        testsavedir = os.path.join(basedir, expname, 
                                   'render-time-interp-%03d' % target + \
                                   '_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)

        save_img_dir = os.path.join(testsavedir, 'images')
        save_depth_dir = os.path.join(testsavedir, 'depths')
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_depth_dir, exist_ok=True)
        for i in range(0, (img_idx_embed.shape[0])):
            print('render time-interp: ', i, img_idx_embed[i])
            ret = render_image_test(args, 0, img_idx_embed[i], num_img, pose, H, W, K, **render_kwargs_test)

            depth = torch.clamp(ret['depth_map']/percentile(ret['depth_map'], 97), 0., 1.)  #1./disp
            rgb = ret['rgb_map'].cpu().numpy()#.append(ret['rgb_map_ref'].cpu().numpy())

            if save_img_dir is not None:
                rgb8 = to8b(rgb)
                depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

                start_y = (rgb8.shape[1] - W) // 2
                rgb8 = rgb8[:, start_y : start_y + W, :]
                depth8 = depth8[:, start_y : start_y + W, :]

                filename = os.path.join(save_img_dir, '{:03d}.jpg'.format(i))
                imageio.imwrite(filename, rgb8)
                filename = os.path.join(save_depth_dir, '{:03d}.jpg'.format(i))
                imageio.imwrite(filename, depth8)
        
        get_video(save_img_dir, save_depth_dir)
    

    print('Render view-time interpolation.')
    with torch.no_grad():
        img_idx_embed = torch.linspace((target - 10) * num, (target + 10) * num - 1, steps=20 * num)
        bt_poses = create_bt_poses(hwf, num_frames=28, max_disp=24.0) 
        bt_poses = bt_poses * 10
        testsavedir = os.path.join(basedir, expname, 
                                   'render-view-time-interp-%03d' % target + \
                                   '_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)

        save_img_dir = os.path.join(testsavedir, 'images')
        save_depth_dir = os.path.join(testsavedir, 'depths')
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_depth_dir, exist_ok=True)
        for i in range(0, (img_idx_embed.shape[0])):
            img_idx = img_idx_embed[i] / (num_img * num - 1) * 2. - 1.0
            ii = int(img_idx_embed[i] // num)
            jj = int(img_idx_embed[i] % num)
            spline_poses = get_pose(args, ii, render_kwargs_train['se3'])
            pose = spline_poses[jj].cpu().numpy()
            int_poses = np.concatenate([pose[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
            int_poses = np.dot(int_poses, bt_poses[i])
            render_pose = torch.Tensor(int_poses).to(device)

            print('render view-time-interp: ', i, img_idx)
            ret = render_image_test(args, 0, img_idx_embed[i], num_img, render_pose[:3, :4], H, W, K, **render_kwargs_test)

            depth = torch.clamp(ret['depth_map']/percentile(ret['depth_map'], 97), 0., 1.)  #1./disp
            rgb = ret['rgb_map'].cpu().numpy()#.append(ret['rgb_map_ref'].cpu().numpy())

            if save_img_dir is not None:
                rgb8 = to8b(rgb)
                depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

                start_y = (rgb8.shape[1] - W) // 2
                rgb8 = rgb8[:, start_y : start_y + W, :]
                depth8 = depth8[:, start_y : start_y + W, :]

                filename = os.path.join(save_img_dir, '{:03d}.jpg'.format(i))
                imageio.imwrite(filename, rgb8)
                filename = os.path.join(save_depth_dir, '{:03d}.jpg'.format(i))
                imageio.imwrite(filename, depth8)
        
        get_video(save_img_dir, save_depth_dir)
    

    print('Render input sharp video.')
    with torch.no_grad():
        img_idx_embed = torch.linspace(0 * num, num_img * num - 1, steps=int(num_img * num))
        # img_idx_embed = img_idx_embed / (num_img * num - 1) * 2. - 1.0
        testsavedir = os.path.join(basedir, expname, 
                                   'render-input-sharp-video' + \
                                   '_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)

        save_img_dir = os.path.join(testsavedir, 'images')
        save_depth_dir = os.path.join(testsavedir, 'depths')
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_depth_dir, exist_ok=True)
        for i in range(int(num_img)):
            spline_poses = get_pose(args, i, render_kwargs_train['se3'])
            for j in range(num):
                pose = spline_poses[j]
                idx = (i * num) + j
                print('render input sharp video: ', idx)
                ret = render_image_test(args, 0, img_idx_embed[idx], num_img, pose, H, W, K, **render_kwargs_test)

                depth = torch.clamp(ret['depth_map']/percentile(ret['depth_map'], 97), 0., 1.)  #1./disp
                rgb = ret['rgb_map'].cpu().numpy()#.append(ret['rgb_map_ref'].cpu().numpy())

                if save_img_dir is not None:
                    rgb8 = to8b(rgb)
                    # depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())
                    depth8 = to8b(depth.cpu().numpy())

                    start_y = (rgb8.shape[1] - W) // 2
                    rgb8 = rgb8[:, start_y : start_y + W, :]
                    # depth8 = depth8[:, start_y : start_y + W, :]
                    depth8 = depth8[:, start_y : start_y + W]

                    colormap = plt.get_cmap('inferno')

                    filename = os.path.join(save_img_dir, '{:03d}.jpg'.format(idx))
                    imageio.imwrite(filename, rgb8)
                    filename = os.path.join(save_depth_dir, '{:03d}.jpg'.format(idx))
                    depth8_ = (colormap(depth8) * 2 ** 8).astype(np.uint8)[:, :, :3]
                    imageio.imwrite(filename, depth8_)
        
        get_video(save_img_dir, save_depth_dir)
        get_video_input_sharp(save_img_dir, save_depth_dir, num)



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    render_test()