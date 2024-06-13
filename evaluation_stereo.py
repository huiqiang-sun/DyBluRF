import torch
import numpy as np
import os
import sys
import time
import models

from config import config_parser
from load_llff_data import *
from nerf_model import *
from spline import *
from render import *
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def evaluation():
    parser = config_parser()
    args = parser.parse_args()

    print('spline numbers: ', args.deblur_images)
    
    K = None
    if args.dataset_type == 'llff':
        images, poses_start, bds, \
        sharp_images, inf_images, \
        render_poses, ref_c2w, poses_train = load_llff_data_eva(args.datadir, args.start_frame, args.end_frame, 
                                                                target_idx=args.target_idx, recenter=True, 
                                                                bd_factor=.9, spherify=args.spherify, 
                                                                final_height=args.final_height)
        hwf = poses_start[0, :3,- 1]
        i_test = []
        i_val = [] 
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        
        poses = poses_start[:, :3, :4]
        poses_start = torch.Tensor(poses_start)
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses = torch.Tensor(poses).to(device)

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
    
    render_kwargs_train, render_kwargs_test, \
    _, _, _, _ = create_nerf(args, poses_start_se3, poses_end_se3, images.shape[0])
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    input_poses_train = torch.Tensor(poses_train[:, :3, :4])
    input_poses_test = poses_start[:, :3, :4]

    num_img = float(images.shape[0])
    num = args.deblur_images    
   
    os.makedirs(os.path.join(basedir, expname, 'output'), exist_ok=True)
    save_path = os.path.join(basedir, expname, 'output')
    
    with torch.no_grad():

        model = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)

        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        count = 0.
        t = time.time()

        # poses = se3_to_SE3(se3)
        # poses = torch.Tensor(poses)

        for i in range(0, int(num_img)):
            print(time.time() - t)
            t = time.time()

            img_idx_embed = i * num + num // 2
            # img_idx_embed = img_idx_embed / (num_img * num - 1) * 2. - 1.0

            input_train_pose = convert3x4_4x4(input_poses_train[i])
            input_test_pose = convert3x4_4x4(input_poses_test[i])
            spline_poses = get_pose(args, i, render_kwargs_test['se3'])
            output_train_pose = convert3x4_4x4(torch.Tensor(spline_poses[num // 2]))

            output_test_pose = input_test_pose @ torch.inverse(input_train_pose) @ output_train_pose

            ret = render_image_test(args, 0, img_idx_embed, num_img, output_test_pose[:3, :4], H, W, K, **render_kwargs_test)
            # ret = render_image_test(args, 0, img_idx_embed, num_img, input_poses_test[i], H, W, K, **render_kwargs_test)

            rgb = ret['rgb_map'].cpu().numpy()

            gt_img_path = os.path.join(args.datadir, 'inference_images', '%05d.png'%i)
            gt_img = cv2.imread(gt_img_path)[:, :, ::-1]
            gt_img = cv2.resize(gt_img, dsize=(rgb.shape[1], rgb.shape[0]), 
                                interpolation=cv2.INTER_AREA)
            gt_img = np.float32(gt_img) / 255

            psnr = peak_signal_noise_ratio(gt_img, rgb)
            ssim = structural_similarity(gt_img, rgb, 
                                                multichannel=True)

            gt_img_0 = im2tensor(gt_img).cuda()
            rgb_0 = im2tensor(rgb).cuda()

            lpips = model.forward(gt_img_0, rgb_0)
            lpips = lpips.item()
            print(psnr, ssim, lpips)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            count += 1

            filename = os.path.join(save_path, 'rgb_{}.jpg'.format(i))
            imageio.imwrite(filename, rgb)
            filename = os.path.join(save_path, 'rgb_{}_gt.jpg'.format(i))
            imageio.imwrite(filename, gt_img)

        mean_psnr = total_psnr / count
        mean_ssim = total_ssim / count
        mean_lpips = total_lpips / count

        print('mean_psnr ', mean_psnr)
        print('mean_ssim ', mean_ssim)
        print('mean_lpips ', mean_lpips)




if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()