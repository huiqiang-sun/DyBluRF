import numpy as np
import os, imageio
import torch
import cv2


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)
    

def load_imgs(path, start_frame, end_frame):
    imgfiles = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles[start_frame:end_frame]
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)
    imgs = torch.tensor(imgs)

    return imgs


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2) 
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def render_wander_path(c2w):
    hwf = c2w[:,4:5]
    num_frames = 60
    max_disp = 24.0 # 64 , 48

    max_trans = max_disp / hwf[2][0] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0 #* 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)#[np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        # print('render_pose ', render_pose.shape)
        # sys.exit()
        output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
    
    return output_poses


def load_data(basedir, start_frame, end_frame, factor=None, width=None, 
              height=None, evaluation=False):
    print('factor: ', factor, ' height: ', height, ' width: ', width)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(round(sh[1] / factor))
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(round(sh[0] / factor))
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles[start_frame:end_frame]
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    imgs_sharp_dir = os.path.join(basedir, 'sharp_images')
    if not os.path.exists(imgs_sharp_dir):
        print('No sharp images.')
        imgs_sharp = None
    else:
        imgs_sharp = load_imgs(imgs_sharp_dir, start_frame, end_frame)

    if evaluation:
        return poses, bds, imgs, imgs_sharp
    
    def read_MiDaS_disp(disp_fi, disp_rescale=10.):
        disp = np.load(disp_fi)
        return disp
    
    disp_dir = os.path.join(basedir, 'disp')  
    dispfiles = [os.path.join(disp_dir, f) for f in sorted(os.listdir(disp_dir)) if f.endswith('npy')]
    dispfiles = dispfiles[start_frame:end_frame]

    disp = [cv2.resize(read_MiDaS_disp(f, 3.0), 
                       (imgs.shape[1], imgs.shape[0]), 
                       interpolation=cv2.INTER_NEAREST) for f in dispfiles]
    disp = np.stack(disp, -1)

    mask_dir = os.path.join(basedir, 'motion_masks')
    maskfiles = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f.endswith('png')]
    maskfiles = maskfiles[start_frame:end_frame]

    masks = [cv2.resize(imread(f) / 255., (imgs.shape[1], imgs.shape[0]), 
                        interpolation=cv2.INTER_NEAREST) for f in maskfiles]
    masks = np.stack(masks, -1)  
    masks = np.float32(masks > 1e-3)

    motion_coords = []
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        coord_y, coord_x = np.where(mask > 0.1)
        coord = np.stack((coord_y, coord_x), -1)
        motion_coords.append(coord)

    print('images shape: ', imgs.shape, ' disp shape: ', disp.shape)
    assert(imgs.shape[0] == disp.shape[0])
    assert(imgs.shape[0] == masks.shape[0])
    assert(imgs.shape[1] == disp.shape[1])
    assert(imgs.shape[1] == masks.shape[1])

    return poses, bds, imgs, imgs_sharp, disp, masks, motion_coords


def load_data_eva(basedir, start_frame, end_frame, factor=None, width=None, 
              height=None):
    print('factor: ', factor, ' height: ', height, ' width: ', width)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # poses_arr = poses_arr[start_frame:end_frame, ...]

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(round(sh[1] / factor))
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(round(sh[0] / factor))
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles[start_frame:end_frame]

    # if poses.shape[-1] != len(imgfiles):
    #     print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
    #     return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    imgs_sharp_dir = os.path.join(basedir, 'sharp_images')
    if not os.path.exists(imgs_sharp_dir):
        print('No sharp images.')
        imgs_sharp = None
    else:
        imgs_sharp = load_imgs(imgs_sharp_dir, start_frame, end_frame)
    
    imgs_inf_dir = os.path.join(basedir, 'inference_images')
    imgs_inf_files = [os.path.join(imgs_inf_dir, f) for f in sorted(os.listdir(imgs_inf_dir)) if f.endswith('png')]
    imgs_inf_files = imgs_inf_files[start_frame:end_frame]

    imgs_inf = [cv2.resize(imread(f) / 255., (imgs.shape[1], imgs.shape[0]), 
                           interpolation=cv2.INTER_NEAREST) for f in imgs_inf_files]
    imgs_inf = np.stack(imgs_inf, -1)

    print('images shape: ', imgs.shape)
    assert(imgs.shape[0] == imgs_inf.shape[0])
    assert(imgs.shape[1] == imgs_inf.shape[1])

    return poses, bds, imgs, imgs_sharp, imgs_inf


def load_llff_data(basedir, start_frame, end_frame, target_idx=10, recenter=True, 
                   bd_factor=.75, spherify=False, final_height=288):
    
    poses, bds, imgs, imgs_sharp, \
    disp, masks, motion_coords = load_data(basedir, start_frame, end_frame,
                                           height=final_height,
                                           evaluation=False)
    print('Loaded', basedir, bds.min(), bds.max())

    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    disp = np.moveaxis(disp, -1, 0).astype(np.float32)
    masks = np.moveaxis(masks, -1, 0).astype(np.float32)

    sc = 1. if bd_factor is None else 1. / (np.percentile(bds[:, 0], 5) * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    poses = poses[::2, ...] # get camera poses from the left camera
    poses = poses[start_frame:end_frame, ...]
    if poses.shape[0] != len(images):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(images), poses.shape[-1]))
        return
    
    c2w = poses[target_idx, :, :]
    render_poses = render_wander_path(c2w)
    render_poses = np.array(render_poses).astype(np.float32)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    disp = disp.astype(np.float32)
    masks = masks.astype(np.float32)

    return images, poses, bds, imgs_sharp, disp, masks, motion_coords, render_poses, c2w



def load_llff_data_eva(basedir, start_frame, end_frame, target_idx=10, recenter=True, 
                       bd_factor=.75, spherify=False, final_height=288):
    
    poses, bds, imgs, imgs_sharp, \
    imgs_inf = load_data_eva(basedir, start_frame, end_frame,
                             height=final_height)
    print('Loaded', basedir, bds.min(), bds.max())

    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    inf_images = np.moveaxis(imgs_inf, -1, 0).astype(np.float32)

    sc = 1. if bd_factor is None else 1. / (np.percentile(bds[:, 0], 5) * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    poses_train = poses[::2, ...]
    poses_train = poses_train[start_frame:end_frame, ...]
    poses = poses[1::2, ...] # get camera poses from the right camera
    poses = poses[start_frame:end_frame, ...]
    
    if poses.shape[0] != len(images):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(images), poses.shape[-1]))
        return
    
    c2w = poses[target_idx, :, :]
    render_poses = render_wander_path(c2w)
    render_poses = np.array(render_poses).astype(np.float32)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    poses_train = poses_train.astype(np.float32)
    inf_images= inf_images.astype(np.float32)

    return images, poses, bds, imgs_sharp, inf_images, render_poses, c2w, poses_train