import torch
import numpy as np

import spline

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def get_pose(args, img_idx, se3):
    if args.linear:
        se3_start = se3.weight[:, :6][img_idx : (img_idx + 1)]
        se3_end = se3.weight[:, 6:][img_idx : (img_idx + 1)]
        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)
        seg_pos_x = torch.arange(se3_start.shape[0]).reshape([se3_start.shape[0], 1]).repeat(1, args.deblur_images)

        se3_start = se3_start[seg_pos_x, :]
        se3_end = se3_end[seg_pos_x, :]

        spline_poses = spline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
    else:
        se3_0 = se3.weight[:, :6][img_idx : (img_idx + 1)]
        se3_1 = se3.weight[:, 6:12][img_idx : (img_idx + 1)]
        se3_2 = se3.weight[:, 12:18][img_idx : (img_idx + 1)]
        se3_3 = se3.weight[:, 18:][img_idx : (img_idx + 1)]
        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, args.deblur_images)

        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]

        spline_poses = spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, pose_nums, args.deblur_images)
    
    return spline_poses



def get_specific_rays(i, j, K, c2w):
    # i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
    #                       torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    # i = i.t()
    # j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[..., :3, :3], -1)
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[..., :3, -1]
    return rays_o, rays_d


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d



def compute_mse(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( (pred - gt)**2 * mask_rep )/ num_pix


def compute_mae(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( torch.abs(pred - gt) * mask_rep )/ num_pix


def compute_depth_loss(pred_depth, gt_depth):   
    # pred_depth_e = NDC2Euclidean(pred_depth_ndc)
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))


def NDC2Euclidean(xyz_ndc, H, W, f):
    z_e = 2./ (xyz_ndc[..., 2:3] - 1. + 1e-6)
    x_e = - xyz_ndc[..., 0:1] * z_e * W/ (2. * f)
    y_e = - xyz_ndc[..., 1:2] * z_e * H/ (2. * f)

    xyz_e = torch.cat([x_e, y_e, z_e], -1)
 
    return xyz_e


def perspective_projection(pts_3d, h, w, f):
    pts_2d = torch.cat([pts_3d[..., 0:1] * f/-pts_3d[..., 2:3] + w/2., 
                        -pts_3d[..., 1:2] * f/-pts_3d[..., 2:3] + h/2.], dim=-1)

    return pts_2d    



def se3_transform_points(pts_ref, raw_rot_ref2prev, raw_trans_ref2prev):
    pts_prev = torch.squeeze(torch.matmul(raw_rot_ref2prev, pts_ref[..., :3].unsqueeze(-1)) + raw_trans_ref2prev)
    return pts_prev


def compute_sf_sm_loss(pts_1_ndc, pts_2_ndc, H, W, f):
    # sigma = 2.
    n = pts_1_ndc.shape[1]

    pts_1_ndc_close = pts_1_ndc[..., :int(n * 0.95), :]
    pts_2_ndc_close = pts_2_ndc[..., :int(n * 0.95), :]

    pts_3d_1_world = NDC2Euclidean(pts_1_ndc_close, H, W, f)
    pts_3d_2_world = NDC2Euclidean(pts_2_ndc_close, H, W, f)
        
    # dist = torch.norm(pts_3d_1_world[..., :-1, :] - pts_3d_1_world[..., 1:, :], 
                      # dim=-1, keepdim=True)
    # weights = torch.exp(-dist * sigma).detach()

    # scene flow 
    scene_flow_world = pts_3d_1_world - pts_3d_2_world

    return torch.mean(torch.abs(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :]))


def compute_sf_lke_loss(pts_ref_ndc, pts_post_ndc, pts_prev_ndc, H, W, f):
    n = pts_ref_ndc.shape[1]

    pts_ref_ndc_close = pts_ref_ndc[..., :int(n * 0.9), :]
    pts_post_ndc_close = pts_post_ndc[..., :int(n * 0.9), :]
    pts_prev_ndc_close = pts_prev_ndc[..., :int(n * 0.9), :]

    pts_3d_ref_world = NDC2Euclidean(pts_ref_ndc_close, 
                                     H, W, f)
    pts_3d_post_world = NDC2Euclidean(pts_post_ndc_close, 
                                     H, W, f)
    pts_3d_prev_world = NDC2Euclidean(pts_prev_ndc_close, 
                                     H, W, f)
    
    # scene flow 
    scene_flow_w_ref2post = pts_3d_post_world - pts_3d_ref_world
    scene_flow_w_prev2ref = pts_3d_ref_world - pts_3d_prev_world

    return 0.5 * torch.mean((scene_flow_w_ref2post - scene_flow_w_prev2ref) ** 2)


def normalize_depth(depth):
    # depth_sm = depth - torch.min(depth)
    return torch.clamp(depth/percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def create_bt_poses(hwf, num_frames=40, max_disp=32.):
    # num_frames = 40
    # max_disp = 32.0 # 64 , 48

    max_trans = max_disp / hwf[2] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    z_shift = -max_trans / 6.#-12.0

    print(z_shift)

    init_pos = np.arcsin(-z_shift / max_trans) * float(num_frames) / (2.0 * np.pi)

    max_trans = max_disp / hwf[2] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /2.0 #* 3.0 / 4.0
        z_trans = 0.#z_shift + max_trans * np.sin(2.0 * np.pi * float(init_pos + i) / float(num_frames))

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)#[np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()
        output_poses.append(i_pose)

    return output_poses


def read_optical_flow(basedir, img_i, start_frame, fwd):
    import os
    flow_dir = os.path.join(basedir, 'flow_i1')

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(start_frame + img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      fwd_mask = np.float32(fwd_mask)  
      
      return fwd_flow, fwd_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(start_frame + img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      bwd_mask = np.float32(bwd_mask)

      return bwd_flow, bwd_mask


def compute_optical_flow(pose_post, pose_ref, pose_prev, H, W, focal,
                         weights, pts_post, pts_prev, n_dim=1):
    pts_2d_post = projection_from_ndc(pose_post, H, W, focal, 
                                      weights, pts_post, n_dim)
    pts_2d_prev = projection_from_ndc(pose_prev, H, W, focal, 
                                      weights, pts_prev, n_dim)

    return pts_2d_post, pts_2d_prev


def projection_from_ndc(c2w, H, W, f, weights_ref, raw_pts, n_dim=1):
    R_w2c = c2w[:3, :3].transpose(0, 1)
    t_w2c = -torch.matmul(R_w2c, c2w[:3, 3:])

    pts_3d = torch.sum(weights_ref[...,None] * raw_pts, -2)  # [N_rays, 3]

    pts_3d_e_world = NDC2Euclidean(pts_3d, H, W, f)

    if n_dim == 1:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0), 
                                              t_w2c.unsqueeze(0))
    else:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0).unsqueeze(0), 
                                              t_w2c.unsqueeze(0).unsqueeze(0))

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f)

    return pts_2d


def compare_flow(p1, p2, p_target):
    flow_1 = p1 - p_target
    flow_2 = p2 - p_target
    flow_norm_1 = torch.norm(flow_1, p=2, dim=-1, keepdim=True)
    flow_norm_2 = torch.norm(flow_2, p=2, dim=-1, keepdim=True)
    mask = torch.where(flow_norm_1 > flow_norm_2, 1, 0)
    output = p1 * mask + p2 * (1 - mask)
    
    return output


def compute_gradient_loss(pred_depth, gt_depth):
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    diff_pred = pred_depth_n[1:] - pred_depth_n[:-1]
    diff_gt = gt_depth_n[1:] - gt_depth_n[:-1]

    return torch.mean(torch.abs(diff_pred - diff_gt))


def compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis_i):
    return torch.cat(
    [
        torch.sum(raw_coeff_x * trajectory_basis_i.unsqueeze(1), axis=-1, keepdim=True),
        torch.sum(raw_coeff_y * trajectory_basis_i.unsqueeze(1), axis=-1, keepdim=True),
        torch.sum(raw_coeff_z * trajectory_basis_i.unsqueeze(1), axis=-1, keepdim=True),
    ],
    dim=-1,
    )


def compute_empty_loss(z_vals, weights, gt_depth, mask, near, far):
    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth = (gt_depth - t_gt) / (s_gt + 1e-8)
    
    gt_max = torch.max(gt_depth)
    gt_min = torch.min(gt_depth)
    gt_depth_n = (gt_depth - gt_min) / (gt_max - gt_min + 1e-8)

    t_front = gt_depth_n - (0.05 * (far - near))
    t_front = t_front[...,None]
    m = torch.zeros_like(z_vals)

    result_front = torch.searchsorted(z_vals, t_front, right=False)

    sel = torch.arange(m.shape[-1]).repeat(m.shape[0], 1)
    mask_ = sel < result_front
    m.masked_fill_(mask_, value=1)
    m = m * (1 - mask)
    m = m.detach()
    num_pix = torch.sum(m) + 1e-8

    return torch.sum(weights**2 * m) / num_pix