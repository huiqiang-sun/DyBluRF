import torch
from run_nerf_helpers import *
import torch.nn.functional as F
import math
import random

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.（包括颜色和体积密度）
        z_vals: [num_rays, num_samples along ray]. Integration time.（表示采样点）
        rays_d: [num_rays, 3]. Direction of each ray.（表示光线的方向向量）
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    return rgb_map, depth_map, weights



def raw2outputs_blending(raw_dy, raw_st, raw_blend_w, z_vals, rays_d, raw_noise_std):

    act_fn = F.relu

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb_dy = torch.sigmoid(raw_dy[..., :3])  # [N_rays, N_samples, 3]
    rgb_st = torch.sigmoid(raw_st[..., :3])  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_dy[...,3].shape) * raw_noise_std

    opacity_dy = act_fn(raw_dy[..., 3] + noise)#.detach() #* raw_blend_w
    opacity_st = act_fn(raw_st[..., 3] + noise)#.detach() #* (1. - raw_blend_w) 

    # alpha with blending weights
    alpha_dy = (1. - torch.exp(-opacity_dy * dists) ) * raw_blend_w
    alpha_st = (1. - torch.exp(-opacity_st * dists)) * (1. - raw_blend_w)

    Ts = torch.cumprod(torch.cat([torch.ones((alpha_dy.shape[0], 1)), 
                                (1. - alpha_dy) * (1. - alpha_st)  + 1e-10], -1), -1)[:, :-1]
    
    weights_dy = Ts * alpha_dy
    weights_st = Ts * alpha_st

    # union map 
    rgb_map = torch.sum(weights_dy[..., None] * rgb_dy + \
                        weights_st[..., None] * rgb_st, -2) 

    weights_mix = weights_dy + weights_st
    depth_map = torch.sum(weights_mix * z_vals, -1)

    # compute dynamic depth only
    alpha_fg = 1. - torch.exp(-opacity_dy * dists)
    weights_fg = alpha_fg * torch.cumprod(torch.cat([torch.ones((alpha_fg.shape[0], 1)), 
                                                                1.-alpha_fg + 1e-10], -1), -1)[:, :-1]
    depth_map_fg = torch.sum(weights_fg * z_vals, -1)
    rgb_map_fg = torch.sum(weights_fg[..., None] * rgb_dy, -2) 

    return rgb_map, depth_map, \
           rgb_map_fg, depth_map_fg, weights_fg, \
           weights_dy, weights_st, weights_mix, \
           opacity_dy, opacity_st, Ts



def raw2outputs_warp(raw_p, z_vals, rays_d, raw_noise_std=0):

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw_p[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw_p[...,3].shape) * raw_noise_std

    act_fn = F.relu
    opacity = act_fn(raw_p[..., 3] + noise)

    alpha = 1. - torch.exp(-opacity * dists)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)

    return rgb_map, depth_map, weights



def get_static_outputs(pts, viewdirs, static_network_fn, static_model, 
                       z_vals, rays_d, raw_noise_std):

    raw_static = static_network_fn(pts[..., :3], viewdirs, static_model)
    raw_rgba_static = raw_static[..., :4]
    raw_blend_w = raw_static[..., 4:]

    rgb_map_st, depth_map_st, weights_st = raw2outputs(raw_rgba_static, z_vals, rays_d, 
                                                       raw_noise_std, white_bkgd=False)

    return rgb_map_st, depth_map_st, weights_st, raw_rgba_static, raw_blend_w[..., 0]



def compute_2d_prob(weights_p_mix, 
                    raw_prob_ref2p):
    prob_map_p = torch.sum(weights_p_mix.detach() * (1.0 - raw_prob_ref2p), -1)
    return prob_map_p



def render(args, step, img_idx, chain_bwd, chain_5frames, poses, ray_idx, 
           num_img, H, W, K, near=0., far=1., training=False, **kwargs):
    num = args.deblur_images
    if training:
        ray_idx_ = ray_idx.repeat(num)
        poses = poses.unsqueeze(1).repeat(1, ray_idx.shape[0], 1, 1).reshape(-1, 3, 4)
        j = ray_idx_.reshape(-1, 1).squeeze() // W
        i = ray_idx_.reshape(-1, 1).squeeze() % W
        rays_o_, rays_d_ = get_specific_rays(i, j, K, poses)
        rays_o_d = torch.stack([rays_o_, rays_d_], 0)
        batch_rays = torch.permute(rays_o_d, [1, 0, 2])
    else:
        rays_list = []
        for p in poses[:, :3, :4]:
            rays_o_, rays_d_ = get_rays(H, W, K, p)
            rays_o_d = torch.stack([rays_o_, rays_d_], 0)
            rays_list.append(rays_o_d)

        rays = torch.stack(rays_list, 0)
        rays = rays.reshape(-1, 2, H * W, 3)
        rays = torch.permute(rays, [0, 2, 1, 3])
        batch_rays = rays[:, ray_idx]
    batch_rays = batch_rays.reshape(-1, 2, 3)
    batch_rays = torch.transpose(batch_rays, 0, 1)

    rays_o, rays_d = batch_rays #[N,3]
    if training:
        img_idx_ = torch.linspace(img_idx * num, (img_idx + 1) * num - 1, steps=num)
        img_idx_embed = img_idx_ / (num_img * num - 1) * 2. - 1.0
        img_idx_embed = img_idx_embed.unsqueeze(1).repeat(1, ray_idx.shape[0]).reshape(-1, 1)
        img_idx_ = img_idx_.unsqueeze(1).repeat(1, ray_idx.shape[0]).reshape(-1, 1)
    else:
        img_idx_embed = img_idx / (num_img * num - 1) * 2. - 1.0
        img_idx_embed = img_idx_embed * torch.ones_like(rays_d[..., :1])
        img_idx_ = img_idx * torch.ones_like(rays_d[..., :1])
    sh = rays_d.shape
    if args.use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    if not args.no_ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, img_idx_embed], -1)

    if args.use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) #[N, 12]

    all_ret = batchify_rays(step, img_idx_, chain_bwd, chain_5frames, num, args.num_basis, 
                            num_img, rays, args.chunk, **kwargs)
    
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
    return all_ret
    

def batchify_rays(step, img_idx, chain_bwd, chain_5frames, num, num_basis, 
                  num_img, rays_flat, chunk=1024*16, **kwargs):
    
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(step, img_idx, chain_bwd, chain_5frames, num, num_basis, 
                          num_img, rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret


def render_rays(step, img_idx, 
                chain_bwd, chain_5frames, num, 
                num_basis, 
                num_img, ray_batch,
                dynamic_network_fn,
                static_network_fn, 
                dynamic_model,
                static_model, 
                se3, 
                trajectory_basis, 
                N_samples,
                perturb=0.,
                N_importance=0,
                dynamic_model_fine=None, 
                static_model_fine=None, 
                use_viewdirs=True, 
                white_bkgd=False, 
                lindisp=False,
                raw_noise_std=0.,
                inference=False):
    
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]
    img_idx_embed = ray_batch[:, 8:9]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    img_idx_rep = img_idx_embed.unsqueeze(1).repeat(1, N_samples, 1)
    pts_ref = torch.cat([pts, img_idx_rep], -1)

    rgb_map_st, depth_map_st, _, \
    raw_rgba_st, raw_blend_w = get_static_outputs(pts_ref, viewdirs, static_network_fn, 
                                                  static_model, z_vals, rays_d, raw_noise_std)
    
    raw_dy = dynamic_network_fn(pts_ref, viewdirs, dynamic_model)
    raw_rgba_dy = raw_dy[:, :, :4]
    raw_coeff = raw_dy[:, :, 4:(4 + num_basis * 3)]
    raw_coeff_x = raw_coeff[..., 0 : num_basis]
    raw_coeff_y = raw_coeff[..., num_basis : num_basis * 2]
    raw_coeff_z = raw_coeff[..., num_basis * 2 : num_basis * 3]

    rgb_map, depth_map, \
    rgb_map_dy, depth_map_dy, weights_dy, \
    weights_dd, weights_st, weights_mix, \
    opacity_dy, opacity_st, Ts = raw2outputs_blending(raw_rgba_dy, raw_rgba_st,
                                                      raw_blend_w, z_vals, rays_d, 
                                                      raw_noise_std)
    weights_map_dd = torch.sum(weights_dd, -1).detach()

    ret = {'rgb_map': rgb_map, 'depth_map' : depth_map, 
           'rgb_map_st':rgb_map_st, 'depth_map_st':depth_map_st, 
           'rgb_map_dy':rgb_map_dy, 'depth_map_dy':depth_map_dy, 
           'weights_map_dd': weights_map_dd, 'weights_dy_only': weights_dd, 
           'weights_dy': weights_dy, 'weights_st': weights_st, 'weights_mix': weights_mix, 
           'opacity_dy': opacity_dy, 'opacity_st':opacity_st, 'Ts': Ts}
    
    if inference:
        return ret
    else:
        traj_basis_ref = trajectory_basis.weight[img_idx.squeeze().long()] # [N_rays, 6]
        if img_idx[0] == 0:
            traj_basis_post = trajectory_basis.weight[(img_idx + num).squeeze().long()]
            traj_basis_prev = traj_basis_ref
        elif img_idx[-1] == (num * num_img - 1):
            traj_basis_post = traj_basis_ref
            traj_basis_prev = trajectory_basis.weight[(img_idx - num).squeeze().long()]
        else:
            traj_basis_post = trajectory_basis.weight[(img_idx + num).squeeze().long()]
            traj_basis_prev = trajectory_basis.weight[(img_idx - num).squeeze().long()]
        traj_pts_ref = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, traj_basis_ref)
        traj_pts_post = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, traj_basis_post)
        traj_pts_prev = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, traj_basis_prev)
        raw_sf_ref2prev = traj_pts_prev - traj_pts_ref
        raw_sf_ref2post = traj_pts_post - traj_pts_ref
        ret['raw_sf_ref2prev'] = raw_sf_ref2prev
        ret['raw_sf_ref2post'] = raw_sf_ref2post
        ret['raw_pts_ref'] = pts_ref[:, :, :3]
        ret['raw_blend_w'] = raw_blend_w
        ret['z_vals'] = z_vals
        ret['sigma_dy'] = raw_rgba_dy[..., 3]  # [N_rays, N_samples]
        ret['sigma_st'] = raw_rgba_st[..., 3]

    img_idx_rep_post = img_idx_rep + num / (num_img * num - 1) * 2.
    pts_post = torch.cat([(pts_ref[:, :, :3] + raw_sf_ref2post), img_idx_rep_post] , -1)

    img_idx_rep_prev = img_idx_rep - num / (num_img * num - 1) * 2.
    pts_prev = torch.cat([(pts_ref[:, :, :3] + raw_sf_ref2prev), img_idx_rep_prev] , -1)
    
    raw_prev = dynamic_network_fn(pts_prev, viewdirs, dynamic_model)
    raw_rgba_prev = raw_prev[:, :, :4]
    raw_coeff_prev = raw_prev[:, :, 4:(4 + num_basis * 3)]
    raw_coeff_x_prev = raw_coeff_prev[..., 0 : num_basis]
    raw_coeff_y_prev = raw_coeff_prev[..., num_basis : num_basis * 2]
    raw_coeff_z_prev = raw_coeff_prev[..., num_basis * 2 : num_basis * 3]

    if img_idx[0] <= 7:
        traj_basis_prevprev = traj_basis_prev
    else:
        traj_basis_prevprev = trajectory_basis.weight[(img_idx - 2 * num).squeeze().long()]
    traj_pts_prev_ref = compute_traj_pts(raw_coeff_x_prev, raw_coeff_y_prev, raw_coeff_z_prev, traj_basis_prev)
    traj_pts_prev_post = compute_traj_pts(raw_coeff_x_prev, raw_coeff_y_prev, raw_coeff_z_prev, traj_basis_ref)
    traj_pts_prev_prev = compute_traj_pts(raw_coeff_x_prev, raw_coeff_y_prev, raw_coeff_z_prev, traj_basis_prevprev)
    raw_sf_prev2prevprev = traj_pts_prev_prev - traj_pts_prev_ref
    raw_sf_prev2ref = traj_pts_prev_post - traj_pts_prev_ref

    rgb_map_prev_dy, _, weights_prev_dy = raw2outputs_warp(raw_rgba_prev,
                                                           z_vals, rays_d, 
                                                           raw_noise_std)
    ret['raw_sf_prev2ref'] = raw_sf_prev2ref
    ret['rgb_map_prev_dy'] = rgb_map_prev_dy

    raw_post = dynamic_network_fn(pts_post, viewdirs, dynamic_model)
    raw_rgba_post = raw_post[:, :, :4]
    raw_coeff_post = raw_post[:, :, 4:(4 + num_basis * 3)]
    raw_coeff_x_post = raw_coeff_post[..., 0 : num_basis]
    raw_coeff_y_post = raw_coeff_post[..., num_basis : num_basis * 2]
    raw_coeff_z_post = raw_coeff_post[..., num_basis * 2 : num_basis * 3]

    if img_idx[-1] >= (num * (num_img - 1) - 1):
        traj_basis_postpost = traj_basis_post
    else:
        traj_basis_postpost = trajectory_basis.weight[(img_idx + 2 * num).squeeze().long()]
    traj_pts_post_ref = compute_traj_pts(raw_coeff_x_post, raw_coeff_y_post, raw_coeff_z_post, traj_basis_post)
    traj_pts_post_post = compute_traj_pts(raw_coeff_x_post, raw_coeff_y_post, raw_coeff_z_post, traj_basis_postpost)
    traj_pts_post_prev = compute_traj_pts(raw_coeff_x_post, raw_coeff_y_post, raw_coeff_z_post, traj_basis_ref)
    raw_sf_post2ref = traj_pts_post_prev - traj_pts_post_ref
    raw_sf_post2postpost = traj_pts_post_post - traj_pts_post_ref

    rgb_map_post_dy, _, weights_post_dy = raw2outputs_warp(raw_rgba_post,
                                                           z_vals, rays_d, 
                                                           raw_noise_std)
    ret['raw_sf_post2ref'] = raw_sf_post2ref
    ret['rgb_map_post_dy'] = rgb_map_post_dy

    raw_prob_ref2prev = raw_dy[:, :, -2]
    raw_prob_ref2post = raw_dy[:, :, -1]

    prob_map_prev = compute_2d_prob(weights_prev_dy, raw_prob_ref2prev)
    prob_map_post = compute_2d_prob(weights_post_dy, raw_prob_ref2post)

    ret['prob_map_prev'] = prob_map_prev
    ret['prob_map_post'] = prob_map_post
    ret['raw_prob_ref2prev'] = raw_prob_ref2prev
    ret['raw_prob_ref2post'] = raw_prob_ref2post
    ret['raw_pts_post'] = pts_post[:, :, :3]
    ret['raw_pts_prev'] = pts_prev[:, :, :3]

    
    gate = math.floor(math.exp(step * 1e-5)) + 1
    ref_i = int(img_idx[0] // num)
    ref_prob = [i for i in range(max(ref_i-gate, 0), min(ref_i+gate+1, int(num_img)))
                if i not in [ref_i-1, ref_i, ref_i+1]]
    ref_pp = random.sample(ref_prob, 1)[0]
    offset = ref_pp - ref_i

    traj_basis_pp = trajectory_basis.weight[(img_idx + offset * num).squeeze().long()]
    traj_pts_pp = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, traj_basis_pp)
    raw_sf_ref2pp = traj_pts_pp - traj_pts_ref

    img_idx_rep_pp = img_idx_rep + offset * num / (num_img * num - 1) * 2.
    pts_pp = torch.cat([(pts_ref[:, :, :3] + raw_sf_ref2pp), img_idx_rep_pp] , -1)
    ret['raw_pts_pp'] = pts_pp[:, :, :3]

    if chain_5frames:
        raw_pp = dynamic_network_fn(pts_pp, viewdirs, dynamic_model)
        raw_rgba_pp = raw_pp[:, :, :4]

        # render from t - 2
        rgb_map_pp_dy, _, weights_pp_dy = raw2outputs_warp(raw_rgba_pp, 
                                                           z_vals, rays_d, 
                                                           raw_noise_std)

        ret['rgb_map_pp_dy'] = rgb_map_pp_dy

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                breakpoint()

    return ret



def render_image_test(args, step, img_idx, num_img, pose, H, W, K, near=0., far=1., **kwargs):
    
    rays_o, rays_d = get_rays(H, W, K, pose)
    sh = rays_d.shape
    if args.use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    if not args.no_ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # img_idx_embed = img_idx * args.deblur_images + args.deblur_images // 2
    img_idx_embed = img_idx / (num_img * args.deblur_images - 1) * 2. - 1.0
    img_idx_embed = img_idx_embed * torch.ones_like(rays_d[..., :1])
    img_idx_ = img_idx * torch.ones_like(rays_d[..., :1])

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, img_idx_embed], -1)

    if args.use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    
    all_ret = batchify_rays(step, img_idx_, 0, False, args.deblur_images, args.num_basis, 
                            num_img, rays, 1024*16, **kwargs)
    
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
    return all_ret