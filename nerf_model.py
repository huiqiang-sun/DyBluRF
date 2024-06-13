import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim



class dyNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True, num_basis=6):
        super(dyNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        self.coeff_linear = nn.Linear(W, num_basis * 3)
        # self.coeff_linear.weight.data.fill_(0.0)
        # self.coeff_linear.bias.data.fill_(0.0)

        self.prob_linear = nn.Linear(W, 2)

    def forward(self, x):

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x 

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        pred_coeff = self.coeff_linear(h)
        prob = nn.functional.sigmoid(self.prob_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, pred_coeff, prob], dim=-1)


class stNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        super(stNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
        self.w_linear = nn.Linear(W, 1)

    def forward(self, x):

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x 

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        v = nn.functional.sigmoid(self.w_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, v], -1)



def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*16):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs[:, :, :3].shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def init_dct_basis(num_basis, num_frames):
    """Initialize motion basis with DCT coefficient."""
    T = num_frames
    K = num_basis
    dct_basis = torch.zeros([T, K])

    for t in range(T):
        for k in range(1, K + 1):
            dct_basis[t, k - 1] = np.sqrt(2.0 / T) * np.cos(
                np.pi / (2.0 * T) * (2 * t + 1) * k
            )

    return dct_basis



def create_nerf(args, poses_start_se3, poses_end_se3, num_frames):

    if args.linear:
        low, high = 0.0001, 0.005
        rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
        poses_start_se3 = poses_start_se3 + rand

        se3 = torch.nn.Embedding(poses_start_se3.shape[0], 6*2)
        start_end = torch.cat([poses_start_se3, poses_end_se3], -1)
        se3.weight.data = torch.nn.Parameter(start_end)
    else:
        low, high = 0.0001, 0.01
        rand1 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
        rand2 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
        rand3 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
        poses_se3_1 = poses_start_se3 + rand1
        poses_se3_2 = poses_start_se3 + rand2
        poses_se3_3 = poses_start_se3 + rand3

        se3 = torch.nn.Embedding(poses_start_se3.shape[0], 6*4)
        start_end = torch.cat([poses_start_se3, poses_se3_1, poses_se3_2, poses_se3_3], -1)
        se3.weight.data = torch.nn.Parameter(start_end)
    grad_vars_se3 = list(se3.parameters())
    optimizer_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.pose_lrate)
    
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, 4)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, 3)

    output_ch = 4
    skips = [4]
    dynamic_model = dyNeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, 
                   output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, 
                   use_viewdirs=args.use_viewdirs, num_basis=args.num_basis).to(device)
    grad_vars = list(dynamic_model.parameters())

    embed_fn_rigid, input_rigid_ch = get_embedder(args.multires, args.i_embed, 3)
    static_model = stNeRF(D=args.netdepth, W=args.netwidth, input_ch=input_rigid_ch, 
                          output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, 
                          use_viewdirs=args.use_viewdirs).to(device)
    grad_vars += list(static_model.parameters())

    if args.N_importance > 0:
        dynamic_model_fine = dyNeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, 
                                    output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, 
                                    use_viewdirs=args.use_viewdirs, num_basis=args.num_basis).to(device)
        grad_vars += list(dynamic_model_fine.parameters())
        static_model_fine = stNeRF(D=args.netdepth, W=args.netwidth, input_ch=input_rigid_ch, 
                                   output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, 
                                   use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(static_model_fine.parameters())
    
    trajectory_basis = torch.nn.Embedding(num_frames * args.deblur_images, args.num_basis)
    dct_basis = init_dct_basis(args.num_basis, num_frames * args.deblur_images)
    trajectory_basis.weight.data = torch.nn.Parameter(dct_basis)

    grad_vars_basis = list(trajectory_basis.parameters())
    optimizer_basis = torch.optim.Adam(params=grad_vars_basis, lr=args.lrate * 0.25)
    
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    dynamic_network_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                           embed_fn=embed_fn,
                                                                           embeddirs_fn=embeddirs_fn,
                                                                           netchunk=args.netchunk)

    static_network_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                          embed_fn=embed_fn_rigid,
                                                                          embeddirs_fn=embeddirs_fn,
                                                                          netchunk=args.netchunk)
    
    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) 
                 for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]

        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_se3.load_state_dict(ckpt['optimizer_se3_state_dict'])
        optimizer_basis.load_state_dict(ckpt['optimizer_basis_state_dict'])
        dynamic_model.load_state_dict(ckpt['dynamic_model_state_dict'])
        static_model.load_state_dict(ckpt['static_model_state_dict'])
        se3.load_state_dict(ckpt['se3_state_dict'])
        trajectory_basis.load_state_dict(ckpt['trajectory_basis_state_dict'])
        if args.N_importance > 0:
            print('Loading fine model.')
            dynamic_model_fine.load_state_dict(ckpt['dynamic_model_fine_state_dict'])
            static_model_fine.load_state_dict(ckpt['static_model_fine_state_dict'])
    
    render_kwargs_train = {
        'dynamic_network_fn' : dynamic_network_fn,
        'static_network_fn' : static_network_fn,
        'dynamic_model' : dynamic_model,
        'static_model' : static_model,
        'se3': se3, 
        'trajectory_basis': trajectory_basis, 
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'inference': False
    }
    if args.N_importance > 0:
        render_kwargs_train['dynamic_model_fine'] = dynamic_model_fine
        render_kwargs_train['static_model_fine'] = static_model_fine
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['inference'] = True

    return render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_se3, optimizer_basis



def create_eva_poses_model(args, poses_start_se3):
    
    # se3 = torch.nn.Embedding(poses_start_se3.shape[0], 6)
    se3 = torch.nn.Parameter(poses_start_se3, requires_grad=True)
    
    grad_vars_se3 = list(se3)
    optimizer_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.pose_lrate)

    return se3, optimizer_se3


class LearnPose(nn.Module):
    def __init__(self, poses_start_se3):
        
        super(LearnPose, self).__init__()
        self.poses_se3 = nn.Parameter(poses_start_se3, requires_grad=True)

    def forward(self, img_i):
        pose_se3 = self.poses_se3[img_i]
        
        return pose_se3.unsqueeze(0)