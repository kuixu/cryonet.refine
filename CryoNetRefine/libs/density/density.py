# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved
# version: 2.0
# Author: jsp
import os,math, time
import numpy as np
import torch


def mol_atom_density_np(atom_coords, atom_weight, res=3.0, voxel_size=1.0):
    """
    numpy version mol density by coords

    args:
        atom_coords: (N,3)
    """
    if isinstance(atom_coords, torch.Tensor):
        atom_coords = atom_coords.numpy()
    if isinstance(atom_weight, float):
        atom_weight = np.zeros(atom_coords.shape[0]) + atom_weight
    
    bmin = np.floor(atom_coords.min(axis=0)) 
    bmax = np.ceil(atom_coords.max(axis=0))  
    # transform coords into density space
    coords_den = atom_coords / voxel_size
    bcen = (bmax + bmin) / 2 / voxel_size
    
    # mol density
    step_size = 50
    grid_edg  = math.sqrt(-math.log(1e-7))
    gauss_radius = res / math.pi / voxel_size
    gbox = int(grid_edg * gauss_radius)
    grid_size = (gbox+1) * step_size
    gauss_grid = np.exp(-(np.arange(grid_size) / (gauss_radius * step_size))**2)

    def mol_atom(x, y, z):

        ii,jj,kk = [],[],[]
        for k in range(-gbox, gbox):
            ind = int(np.fabs(k - z) * step_size)
            # if ind >=300:
            #     import pdb;pdb.set_trace()
            kk.append(gauss_grid[ind])
        for j in range(-gbox, gbox):
            ind = int(np.fabs(j - y) * step_size)
            # if ind >=300:
            #     import pdb;pdb.set_trace()
            jj.append(gauss_grid[ind])
        for i in range(-gbox, gbox):
            ind = int(np.fabs(i - x) * step_size)
            # if ind >=300:
            #     import pdb;pdb.set_trace()
            ii.append(gauss_grid[ind])
        ade = np.einsum('i,j,k',ii,jj,kk)
        return ade

    boxsize = np.ceil((bmax - bmin) / voxel_size + 2 * gbox).astype(int)
    box0 = (bcen - boxsize / 2).astype(int)
    box1 = (bcen + boxsize / 2).astype(int)
    density = np.zeros((boxsize[0] , boxsize[1] , boxsize[2]))

    for s in range(0, coords_den.shape[0]):
        c = coords_den[s]
        aw = atom_weight[s]
        p0 = (c - gbox).astype(int) - box0
        p1 = (c + gbox).astype(int) - box0
        c  = np.round(c,8)
        atom_cube = aw * mol_atom(c[0]-int(c[0]), c[1]-int(c[1]) ,c[2]-int(c[2]))
        w,h,l = atom_cube.shape
        density[p0[0]:p0[0]+w, p0[1]:p0[1]+h, p0[2]:p0[2]+l] += atom_cube
    return density, box0

def mol_atom_density_th(atom_coords, atom_weight, res=3.0, voxel_size=1.0):
    """
    torch version mol density by coords

    args:
        atom_coords: (N,3)
    """
    # print(f"atom_coords.shape: {atom_coords.shape}")
    # print(f"atom_coords: {atom_coords}")
    if isinstance(atom_coords, np.ndarray):
        atom_coords = torch.from_numpy(atom_coords)
    if isinstance(atom_weight, float):
        atom_weight = torch.zeros(atom_coords.shape[0]) + atom_weight
    # print(atom_coords.shape, voxel_size, res)
    device = atom_coords.device
    bmin = torch.floor(atom_coords.amin(dim=0))  # Calculate minimum values along dimension 0
    bmax = torch.ceil(atom_coords.amax(dim=0))
    coords_den = atom_coords / voxel_size
    bcen = (bmax + bmin) / 2 / voxel_size
    
    # mol density
    step_size = 50
    grid_edg  = math.sqrt(-math.log(1e-7))
    gauss_radius = res / math.pi / voxel_size
    gbox = int(grid_edg * gauss_radius)
    grid_size = (gbox+1) * step_size
    gsphere = int(4 * gauss_radius) # todo
    
    # Compute boxsize and initialize density tensor
    # boxsize = torch.ceil((bmax - bmin) / voxel_size + 2 * gsphere).int()
    # Add 1 as buffer margin
    boxsize = torch.ceil((bmax - bmin) / voxel_size + 2 * gsphere+ 1).int()    
    box0 = torch.floor(bcen - (boxsize.float() / 2.0)).int()
    box1 = torch.floor(bcen + (boxsize.float() / 2.0)).int()
    density = torch.zeros((boxsize[0].item(), boxsize[1].item(), boxsize[2].item()), device=device)
    
    # Vectorized computation for all atoms
    N = coords_den.shape[0]
    if N == 0:
        raise RuntimeError(f"Warning: No atoms in mol_atom_density_th for {atom_coords.shape}")
    L = 2 * gsphere  # Number of grid points per axis in the Gaussian cube
    
    # Calculate fractional offsets for each atom's position
    # x_offsets = coords_den[:, 0] - coords_den[:, 0].int().float()
    # y_offsets = coords_den[:, 1] - coords_den[:, 1].int().float()
    # z_offsets = coords_den[:, 2] - coords_den[:, 2].int().float()
    # Modified: use rounding for integer part calculation
    rounded_coords_den = torch.round(coords_den)
    x_offsets = coords_den[:, 0] - rounded_coords_den[:, 0].float()
    y_offsets = coords_den[:, 1] - rounded_coords_den[:, 1].float()
    z_offsets = coords_den[:, 2] - rounded_coords_den[:, 2].float()

    # Modified: calculate p0_all using rounded integer parts
    p0_all = (rounded_coords_den -gsphere ).int() - box0
    # breakpoint()
    # Generate Gaussian distributions for each axis
    ra = torch.arange(-gsphere, gsphere, device=device).float()  # (L,)
    ra_expanded = ra.unsqueeze(0).expand(N, -1)  # (N, L)
    
    ii = torch.exp(-torch.pow(ra_expanded - x_offsets.unsqueeze(1), 2))  # (N, L)
    jj = torch.exp(-torch.pow(ra_expanded - y_offsets.unsqueeze(1), 2))  # (N, L)
    kk = torch.exp(-torch.pow(ra_expanded - z_offsets.unsqueeze(1), 2))  # (N, L)
    
    # Compute 3D Gaussian cubes for all atoms
    atom_cubes = torch.einsum('ni,nj,nk->nijk', ii, jj, kk)  # (N, L, L, L)
    atom_cubes *= atom_weight.view(-1, 1, 1, 1)  # Apply atom weights
    
    # Calculate global coordinates for each cube's grid points
    # p0_all = (coords_den - gsphere).int() - box0  # (N, 3)
    offsets = torch.stack(torch.meshgrid(
        torch.arange(L, device=device),
        torch.arange(L, device=device),
        torch.arange(L, device=device),
    indexing='ij')).to(device)  # (3, L, L, L)
    
    # Expand dimensions for broadcasting
    p0_all_expanded = p0_all.view(N, 3, 1, 1, 1)  # (N, 3, 1, 1, 1)
    offsets_expanded = offsets.unsqueeze(0)  # (1, 3, L, L, L)
    global_coords = p0_all_expanded + offsets_expanded  # (N, 3, L, L, L)
    
    # Create validity mask for coordinates
    boxsize_tensor = torch.tensor([boxsize[0], boxsize[1], boxsize[2]], device=device)
    # Use strict less-than to prevent floating point equality errors
    valid_x = (global_coords[:, 0] >= 0) & (global_coords[:, 0] < boxsize_tensor[0].float() - 1e-6)
    valid_y = (global_coords[:, 1] >= 0) & (global_coords[:, 1] < boxsize_tensor[1].float() - 1e-6)
    valid_z = (global_coords[:, 2] >= 0) & (global_coords[:, 2] < boxsize_tensor[2].float() - 1e-6)

    valid_mask = valid_x & valid_y & valid_z  # (N, L, L, L)
    # Apply mask to atom cubes
    atom_cubes *= valid_mask.float()  # Zero out invalid positions
    
    # Flatten indices and values for scatter addition
    x_indices = global_coords[:, 0, ...].reshape(-1).long()
    y_indices = global_coords[:, 1, ...].reshape(-1).long()
    z_indices = global_coords[:, 2, ...].reshape(-1).long()
    values = atom_cubes.reshape(-1)
    if x_indices.numel() == 0:
        raise RuntimeError(f"Warning: All atoms filtered out in mol_atom_density_th (N={N}, valid_mask.sum()={valid_mask.sum()})")
        return density, box0
    assert x_indices.max() < boxsize[0], f"X index overflow: {x_indices.max()} vs {boxsize[0]}"
    assert y_indices.max() < boxsize[1], f"Y index overflow: {y_indices.max()} vs {boxsize[1]}"
    assert z_indices.max() < boxsize[2], f"Z index overflow: {z_indices.max()} vs {boxsize[2]}"
    # Accumulate values into density tensor
    density.index_put_((x_indices, y_indices, z_indices), values, accumulate=True)
    return density, box0

def mol_atom_density(atom_coords, atom_weight, res=3.0, voxel_size=1.0, datatype="numpy"):
    if datatype=="numpy":
        return mol_atom_density_np(atom_coords, atom_weight, res, voxel_size)
    elif datatype=="torch":
        return mol_atom_density_th(atom_coords, atom_weight, res, voxel_size)
    else:
        raise "wrong datatype: "+datatype
class DensityInfo:
    def __init__(self, mrc_path=None, density=None, offset=(0.0, 0.0, 0.0), apix=1.0, voxel_size_tensor=None, voxel_size=1.0, ispg=1, parser="mrc",
                datatype="torch", device=torch.device("cpu"), resolution=None, stats=True, verbose=0):
        self.device = device
        self.datatype = datatype
        self.path = mrc_path
        self.parser = parser
        self.verbose = verbose
        self.voxel_size_tensor = voxel_size_tensor
        self.voxel_size = voxel_size
        self.resolution = resolution
        if mrc_path is None:
            self.set_density(density)
            self.set_offset(offset)
            self.set_apix(apix)
            self.set_ispg(ispg)
            self.set_voxel_size_tensor(voxel_size_tensor)

        else:
            den_info = self.__get_map_data__(mrc_path)
            self.set_density(den_info['density'])
            self.set_offset(den_info['offset'])
            self.set_apix(den_info['apix'])
            self.set_ispg(den_info['ispg'])
            self.set_voxel_size_tensor(den_info['voxel_size_tensor'])
        if stats:
            self.set_stats()
        self.shape   = self.density.shape

    def __get_map_data__(self, mrc_path):
        
        if mrc_path.endswith(".npz"):
            if os.path.exists(mrc_path):
                return self.__get_map_data_from_npz__(mrc_path)
            else:
                mrc_path = mrc_path.replace(".npz",".mrc")
        assert os.path.exists(mrc_path), mrc_path+" not found!"
        # parse map file
        if self.parser=="mrc":
            return self.__get_map_data_by_mrc__(mrc_path)
        
        #--------------------
        import mrcfile
        try:
            mapfile = mrcfile.open(mrc_path)
        except Exception as e:
            raise (mrc_path, " ERROR!!!!", e)

        dsize = mapfile.data.shape
        if len(dsize) != 3:
            print(mrc_path, "size error:", len(dsize))
            return
        x = int(mapfile.header["nxstart"])
        y = int(mapfile.header["nystart"])
        z = int(mapfile.header["nzstart"])
        c = mapfile.header.mapc - 1
        r = mapfile.header.mapr - 1
        s = mapfile.header.maps - 1
        crs = [c, r, s]
        x0, y0, z0 = list(mapfile.header["origin"].tolist())
        apix = mapfile.voxel_size.x

        # offset = [x + x0/apix, y + y0/apix, z + z0/apix]
        offset = [x*apix + x0, y*apix + y0, z*apix + z0]
        offset = (offset[crs[0]], offset[crs[1]], offset[crs[2]])
        
        cella = np.array(mapfile.header["cella"].tolist())

        m_data = mapfile.data.T.transpose(crs[0],crs[1],crs[2])
        return {'density': m_data, 'offset': offset, 'apix': apix, 'ispg': 1}

    def __get_map_data_from_npz__(self, npz_path):
        if self.verbose>0:
            print("reading map file from: ", npz_path)
        blob = np.load(npz_path, allow_pickle=True)
        # if 'data' in blob.keys():
        #     density_key = 'data'
        # else:
        density_key = 'density'
        # if 'im_info' in blob.keys():
        #     info     = blob['im_info']
        # elif 'densityinfo' in blob.keys():
        info     = blob['densityinfo']
        # else:
        #     info     = blob['info']
        apix = info[3]
        offset = info[6:9]
        # return {'density': blob[density_key].squeeze(), 'offset': offset, 'apix': apix, 'ispg': 1}
        return {'density': blob[density_key], 'offset': offset, 'apix': apix, 'ispg': 1}

    def __get_map_data_by_mrc__(self, mrc_path):
        from .mrc import MRC
        mapfile = MRC(mrc_path)
        return {'density': mapfile.data, 'offset': mapfile.offset, 'apix': mapfile.apix, 'ispg': 1, 'voxel_size_tensor': mapfile.voxel_size_tensor}

    def set_density(self, density):
        if self.datatype == "numpy":
            self.density = density
        else:
            if isinstance(density, torch.Tensor):
                self.density = density
            elif isinstance(density, np.ndarray):
                self.density = torch.from_numpy(density).to(self.device)
                # self.density = torch.from_numpy(deepcopy(density)).to(self.device)
            else:
                raise "error data type of density {}".format(type(density))
    
    def get_density(self):
        return self.density

    def to(self, device):
    
        return DensityInfo(density=self.density.to(device), offset=self.offset.to(device), apix=self.apix, voxel_size_tensor = self.voxel_size_tensor.to(device),voxel_size=self.voxel_size,device=device)

    def to_torch(self):
        self.datatype = "torch"
        self.density = torch.from_numpy(self.density).to(self.device)
        self.set_offset(self.offset)


    def to_dict(self):
        return dict(
            density=self.density,
            densityinfo=self.get_info(),
            apix=self.apix,
            offset=self.offset)

    def __repr__(self):
        """Print Density object as ."""
        s = self.size
        info = "<DensityInfo ({},{},{}) > \n".format(s[0].item(), s[1].item(), s[2].item(), ) \
             + "offset:  \t({:.4f},{:.4f},{:.4f})\n".format(self.offset[0], self.offset[1], self.offset[2]) \
             + "apix:    \t{:.4f}\n".format(self.apix) \
             + "voxel_size_tensor: \t({:.4f},{:.4f},{:.4f})\n".format(self.voxel_size_tensor[0], self.voxel_size_tensor[1], self.voxel_size_tensor[2]) \
             + "min:     \t{:.4f}\n".format(self.min) \
             + "max:     \t{:.4f}\n".format(self.max) \
             + "mean:    \t{:.4f}\n".format(self.mean) \
             + "mean1s:  \t{:.4f}\n".format(self.mean1s) \
             + "meanp:   \t{:.4f}\n".format(self.meanp) \
             + "meanp1s: \t{:.4f}\n".format(self.meanp1s) 
        return info

    def set_stats(self):
        self.mean  = self.density.mean()
        self.meanp = self.density[self.density>self.mean].mean()
        
        self.std   = self.density.std()
        self.stdp  = self.density[self.density>self.mean].std()

        self.mean1s = self.mean + self.std
        self.meanp1s = self.meanp + self.stdp

        self.min   = self.density.min()
        self.max   = self.density.max()

        self.size = torch.tensor(self.density.shape, device=self.device)

        self.shape   = self.density.shape
        # self.size    = self.density.shape


    def set_offset(self, offset):
        if self.datatype == "numpy":
            if isinstance(offset, tuple) or isinstance(offset, list):
                self.offset = np.array(offset)
            else:
                self.offset = offset
        else:
            if isinstance(offset, torch.Tensor):
                self.offset = offset
            elif isinstance(offset, np.ndarray) or isinstance(offset, tuple) or isinstance(offset, list):
                self.offset = torch.tensor(offset).to(self.device)
            else:
                raise "error data type of offset {}".format(type(offset))
                # self.offset=self.offset.to(torch.float32)

    def get_offset(self):
        return self.offset

    def set_apix(self, apix):
        self.apix = apix
        self.voxel_size = apix
        # self.voxel_size_tensor = (torch.ones_like(self.offset)*self.voxel_size)
    def set_voxel_size(self, voxel_size):
        self.voxel_size = voxel_size
    def set_voxel_size_tensor(self, voxel_size):
        self.voxel_size_tensor = voxel_size
    def get_voxel_size_tensor(self):
        return self.voxel_size_tensor
    def get_apix(self):
        return self.apix

    def set_ispg(self, ispg):
        self.ispg = ispg

    def get_ispg(self):
        return self.ispg

    def get_info(self):
        return np.array(
            [
                self.density.shape[0],
                self.density.shape[1],
                self.density.shape[2],
                self.apix,
                self.apix,
                self.apix,
                self.offset[0],
                self.offset[1],
                self.offset[2],
            ]
        )

    def is_empty(self, density):
        if np.isnan(density).any():
            return True
        elif density.max() == density.min() or np.isclose(density,0.0).all():
            return True
        else:
            return False
   
    
    def pad(self, cube_width, return_mask=False, no_padding=False, scale_factor=4):
        target_shape = list(self.density.shape)
        for i in range(len(target_shape)):
            if no_padding:
                target_shape[i] = target_shape[i] + (scale_factor - target_shape[i] % scale_factor)
            else:
                if target_shape[i] < cube_width:
                    target_shape[i] = cube_width

        value = self.density.min()
        cubes = self.density
        if isinstance(cubes, np.ndarray):
            m = np.array(target_shape) - cubes.shape
            mask = np.ones(target_shape)
            if m[0] > 0:
                b = np.zeros([m[0], cubes.shape[1], cubes.shape[2]]) + value
                mask[cubes.shape[0]:,:,:] = 0
                cubes = np.concatenate((cubes, b), axis=0)
            if m[1] > 0:
                b = np.zeros([cubes.shape[0], m[1], cubes.shape[2]]) + value
                mask[:,cubes.shape[1]:,:] = 0
                cubes = np.concatenate((cubes, b), axis=1)
            if m[2] > 0:
                b = np.zeros([cubes.shape[0], cubes.shape[1], m[2]]) + value
                mask[:,:,cubes.shape[2]:] = 0
                cubes = np.concatenate((cubes, b), axis=2)
        elif isinstance(cubes, torch.Tensor):
            m = (torch.tensor(target_shape) - torch.tensor(cubes.shape)).int()
            mask = torch.ones(target_shape)
            if m[0] > 0:
                b = torch.zeros([m[0], cubes.shape[1], cubes.shape[2]]) + value
                mask[cubes.shape[0]:,:,:] = 0
                cubes = torch.cat((cubes, b), dim=0)
            if m[1] > 0:
                b = torch.zeros([cubes.shape[0], m[1], cubes.shape[2]]) + value
                mask[:,cubes.shape[1]:,:] = 0
                cubes = torch.cat((cubes, b), dim=1)
            if m[2] > 0:
                b = torch.zeros([cubes.shape[0], cubes.shape[1], m[2]]) + value
                mask[:,:,cubes.shape[2]:] = 0
                cubes = torch.cat((cubes, b), dim=2)
        else:
            raise "wrong data type"
        if return_mask:
            return cubes, mask
        else:
            return cubes


    def overlap_right(self, mol_den):
        tgt_den = self
        device  = tgt_den.device
        tgt_off = (tgt_den.get_offset()/self.apix).round().int()# fix add by huangfuyao 9/2 .round
        mol_off = (mol_den.get_offset()/self.apix).round().int()
        t_s = tgt_den.shape
        m_s = mol_den.shape

        t_whl = torch.tensor(t_s, device=device).int() 
        m_whl = torch.tensor(m_s, device=device).int()

        t_box = torch.cat((tgt_off, tgt_off + t_whl)) 
        m_box = torch.cat((mol_off, mol_off + m_whl))
        
        bb     = torch.stack((t_box,m_box)) 
        bl,_ = bb[:,:3].max(dim=0) 
        br,_ = bb[:,3:].min(dim=0) 
        nwhl   = br - bl 

        t_o = bl - tgt_off
        m_o = bl - mol_off

        t_e = t_o + nwhl
        m_e = m_o + nwhl

        t_ov = torch.zeros_like(mol_den.density, device=device)
        # import pdb;pdb.set_trace()
        s0 = t_ov[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]].shape
        s1 = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]].shape
        if s0 == s1:
            t_ov[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]] = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] 
            m_ov = mol_den.density.reshape(-1)
            t_ov = t_ov.reshape(-1)[m_ov>0]
            m_ov = m_ov[m_ov>0]
            return t_ov.reshape(1,-1), m_ov.reshape(1,-1)
        else:
            
            print("warning!!! two different shape: ", s0, s1)
            return torch.zeros(2,2).reshape(1,-1), torch.ones(2,2).reshape(1,-1)


    def mask(self, msk_den, keep_origin=False):
        if self.datatype != 'torch':
            self.to_torch()
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset().int()
        msk_off = msk_den.get_offset().int()
        t_s = tgt_den.shape
        m_s = msk_den.shape

        t_whl = torch.tensor(t_s, device=device)
        m_whl = torch.tensor(m_s, device=device)

        bb     = torch.stack((tgt_off, msk_off))
        bmin,_ = bb[:,:3].max(dim=0)

        ww     = torch.stack((t_whl, m_whl))
        bmax,_ = (bb+ww)[:,:3].min(dim=0)
        nwhl   = bmax - bmin

        t_o =  bmin - tgt_off
        m_o =  bmin - msk_off

        t_e = t_o + nwhl 
        m_e = m_o + nwhl 
        
        tgt_region = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]]
        msk_region = msk_den.density[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]]

        s0 = tgt_region.shape
        s1 = msk_region.shape
        if s0 == s1:
            msk_density = tgt_region * msk_region
            if keep_origin:
                tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] = msk_density
                return tgt_den.density, tgt_off
            else:
                return msk_density, bmin
        else:
            return torch.empty(0), bmin
    
    def mask_box(self, box_min, box_max, keep_origin=False):
        if self.datatype != 'torch':
            self.to_torch()
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset().int()
        t_s = tgt_den.shape

        box_min = box_min.int()
        box_max = box_max.int()

        t_whl = torch.tensor(t_s, device=device)
        m_whl = torch.tensor(box_max-box_min, device=device)

        bb     = torch.stack((tgt_off, box_min))
        bmin,_ = bb[:,:3].max(dim=0)

        ww     = torch.stack((t_whl, m_whl))
        bmax,_ = (bb+ww)[:,:3].min(dim=0)
        nwhl   = bmax - bmin

        t_o =  bmin - tgt_off
        m_o =  bmin - box_min

        t_e = t_o + nwhl 
        m_e = m_o + nwhl 
        
        tgt_region = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]]
        if keep_origin:
            tgt_den.density = torch.zeros_like(tgt_den.density)
            tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] = tgt_region
            return tgt_den.density, tgt_off
        else:
            return tgt_region, bmin

    def int(self, var):
        if isinstance(var, torch.Tensor):
            return var.int()
        else:
            return var.astype(int)

    def save(self, map_path):
        self.path = map_path
        if self.parser =="mrc":
            self.save_mrc_new(map_path, self.density, self.offset, self.apix, self.ispg, self.voxel_size_tensor)
        else:
            self.save_mrc(map_path, self.density, self.offset, self.apix, self.ispg)
    
    @staticmethod
    def save_mrc_new(map_path, data, offset=(0.0, 0.0, 0.0), apix=1.0, ispg=1, voxel_size_tensor=None): 
        from .mrc import write
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                data= data.cpu()
            data = data.detach().numpy()
        if type(data[0,0,0])!=np.float32:
            data = data.astype(np.float32)
        
        if isinstance(offset, torch.Tensor):
            if offset.is_cuda:
                offset = offset.cpu()
            offset = offset.detach().numpy()
        if isinstance(apix, torch.Tensor):
            if apix.is_cuda:
                apix = apix.cpu()
            apix = apix.detach().numpy()
        if isinstance(voxel_size_tensor, torch.Tensor):
            if voxel_size_tensor.is_cuda:
                voxel_size_tensor = voxel_size_tensor.cpu()
            voxel_size_tensor = voxel_size_tensor.detach().numpy()

        write(map_path, data.T, header=None, Apix=apix, offset=offset, voxel_size_tensor = voxel_size_tensor)

    @staticmethod
    def save_mrc(map_path, data, offset=(0.0, 0.0, 0.0), apix=1.0, ispg=1): 
        
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                data= data.cpu()
            data = data.detach().numpy()
        if type(data[0,0,0])!=np.float32:
            data = data.astype(np.float32)
        
        if isinstance(offset, torch.Tensor):
            if offset.is_cuda:
                offset = offset.cpu()
            offset = offset.detach().numpy()
        if isinstance(apix, torch.Tensor):
            if apix.is_cuda:
                apix = apix.cpu()
            apix = apix.detach().numpy()[0]
        # print(voxel_size)
        # import pdb;pdb.set_trace()
        new_map = mrcfile.new(map_path, overwrite=True)
        new_map.set_data(data.T)
        new_map.header.nx = data.shape[0]
        new_map.header.ny = data.shape[1]
        new_map.header.nz = data.shape[2]
        new_map.header.nxstart  = offset[0]
        new_map.header.nystart  = offset[1]
        new_map.header.nzstart  = offset[2]
        new_map.header.mapc = 1
        new_map.header.mapr = 2
        new_map.header.maps = 3
        new_map.header.cella.x = data.shape[0] * apix
        new_map.header.cella.y = data.shape[1] * apix
        new_map.header.cella.z = data.shape[2] * apix
        new_map.header.mx = data.shape[0]
        new_map.header.my = data.shape[1]
        new_map.header.mz = data.shape[2]
        new_map.header.ispg = 1
        new_map.header.nversion = 20190801
        new_map.header.label[1] = 'by CryoNet, Author: Kui Xu, xukui.cs@gmail.com, Tsinghua University.'
        new_map.header.label[2] = "{:.6f}, {:.6f}, {:.6f}".format(offset[0], offset[1], offset[2])
        new_map.header.label[3] = "apix: {:.6f}".format(apix)
        new_map.header.label[4] = "MODIFIED: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        new_map.header.nlabl = 5
        new_map.close()
