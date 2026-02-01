"""
Geometry utility functions and data loaders for protein structure analysis.
Consolidated from utils.py and rama_utils.py for better maintainability.
"""
import os
import math
import pickle
import torch
from .n_dim_table import NDimTable


# ============================================================================
# Constants and Mappings
# ============================================================================

# Ramachandran constants
RAMALYZE_OUTLIER = 1
RAMALYZE_ALLOWED = 0
RAMALYZE_FAVORED = 0
RAMALYZE_ANY = 3
RAMALYZE_NOT_FAVORED = 4

# Secondary structure types
SS_TYPE = ['L', 'H', 'S']
ss2idx = {"L": 0, "H": 1, "S": 2}

# Ramachandran residue names
RAMA_RESNAME = [
    'prePRO', 'ILE', 'GLN', 'GLY', 'GLU', 'CYS', 'ASP', 'SER', 'LYS', 'ASN',
    'transPRO', 'VAL', 'THR', 'cisPRO', 'HIS', 'TRP', 'PHE', 'ALA', 'MET', 'LEU', 'ARG', 'TYR'
]
rama_res2idx = {r: i for i, r in enumerate(RAMA_RESNAME)}

# Amino acid to rotamer file name mapping
aa2fname = {
    'arg': 'arg', 'asn': 'asn', 'asp': 'asp', 'cys': 'cys', 'gln': 'gln',
    'glu': 'glu', 'his': 'his', 'ile': 'ile', 'leu': 'leu', 'lys': 'lys',
    'met': 'met', 'phe': 'phetyr', 'pro': 'pro', 'ser': 'ser', 'thr': 'thr',
    'trp': 'trp', 'tyr': 'phetyr', 'val': 'val',
}

# Secondary structure calibration values for Z-scores
SS_CALIBRATION_VALUES = {
    "H": (-0.045355950779513175, 0.1951165524439217),
    "S": (-0.0425581278436754, 0.20068584887814633),
    "L": (-0.018457764754231075, 0.15788374669456848),
    "W": (-0.016806654295023003, 0.12044960331869274),
}


# ============================================================================
# Data Loading
# ============================================================================

def _get_geometry_data_dir():
    """Get the path to the geometry data directory."""
    return os.path.dirname(os.path.abspath(__file__))


def load_rotamer_tables():
    """Load rotamer lookup tables for all amino acids."""
    tables = {}
    data_dir = os.path.join(_get_geometry_data_dir(), "data")
    for aa, fname in aa2fname.items():
        filepath = os.path.join(data_dir, f"rota8000-{fname}.data")
        tables[aa] = NDimTable.createFromText(filepath)
    return tables


def load_rama_tables():
    """Load Ramachandran lookup tables."""
    rama_files = [
        "rama8000-general-noGPIVpreP.data",
        "rama8000-gly-sym.data",
        "rama8000-cispro.data",
        "rama8000-transpro.data",
        "rama8000-prepro-noGP.data",
        "rama8000-ileval-nopreP.data"
    ]
    tables = []
    data_dir = os.path.join(_get_geometry_data_dir(), 'data')
    for rama_file in rama_files:
        filepath = os.path.join(data_dir, rama_file)
        tables.append(NDimTable.createFromText(filepath))
    return tables


def load_ramaz_db():
    """Load Ramachandran Z-score database."""
    data_dir = _get_geometry_data_dir()
    db_path = os.path.join(data_dir, 'data', 'top8000_rama_z_dict.pkl')
    with open(db_path, 'rb') as rf:
        return pickle.load(rf)


# Initialize global data tables (loaded once at module import)
aaTables = load_rotamer_tables()
rama_tables = load_rama_tables()
ramaz_db = load_ramaz_db()


# ============================================================================
# Angle and Dihedral Calculations
# ============================================================================

def calc_dihedral(four_atom_sites):
    """
    Calculate dihedral angle given four atom positions.
    
    Args:
        four_atom_sites: List or tuple of 4 3D coordinates
        
    Returns:
        Dihedral angle in degrees
    """
    v1, v2, v3, v4 = four_atom_sites
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    cb_norm = cb / (torch.norm(cb) + 1e-8)

    v = ab - torch.dot(ab, cb_norm) * cb_norm
    w = db - torch.dot(db, cb_norm) * cb_norm

    x = torch.dot(v, w)
    y = torch.dot(torch.linalg.cross(cb_norm, v), w)
    angle = torch.atan2(y, x) * (180.0 / math.pi)
    return angle


def calc_dihedral_batch(four_atom_sites_batch):
    """
    Batch calculation of dihedral angles.
    
    Args:
        four_atom_sites_batch: (N,4,3) or (4,3) tensor
        
    Returns:
        (N,) tensor or scalar of angles in radians
    """
    if isinstance(four_atom_sites_batch, list):
        four_atom_sites_batch = torch.stack(four_atom_sites_batch, dim=0)

    if four_atom_sites_batch.ndim == 2 and four_atom_sites_batch.shape[0] == 4:
        # Single case → expand to (1,4,3)
        four_atom_sites_batch = four_atom_sites_batch.unsqueeze(0)

    v1, v2, v3, v4 = (
        four_atom_sites_batch[:, 0],
        four_atom_sites_batch[:, 1],
        four_atom_sites_batch[:, 2],
        four_atom_sites_batch[:, 3]
    )

    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3

    cb_norm_val = torch.norm(cb, dim=1, keepdim=True)
    denom = torch.clamp(cb_norm_val, min=1e-6) + 1e-8
    cb_norm = cb / denom

    v = ab - torch.sum(ab * cb_norm, dim=1, keepdim=True) * cb_norm
    w = db - torch.sum(db * cb_norm, dim=1, keepdim=True) * cb_norm

    x = torch.sum(v * w, dim=1)
    y = torch.sum(torch.cross(cb_norm, v, dim=1) * w, dim=1)

    eps_a = 1e-8
    x_safe = torch.where(x.abs() < eps_a, torch.full_like(x, eps_a), x)
    y_safe = torch.where(y.abs() < eps_a, torch.zeros_like(y), y)
    angle = torch.atan2(y_safe, x_safe)

    if angle.shape[0] == 1:
        return angle[0]  # scalar
    return angle


def calc_dihedrals(atom_pos):
    """
    Calculate phi and psi angles for a protein given its atom positions.
    
    Args:
        atom_pos: (L, 14, 3) tensor of atom positions
        
    Returns:
        (L-2, 2) tensor of [phi, psi] angles in radians
    """
    prot_len = atom_pos.shape[0]
    
    if prot_len < 3:
        return torch.zeros((0, 2), device=atom_pos.device, dtype=atom_pos.dtype)
    
    # Batch extraction of atom positions to avoid for-loops
    # phi: C[i], N[i+1], CA[i+1], C[i+1]
    phi_atoms = torch.stack([
        atom_pos[:-2, 2],    # C[i]
        atom_pos[1:-1, 0],   # N[i+1]
        atom_pos[1:-1, 1],   # CA[i+1]
        atom_pos[1:-1, 2]    # C[i+1]
    ], dim=1)  # shape: (prot_len-2, 4, 3)
    
    # psi: N[i+1], CA[i+1], C[i+1], N[i+2]
    psi_atoms = torch.stack([
        atom_pos[1:-1, 0],   # N[i+1]
        atom_pos[1:-1, 1],   # CA[i+1]
        atom_pos[1:-1, 2],   # C[i+1]
        atom_pos[2:, 0]      # N[i+2]
    ], dim=1)  # shape: (prot_len-2, 4, 3)
    
    phis = calc_dihedral_batch(phi_atoms)
    psis = calc_dihedral_batch(psi_atoms)
    return torch.stack((phis, psis), dim=-1)


# ============================================================================
# Chirality and Structure Building
# ============================================================================

def volume_model(ca, n, cb, c):
    """Calculate chirality volume."""
    d_01 = n - ca
    d_02 = cb - ca
    d_03 = c - ca
    d_02_cross_d_03 = torch.linalg.cross(d_02, d_03)
    volume = torch.dot(d_01, d_02_cross_d_03)
    return volume


def chiralty(atom_pos):
    """
    Calculate chirality volume for all residues.
    
    Args:
        atom_pos: (L, 14, 3) tensor of atom positions
        
    Returns:
        (L,) tensor of chirality volumes
    """
    prot_len = int(atom_pos.shape[0])
    chiralty_volume = torch.zeros(prot_len)
    for i in range(prot_len):
        # ca, n, cb, c
        c = volume_model(atom_pos[i][1], atom_pos[i][0], atom_pos[i][4], atom_pos[i][2])
        chiralty_volume[i] = c
    return chiralty_volume


def cross(a, b):
    """Cross product of two 3D vectors."""
    assert a.shape == b.shape
    device = a.device
    return torch.tensor([
        a[1] * b[2] - b[1] * a[2],
        a[2] * b[0] - b[2] * a[0],
        a[0] * b[1] - b[0] * a[1]
    ], device=device)


def rotate_point_around_axis(axis_point_1, axis_point_2, point, angle, deg=False):
    """
    Rotate a 3D coordinate about a given arbitrary axis by the specified angle.
    
    Args:
        axis_point_1: Tensor representing 3D coordinate at one end of the axis
        axis_point_2: Tensor representing 3D coordinate at other end of the axis
        point: Tensor representing 3D coordinate of starting point to rotate
        angle: Rotation angle (defaults to radians)
        deg: Boolean (default=False), specifies whether the angle is in degrees
        
    Returns:
        Tensor of rotated point
    """
    device = axis_point_1.device
    
    if deg:
        angle *= math.pi / 180.
    a = axis_point_1
    b = axis_point_2
    l = b - a
    lsq = l ** 2

    dlsq = lsq.sum()
    dl = dlsq ** 0.5
    ca = math.cos(angle)
    dsa = math.sin(angle) / dl
    oca = (1 - ca) / dlsq
    lo = torch.tensor([l[0]*l[1]*oca, l[0]*l[2]*oca, l[1]*l[2]*oca], device=device)

    ma = point - a
    m = torch.tensor([
        [lsq[0]*oca+ca, lo[0]-l[2]*dsa, lo[1]+l[1]*dsa],
        [lo[0]+l[2]*dsa, lsq[1]*oca+ca, lo[2]-l[0]*dsa],
        [lo[1]-l[1]*dsa, lo[2]+l[0]*dsa, lsq[2]*oca+ca]
    ], device=device)

    ma = torch.unsqueeze(ma, -1)
    xyz_new = torch.matmul(m, ma)
    xyz_new = torch.squeeze(xyz_new)
    xyz_new = xyz_new + a

    return xyz_new


def construct_fourth(resN, resCA, resC, dist, angle, dihedral, method="NCAB"):
    """
    Construct fourth atom position (e.g., C-beta) from three backbone atoms.
    
    Args:
        resN: N atom coordinate
        resCA: CA atom coordinate
        resC: C atom coordinate
        dist: Distance from CA to the constructed atom
        angle: Angle at CA
        dihedral: Dihedral angle
        method: "NCAB" or "CNAB"
        
    Returns:
        3D coordinate of the constructed atom
    """
    device = resN.device
    
    if method == "NCAB":
        res0, res1, res2 = resN, resC, resCA
    elif method == "CNAB":
        res0, res1, res2 = resC, resN, resCA
    else:
        raise ValueError("method must be NCAB or CNAB")
        
    a = res2 - res1
    b = res0 - res1
    c = cross(a, b)
    cmag = torch.sqrt(torch.sum(c**2))
    if cmag > 0.000001:
        c *= dist / cmag
    c += res2
    d = c
    angledhdrl = dihedral - torch.tensor(90.0, device=device)
    a = res1
    b = res2
    
    newD = rotate_point_around_axis(
        axis_point_1=res1, axis_point_2=res2, point=d, angle=angledhdrl, deg=True
    )
    a = newD - res2
    b = res1 - res2
    c = cross(a, b)
    cmag = torch.sqrt(torch.sum(c**2))
    if cmag > 0.000001:
        c *= dist / cmag
    angledhdrl = torch.tensor(90.0, device=device) - angle
    a = res2
    c += a
    b = c
    
    if torch.allclose(a, b):
        return newD
    return rotate_point_around_axis(
        axis_point_1=a, axis_point_2=b, point=newD, angle=angledhdrl, deg=True
    )


def construct_fourth_batch(resN, resCA, resC, dist, angle, dihedral, method="NCAB"):
    """
    Batch computation of the fourth atom coordinates (e.g., theoretical C-beta positions).
    
    Args:
        resN: Coordinates of N atoms, shape (N, 3)
        resCA: Coordinates of CA atoms, shape (N, 3)
        resC: Coordinates of C atoms, shape (N, 3)
        dist: Distance from CA to the constructed atom, shape (N,)
        angle: Angle at CA (in radians), shape (N,)
        dihedral: Dihedral angle (in radians), shape (N,)
        method: "NCAB" or "CNAB". Defines calculation order
        
    Returns:
        Constructed fourth atom coordinates, shape (N, 3)
    """
    if method == "NCAB":
        res0, res1, res2 = resN, resC, resCA
    elif method == "CNAB":
        res0, res1, res2 = resC, resN, resCA
    else:
        raise ValueError("method must be NCAB or CNAB")

    # a = res2 - res1, b = res0 - res1 (bond vectors)
    a = res2 - res1
    b = res0 - res1

    # Build local coordinate system
    e1 = a / torch.norm(a, dim=-1, keepdim=True)  # unit vector along CA->CA neighbor
    n = torch.cross(a, b, dim=-1)                 # plane normal
    n = n / torch.norm(n, dim=-1, keepdim=True)   # normalized normal vector
    e2 = torch.cross(n, e1, dim=-1)               # third axis orthogonal to both

    # Direction vector for fourth atom placement
    # x: along -e1 (bond direction), y: in e2 (angular deviation), z: in n (dihedral)
    x = torch.cos(angle).unsqueeze(-1) * (-e1)
    y = torch.sin(angle).unsqueeze(-1) * torch.cos(dihedral).unsqueeze(-1) * e2
    z = torch.sin(angle).unsqueeze(-1) * torch.sin(dihedral).unsqueeze(-1) * n
    direction = x + y + z

    return res2 + dist.unsqueeze(-1) * direction


# ============================================================================
# Interpolation Functions
# ============================================================================

def interpolate(x, p1x, p2x, p1y, p2y):
    """Linear interpolation between two points."""
    dx = p2x - p1x
    dy = p2y - p1y
    return p1y + dy * (x - p1x) / dx


def interpolate_2d(p1x, p1y, p2x, p2y, v1, v2, v3, v4, xy):
    """Bilinear interpolation in 2D."""
    v14 = interpolate(xy[0], p1x, p2x, v1, v4)
    v32 = interpolate(xy[0], p1x, p2x, v3, v2)
    result = interpolate(xy[1], p1y, p2y, v14, v32)
    return result
