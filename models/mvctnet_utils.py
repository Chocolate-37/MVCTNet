"""
MVCTNet Core Utilities
Contains point cloud processing, geometric feature computation,
sampling, grouping, and the RISP feature extraction pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from pointops.functions import pointops


def timeit(tag, t):
    """Simple timer utility. Prints elapsed time with a label."""
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    """
    Normalize a point cloud to zero-mean and unit-sphere scale.
    Args:
        pc: [N, 3] numpy array of point coordinates
    Returns:
        Normalized point cloud [N, 3]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid                                    # zero-center
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))         # farthest distance from centroid
    pc = pc / m                                           # scale to unit sphere
    return pc


def square_distance(src, dst):
    """
    Compute pairwise squared Euclidean distances between two point sets.
    Uses the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    Args:
        src: [B, N, C] source point cloud
        dst: [B, M, C] target point cloud
    Returns:
        dist: [B, N, M] matrix of squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Gather point features using an index array.
    Args:
        points: [B, N, C]  input point cloud / feature tensor
        idx:    [B, S] or [B, S, K]  index array
    Returns:
        new_points: [B, S, C] or [B, S, K, C]  indexed output
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (torch.arange(B, dtype=torch.long)
                     .to(device)
                     .view(view_shape)
                     .repeat(repeat_shape))
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS): iteratively select the point
    farthest from all already-selected points, producing a spatially
    uniform sub-sample.
    Args:
        xyz:    [B, N, 3] input point cloud
        npoint: target number of sampled points
    Returns:
        centroids: [B, npoint] indices of the selected points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10   # initialised to a large value
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # random start point
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # squared dist to current centre
        mask = dist < distance
        distance[mask] = dist[mask]                  # keep minimum distances
        farthest = torch.max(distance, -1)[1]        # pick the farthest point next
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query: for each centre point, find all neighbours within a radius.
    Returns exactly nsample indices; pads with the first valid neighbour if fewer found.
    Args:
        radius:  search radius
        nsample: max neighbours to return
        xyz:     [B, N, 3] all points
        new_xyz: [B, S, 3] query centre points
    Returns:
        group_idx: [B, S, nsample] neighbour indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (torch.arange(N, dtype=torch.long)
                 .to(device).view(1, 1, N).repeat([B, S, 1]))
    dists = torch.cdist(new_xyz, xyz)
    if radius is not None:
        group_idx[dists > radius] = N           # mark points outside radius
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]         # pad with first valid neighbour
    return group_idx


def compute_LRA_one(group_xyz, weighting=False):
    """
    Compute the Local Reference Axis (LRA) for grouped points via PCA
    (eigendecomposition of the scatter matrix). The LRA establishes a
    rotation-invariant local coordinate frame.
    Args:
        group_xyz: [B, S, N, C]  grouped point coordinates
        weighting: if True, apply inverse-distance weighting
    Returns:
        LRA: [B, S, 3] unit local reference axes
    """
    B, S, N, C = group_xyz.shape
    dists = torch.norm(group_xyz, dim=-1, keepdim=True)  # distances to the centre

    if weighting:
        # Closer points receive higher weight
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0                # replace NaN with 1
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)

    # Smallest eigenvector gives the most structurally stable axis
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    LRA = vec[:, :, :, 0]                               # eigenvector of smallest eigenvalue
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length                               # normalise to unit length
    return LRA


def compute_LRA(xyz, weighting=False, nsample=64):
    """Compute the LRA for every point using its k nearest neighbours."""
    dists = torch.cdist(xyz, xyz)
    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)             # convert to relative coords

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)

    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    LRA = vec[:, :, :, 0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA


def knn_point(nsample, xyz, new_xyz):
    """
    K-Nearest Neighbour query.
    Args:
        nsample: number of neighbours K
        xyz:     [B, N, C] candidate points
        new_xyz: [B, S, C] query points
    Returns:
        group_idx: [B, S, K] indices of K nearest neighbours
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample(npoint, xyz, norm=None, sampling='fps'):
    """
    Sample npoint points from a point cloud.
    Args:
        npoint:   target sample size
        xyz:      [B, N, 3] input coordinates
        norm:     [B, N, 3] surface normals (optional, sampled alongside xyz)
        sampling: 'fps' (farthest point) or 'random'
    Returns:
        new_xyz:  [B, npoint, 3] sampled coordinates
        new_norm: [B, npoint, 3] sampled normals (if norm was provided)
    """
    B, N, C = xyz.shape
    xyz = xyz.contiguous()
    if sampling == 'fps':
        fps_idx = pointops.furthestsampling(xyz, npoint).long()
        new_xyz = index_points(xyz, fps_idx)
        if norm is not None:
            new_norm = index_points(norm, fps_idx)
    elif sampling == 'random':
        shuffle = np.arange(xyz.shape[1])
        np.random.shuffle(shuffle)
        new_xyz = xyz[:, shuffle[:npoint], :]
        if norm is not None:
            new_norm = norm[:, shuffle[:npoint], :]
    else:
        print('Unknown sampling method: {}. Exiting.'.format(sampling))
        exit()
    return new_xyz, new_norm


def group_index(nsample, radius, xyz, new_xyz, group='knn'):
    """Return neighbour indices via KNN or ball-query grouping."""
    if group == 'knn':
        idx = knn_point(nsample, xyz, new_xyz.contiguous())
    elif group == 'ball':
        idx = pointops.ballquery(radius, nsample, xyz, new_xyz.contiguous())
        idx = idx.long()
    else:
        print('Unknown grouping method: {}. Exiting.'.format(group))
        exit()
    return idx


def order_index(xyz, new_xyz, new_norm, idx):
    """
    Sort neighbour indices by angular position around the local normal axis.
    This angular ordering is the key operation that gives RISP features their
    rotation invariance.
    Args:
        xyz:      [B, N, 3] all point coordinates
        new_xyz:  [B, S, 3] centre point coordinates
        new_norm: [B, S, 3] centre point normals
        idx:      [B, S, nsample] initial neighbour indices
    Returns:
        dots_sorted: sorted angular cosine values
        idx_ordered: [B, S, nsample] angularly sorted neighbour indices
    """
    epsilon = 1e-7
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # local (relative) coords

    # Project onto the tangent plane perpendicular to the surface normal
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane * new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / proj_xyz_length
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # replace NaN

    # Use the farthest projected point as the angular reference direction
    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(
        2, length_max_idx.unsqueeze(-1).repeat(1, 1, 1, 3))

    # Compute signed angular values and sort in descending order
    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(projected_xyz_unit,
                       vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1))
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.                      # reference point is always positive
    dots = sign * dots - (1 - sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))
    return dots_sorted, idx_ordered


def caculate_distance(xi):
    """Compute all pairwise Euclidean distances within a grouped neighbourhood."""
    B, N, num, C = xi.shape
    xi1 = torch.unsqueeze(xi, dim=2)   # [B, N, 1, num, C]
    xi2 = torch.unsqueeze(xi, dim=3)   # [B, N, num, 1, C]
    res_xi = xi1 - xi2
    e = 1e-16
    xi_distance = torch.sqrt(torch.sum(res_xi ** 2, dim=-1) + e)
    return xi_distance


def calculate_two_surface_feature(x1, x1_norm, x2, x2_norm):
    """
    Compute geometric features between two surface points:
    sine angles of both normals w.r.t. the connecting vector,
    and the Euclidean distance between them.
    These are the building blocks of 14-dim RISP descriptors.
    """
    epsilon = 1e-7
    offest = x1 - x2
    surface_offest_length = torch.norm(offest, dim=-1, keepdim=True)     # Euclidean distance
    offest_x12_unit = offest / (surface_offest_length + epsilon)          # unit direction
    offest_x12_unit[offest_x12_unit != offest_x12_unit] = 0              # replace NaN

    sin_angle_1 = -(offest_x12_unit * x1_norm).sum(-1, keepdim=True)    # normal1 x direction
    sin_angle_2 = (offest_x12_unit * x2_norm).sum(-1, keepdim=True)     # normal2 x direction
    return sin_angle_1, sin_angle_2, surface_offest_length


def calculate_unit(new_xi, x1):
    """Return the unit direction vector pointing from x1 to new_xi."""
    epsilon = 1e-7
    offest_xi = new_xi - x1
    surface_offest_length = torch.norm(offest_xi, dim=-1, keepdim=True)
    offest_xi_unit = offest_xi / (surface_offest_length + epsilon)
    offest_xi_unit[offest_xi_unit != offest_xi_unit] = 0
    return offest_xi_unit


def calculate_surface_norm(surface_norm1, surface_norm2):
    """Compute the cross product of two surface normal vectors."""
    norm_x = (surface_norm1[:, :, :, 1] * surface_norm2[:, :, :, 2]
              - surface_norm1[:, :, :, 2] * surface_norm2[:, :, :, 1])
    norm_y = (surface_norm1[:, :, :, 2] * surface_norm2[:, :, :, 0]
              - surface_norm1[:, :, :, 0] * surface_norm2[:, :, :, 2])
    norm_z = (surface_norm1[:, :, :, 0] * surface_norm2[:, :, :, 1]
              - surface_norm1[:, :, :, 1] * surface_norm2[:, :, :, 0])
    norm = torch.cat([norm_x.unsqueeze(-1),
                      norm_y.unsqueeze(-1),
                      norm_z.unsqueeze(-1)], dim=-1)
    return norm


def calculate_new_surface_feature(new_xi, new_xi_norm, x1, x1_norm,
                                   x2, x2_norm, x3, x3_norm):
    """
    Compute 5 higher-order geometric features from four surface points.
    These capture more complex local surface geometry to enrich the RISP descriptor.
    """
    pxi_u = calculate_unit(new_xi, x1)
    px2_u = calculate_unit(x2, x1)
    x2xi_u = calculate_unit(new_xi, x2)
    px3_u = calculate_unit(x3, x1)

    surface_norm1 = calculate_surface_norm(pxi_u, px2_u)
    surface_norm2 = calculate_surface_norm(px3_u, px2_u)

    sin_angle_1_1 = (pxi_u * px2_u).sum(-1, keepdim=True)
    sin_angle_1_2 = (pxi_u * x2xi_u).sum(-1, keepdim=True)
    sin_angle_3 = (surface_norm1 * surface_norm2).sum(-1, keepdim=True)
    sin_angle_2_1 = (x2xi_u * new_xi_norm).sum(-1, keepdim=True)
    sin_angle_2_2 = (px2_u * new_xi_norm).sum(-1, keepdim=True)

    new_feature = torch.cat([
        sin_angle_1_1, sin_angle_1_2,
        sin_angle_2_1, sin_angle_2_2,
        sin_angle_3
    ], dim=-1)
    return new_feature


def RISP_features(xyz, norm, new_xyz, new_norm, idx, group_all=False):
    """
    Compute 14-dim Rotation-Invariant Surface Point (RISP) features.
    This is the core geometric descriptor of MVCTNet.
    Feature breakdown:
      - 3 distance / length features
      - 6 basic angle features (normal vs. connecting-vector dot products)
      - 5 higher-order geometric features (cross-product based angles)

    Args:
        xyz:      [B, N, 3]         all point coordinates
        norm:     [B, N, 3]         all point normals
        new_xyz:  [B, S, 3]         centre point coordinates
        new_norm: [B, S, 3]         centre point normals
        idx:      [B, S, nsample]   neighbour indices
        group_all: use absolute coords (True for the global bottleneck layer)
    Returns:
        ri_feat:     [B, S, nsample, 14]  RISP geometric features
        idx_ordered: [B, S, nsample]      angularly sorted neighbour indices
    """
    B, N, C = new_xyz.shape
    num_neighbor = idx.shape[-1]

    # Angular ordering gives rotation-invariant neighbourhood representation
    dots_sorted, idx_ordered = order_index(xyz, new_xyz, new_norm.unsqueeze(-1), idx)

    grouped_center = index_points(xyz, idx_ordered)     # neighbour coordinates
    xi_norm = index_points(norm, idx_ordered)           # neighbour normals

    if not group_all:
        xi = grouped_center - new_xyz.view(B, N, 1, C) # local (relative) coordinates
    else:
        xi = grouped_center                             # absolute coords (global layer)

    p_point = torch.zeros_like(xi)                                        # centre = origin
    p_norm = (new_norm.unsqueeze(-2)).repeat([1, 1, num_neighbor, 1])     # broadcast normal

    # For denser point clouds, use a 2-step roll to capture a wider angular span
    num_shifts = 1
    if N >= 1024:
        num_shifts = 2

    x3 = torch.roll(xi, shifts=num_shifts, dims=2)           # previous neighbour xi-1
    x3_norm = torch.roll(xi_norm, shifts=num_shifts, dims=2)

    # 9 basic geometric features from three point pairs
    sin_angle_1_0, sin_angle_2_0, length_0 = calculate_two_surface_feature(
        p_point, p_norm, xi, xi_norm)        # centre -> xi
    sin_angle_1_1, sin_angle_2_1, length_1 = calculate_two_surface_feature(
        p_point, p_norm, x3, x3_norm)        # centre -> xi-1
    sin_angle_1_2, sin_angle_2_2, length_2 = calculate_two_surface_feature(
        xi, xi_norm, x3, x3_norm)            # xi -> xi-1

    angle_0 = (calculate_unit(p_point, xi) * calculate_unit(p_point, x3)).sum(-1, keepdim=True)
    angle_1 = (calculate_unit(x3, p_point) * calculate_unit(x3, xi)).sum(-1, keepdim=True)

    ri_feat = torch.cat([
        length_0,       # feature 1: distance centre->xi
        sin_angle_1_0,  # feature 2: normal1 angle
        sin_angle_2_0,  # feature 3: normal2 angle
        angle_0,        # feature 4: angle at centre
        sin_angle_1_1,  # feature 5
        sin_angle_2_1,  # feature 6
        angle_1,        # feature 7: angle at xi-1
        sin_angle_1_2,  # feature 8
        sin_angle_2_2,  # feature 9
    ], dim=-1)

    # 5 higher-order features using next neighbour xi+1
    x4 = torch.roll(xi, shifts=-num_shifts, dims=2)
    x4_norm = torch.roll(xi_norm, shifts=-num_shifts, dims=2)
    new_feature = calculate_new_surface_feature(
        x4, x4_norm, p_point, p_norm, xi, xi_norm, x3, x3_norm)
    ri_feat = torch.cat([ri_feat, new_feature], dim=-1)  # final 14-dim RISP

    return ri_feat, idx_ordered


def sample_and_group(npoint, radius, nsample, xyz, norm):
    """
    Full sample-and-group pipeline for one SA encoder layer:
      1. FPS sampling to select centre points
      2. KNN grouping to find neighbourhood
      3. RISP feature computation for each neighbourhood
    Args:
        npoint: number of centre points
        radius: neighbourhood radius parameter
        nsample: neighbours per centre point
        xyz:  [B, N, 3] input coordinates
        norm: [B, N, 3] input normals
    Returns:
        new_xyz:     [B, npoint, 3]             sampled centres
        ri_feat:     [B, npoint, nsample, 14]   RISP features
        new_norm:    [B, npoint, 3]             sampled normals
        idx_ordered: [B, npoint, nsample]       sorted neighbour indices
    """
    xyz = xyz.contiguous()
    norm = norm.contiguous()

    # Step 1: Farthest point sampling
    new_xyz, new_norm = sample(npoint, xyz, norm=norm, sampling='fps')

    # Step 2: KNN grouping
    idx = group_index(nsample, radius, xyz, new_xyz, group='knn')

    # Step 3: 14-dim RISP feature computation
    ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx)

    return new_xyz, ri_feat, new_norm, idx_ordered


def sample_and_group_all(xyz, norm):
    """
    Global grouping: all points form one single group.
    Used in the deepest encoder layer to capture global shape context.
    """
    device = xyz.device
    B, N, C = xyz.shape
    S = 1

    new_xyz = torch.mean(xyz, dim=1, keepdim=True)          # barycentre as centre
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)

    # Compute LRA from the global point set (distance-weighted)
    new_norm = compute_LRA_one(grouped_xyz_local, weighting=True)

    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx, group_all=True)
    return None, ri_feat, new_norm, idx_ordered

def sample_and_group_deconv(nsample, xyz, norm, new_xyz, new_norm):
    """
    Sample-and-group for decoder (feature propagation) layers.
    Builds neighbourhoods from the low-resolution source to the high-resolution target.
    """
    idx = group_index(nsample, 0.0, xyz, new_xyz, group='knn')
    ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx)
    return ri_feat, idx_ordered


# ---------------------------------------------------------------------------
# Self-Attention Layers
# ---------------------------------------------------------------------------

class SA_Layer(nn.Module):
    """
    1D Self-Attention over a set of N point features.
    Captures long-range context; applied after MaxPooling in each SA/FP layer.
    Input / Output shape: [B, C, N]
    """
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # Query
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # Key
        self.v_conv = nn.Conv1d(channels, channels, 1)                   # Value
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)   # [B, N, C//4]
        x_k = self.k_conv(x)                     # [B, C//4, N]
        x_v = self.v_conv(x)                     # [B, C, N]

        energy = x_q @ x_k                       # attention matrix [B, N, N]
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = x_v @ attention                    # weighted aggregation
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r                              # residual connection
        return x


class SA_Layer_2d(nn.Module):
    """
    2D Self-Attention over local neighbourhood features [B, C, N, K].
    Applied within each grouped neighbourhood before aggregation.
    """
    def __init__(self, channels):
        super(SA_Layer_2d, self).__init__()
        self.q_conv = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.trans_conv = nn.Conv2d(channels, channels, 1)
        self.after_norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 3, 2, 1)   # [B, K, N, C//4]
        x_k = self.k_conv(x).permute(0, 3, 1, 2)   # [B, K, C//4, N]
        x_v = self.v_conv(x).permute(0, 3, 1, 2)   # [B, K, C, N]

        energy = x_q @ x_k                          # [B, K, N, N]
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = x_v @ attention                       # [B, K, C, N]
        x_r = x_r.permute(0, 2, 3, 1)              # [B, C, N, K]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


# ---------------------------------------------------------------------------
# Encoder: Set Abstraction Layer
# ---------------------------------------------------------------------------

class MVCTNetSetAbstraction(nn.Module):
    """
    MVCTNet Set Abstraction (SA) Layer — core encoder block.
    Pipeline per forward pass:
      1. Sample + Group: FPS sampling then KNN grouping
      2. RISP embedding: 14-dim -> 32-dim -> 64-dim via Conv2d + BN
      3. 2D self-attention: enhance intra-neighbourhood features
      4. Feature concat: merge RISP with incoming skip features
      5. MVCTNet conv: project concatenated features to output_dim
      6. MaxPooling: aggregate neighbourhood features into per-point vector
      7. 1D self-attention: capture global context over downsampled points
    """

    def __init__(self, npoint, radius, nsample, in_channel, out_channel, group_all):
        super(MVCTNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # RISP embedding network: 14 -> 32 -> 64
        raw_in_channel = [14, 32]
        raw_out_channel = [32, 64]
        self.embedding = nn.Sequential(
            nn.Conv2d(raw_in_channel[0], raw_out_channel[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[0]),
            nn.Conv2d(raw_in_channel[1], raw_out_channel[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[1])
        )

        # Local 2D self-attention over neighbourhood
        self.self_attention_0 = SA_Layer_2d(raw_out_channel[1])

        # Main MVCTNet convolution: (RISP + skip) -> output_dim
        self.mvctnet = nn.Sequential(
            nn.Conv2d(raw_out_channel[1] + in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )

        # Global 1D self-attention over the downsampled points
        self.self_attention_1 = SA_Layer(out_channel)

    def forward(self, xyz, norm, points):
        """
        Args:
            xyz:    [B, N, 3]   coordinates
            norm:   [B, N, 3]   surface normals
            points: [B, C, N]   features from previous layer (None at first layer)
        Returns:
            new_xyz:    [B, npoint, 3]           downsampled coordinates
            new_norm:   [B, npoint, 3]           downsampled normals
            risur_feat: [B, out_channel, npoint] output features
        """
        if points is not None:
            points = points.permute(0, 2, 1)     # [B, C, N] -> [B, N, C]
        B, N, C = xyz.shape

        if self.group_all:
            new_xyz, ri_feat, new_norm, idx = sample_and_group_all(xyz, norm)
        else:
            new_xyz, ri_feat, new_norm, idx = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, norm)

        # Embed RISP: [B, 14, nsample, npoint] -> [B, 64, nsample, npoint]
        ri_feat = F.relu(self.embedding(ri_feat.permute(0, 3, 2, 1)))
        ri_feat = self.self_attention_0(ri_feat)

        # Concatenate with skip-connection features
        if points is not None:
            if idx is not None:
                grouped_points = index_points(points, idx)
            else:
                grouped_points = points.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        new_points = F.relu(self.mvctnet(new_points))
        risur_feat = torch.max(new_points, 2)[0]     # MaxPooling -> [B, C, npoint]
        risur_feat = self.self_attention_1(risur_feat)
        return new_xyz, new_norm, risur_feat


# ---------------------------------------------------------------------------
# Decoder: Feature Propagation Layer
# ---------------------------------------------------------------------------

class MVCTNetFeaturePropagation(nn.Module):
    """
    MVCTNet Feature Propagation (FP) Layer — core decoder block.
    Pipeline per forward pass:
      1. Deconv grouping: KNN from low-res source to high-res target
      2. RISP embedding + 2D self-attention
      3. Feature concat with grouped low-res features
      4. MVCTNet conv -> MaxPool -> 1D self-attention
      5. Merge with high-res skip connection; apply optional MLP refinement
    """

    def __init__(self, radius, nsample, in_channel, in_channel_2, out_channel, mlp):
        super(MVCTNetFeaturePropagation, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        raw_in_channel = [14, 32]
        raw_out_channel = [32, 64]
        self.embedding = nn.Sequential(
            nn.Conv2d(raw_in_channel[0], raw_out_channel[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[0]),
            nn.Conv2d(raw_in_channel[1], raw_out_channel[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[1])
        )
        self.self_attention_0 = SA_Layer_2d(raw_out_channel[1])
        self.mvctnet = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.self_attention_1 = SA_Layer(out_channel)

        # Optional MLP refinement applied after merging the skip connection
        last_channel = in_channel_2
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_channel = out_ch

    def forward(self, xyz1, xyz2, norm1, norm2, points1, points2):
        """
        Args:
            xyz1:    [B, N1, 3]  high-resolution target coordinates
            xyz2:    [B, N2, 3]  low-resolution source coordinates
            norm1:   [B, N1, 3]  high-resolution normals
            norm2:   [B, N2, 3]  low-resolution normals
            points1: [B, C1, N1] high-resolution skip-connection features
            points2: [B, C2, N2] low-resolution features from encoder
        Returns:
            new_points: [B, out_channel, N1] upsampled features at N1 resolution
        """
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape

        # Build upsampling neighbourhoods (low-res -> high-res)
        ri_feat, idx_ordered = sample_and_group_deconv(
            self.nsample, xyz2, norm2, xyz1, norm1)

        ri_feat = F.relu(self.embedding(ri_feat.permute(0, 3, 2, 1)))
        ri_feat = self.self_attention_0(ri_feat)

        if points2 is not None:
            if idx_ordered is not None:
                grouped_points = index_points(points2, idx_ordered)
            else:
                grouped_points = points2.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        new_points = F.relu(self.mvctnet(new_points))
        new_points = torch.max(new_points, 2)[0]
        new_points = self.self_attention_1(new_points)

        # Merge with high-res skip connection and apply MLP
        if points1 is not None:
            new_points = torch.cat([new_points, points1], dim=1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))

        return new_points


if __name__ == '__main__':
    # Performance benchmark: custom KNN vs pointops KNN
    nsample = 64
    ref = torch.rand(16, 100, 3).cuda()
    query = torch.rand(16, 20, 3).cuda()

    start = time()
    for i in range(50):
        idx = knn_point(nsample, ref, query.contiguous())
    print('Custom KNN time: {:.4f}s'.format(time() - start))

    start = time()
    for i in range(50):
        idx = pointops.knnquery(nsample, ref, query.contiguous())
    print('Pointops KNN time: {:.4f}s'.format(time() - start))
