import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import time as timelib
import numpy as np
from queue import PriorityQueue
import time as timelib
from torch_cluster import fps

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def project_and_sort(xyz):
  num_points = xyz.shape[1]
  projected_values = np.sum(xyz, 2)
  projected_values_ = np.sort(projected_values)
  order = np.argsort(projected_values)
  #projected_values, order = projected_values.sort()
  return (projected_values_, order)


def binary_search(projected, left, right, query):
  middle = int((left+right)/2)
  if right < left:
    return (0, left)
  if query == projected[0,middle]:
    return (1, middle)
  elif query< projected[0,middle]:
    return binary_search(projected, left, middle-1, query)
  elif query> projected[0,middle]:
    return binary_search(projected, middle+1, right, query)

def find_middle_candidate(projected, left, right):
  query = (projected[0,left] + projected[0,right])/2
  suc, res = binary_search(projected, left, right, query)
  #print("the result fo r binary search of ", left, " to ", right, " is ", res)
  #print("left value is: ", projected[0,left], " right value is: ", projected[0,right], "the query is : ", query)
  if suc:
    #print("succefull binary search", left, right, res)
    return res, abs(projected[0,res] - projected[0,left])
  elif res == right + 1:
    return right, 0
  elif res == 0:
    return 0, 0
  #if abs(projected[res-1] - left) > abs(projected[res]- projected[right]):
  else:
    if abs(projected[0,res-1] - query) <= abs(projected[0,res]- projected[0,right]):
      return res - 1, abs(projected[0,res-1] - projected[0,left])
    else:
      return res, abs(projected[0,res] - projected[0,right])


def fps_(xyz, npoint):
    fps_start_time = timelib.time()
    B, N, C = xyz.shape
    pre_process_start_time = timelib.time()
    projected_values, order = project_and_sort(xyz)
    pre_process_time = timelib.time() - pre_process_start_time
    #print("Pre process time: ", pre_process_time)
    selected_points = list(np.expand_dims(np.random.randint(1, N-1), axis=0))
    #print("projected_values.shpe:", projected_values.shape)

    head_canidate_score = abs(projected_values[0, selected_points[0]] - projected_values[0, 0])
    tail_candidate_score = abs(projected_values[0, selected_points[0]] - projected_values[0, N-1])
    candidates = PriorityQueue() 
    candidates.put((-1 *head_canidate_score, 0, -2, selected_points[0]))
    candidates.put((-1 *tail_candidate_score, N-1, selected_points[0], -1))
    sum_loop_time = 0
    sum_find_middle_time = 0
    for i in range(npoint-1):
      find_middle_time = 0
      loop_start = timelib.time()
      _, next_selected, left_selected, right_selected = candidates.get()

      #selected_points = torch.cat((selected_points, torch.tensor([next_selected])), 0)
      selected_points.append(next_selected)
      # Adding the right-side candidate:
      if not (right_selected == -1 or right_selected==next_selected+1):
        find_middle_start_time = timelib.time()
        middle, score = find_middle_candidate(projected_values, next_selected, right_selected)
        find_middle_time += timelib.time() - find_middle_start_time
        

        #print(middle, " added as the right side candidate between ", next_selected, " and ", right_selected)
        candidates.put((-1 * score, middle, next_selected, right_selected))
      
      # Adding the left-side candidate:
      if not(left_selected == -2 or left_selected==next_selected-1):
        find_middle_start_time = timelib.time()
        middle, score = find_middle_candidate(projected_values, left_selected, next_selected)
        find_middle_time += timelib.time() - find_middle_start_time
        #print(middle, " added as the left side candidate between ", left_selected, " and ", next_selected)
        candidates.put((-1 * score, middle, left_selected, next_selected))
      loop_time = timelib.time() - loop_start
      sum_find_middle_time += find_middle_time
      sum_loop_time += loop_time
      #print("loop time: ", loop_time)
      
      
    loop_time_ave = sum_loop_time / npoint
    find_middle_time_ave = sum_find_middle_time / npoint
    #print("average loop time: ", loop_time_ave, "sum loop time: ", sum_loop_time)
    #print("average find middle time: ", find_middle_time_ave, "sum loop time: ", sum_find_middle_time)
    #print("--------------------------")
    centroids = np.zeros((1, npoint))
    #print("----------------")
    #print("Final result")
    #print("selected_points", selected_points)
    #print("The shapes: ")
    #print("centroids.shape", centroids.shape)
    #print("order.shape", order.shape)
    #print("selected_points.shape", selected_points.shape)
    #print("That was the shapes")
    centroids[0, 0:npoint] = order[0,selected_points]
    #print("centroids", centroids)
    #print("*********************************")
    # TODO (important): re-arrange the selected points by the order tensor
    fps_time = timelib.time() - fps_start_time
    #print("fps time: ", fps_time)
    #with open("/content/drive/MyDrive/Research/results/proposedFPSonGPU", 'a') as f:
      #f.write(str(fps_time))
      #f.write("\n")
    #print("fps time: ", fps_time, " for ", npoint , " points ")
    return centroids

def farthest_point_sample_from_IFPN(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    #num_pnts = 5
    #print("pcls", pcls)
    #print("pcls.shape", pcls.shape)
    #print("--------------------")
    #print(num_pnts)
    #print(type(pcls))
    #print(pcls.shape)
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    #print(sampled)
    #print(indices)
    #print(sampled.shape)
    #print(type(indices))
    #print(len(indices))
    #print(type(indices[0]).shape)
    #print("outputs: ")
    #print(sampled)
    #print(indices[0])
    #print("output: ", indices[0], type(indices[0]), indices[0].shape)
    result = indices[0].unsqueeze(0)
    return result
    return indices[0]
    return sampled, indices



def farthest_point_sample_orig_repo(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    print("output: ", centroids, type(centroids), centroids.shape)
    return centroids




def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # implementing the proposed FPS is desinged for batch_size=1 (for inference)
    #print("shape1: ", xyz.shape)
    #print("shape2: ",npoint)
    fps_start_time = timelib.time()
    device = xyz.device
    B, N, C = xyz.shape
    xyz = xyz.cpu().numpy()
    result = fps_(xyz, npoint)
    #print(result)
    
    #print(result.shape)
    #print("output --------------")
    #print(type(result))
    #print(result.shape)
    return result
    


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

