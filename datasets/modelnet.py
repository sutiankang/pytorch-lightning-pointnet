import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape  
    xyz = point[:, :3] 
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10  
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest 
        centroid = xyz[farthest, :]  
        dist = np.sum((xyz - centroid) ** 2, -1)  
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pc_normalize(pc):
    """
        input: N x D
        output: N x D
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid  
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  
    pc = pc / m  # [-1, 1]
    return pc


class ModelNetDataset(Dataset):
    def __init__(self, cfg, split='train'):
        assert split in ['train', 'val', 'test']
        self.data_dir = cfg.data_dir
        self.split = split
        self.use_fps = cfg.use_fps
        self.num_points = cfg.num_points
        self.use_normals = cfg.use_normals
        self.use_cache = cfg.use_cache
        self.classes_map = {cfg.classes[i]: i for i in range(len(cfg.classes))}
        self.points = []
        self.labels = []

        if len(cfg.classes) == 10:
            self.data_files = {
                osp.join(self.data_dir, '_'.join(line.strip().split('_')[:-1]), line.strip() + '.txt'): '_'.join(line.strip().split('_')[:-1])
                for line in open(osp.join(self.data_dir, f'modelnet10_{self.split}.txt'))}
        else:
            self.data_files = {
                osp.join(self.data_dir, '_'.join(line.strip().split('_')[:-1]), line.strip() + '.txt'): '_'.join(line.strip().split('_')[:-1])
                for line in open(osp.join(self.data_dir, f'modelnet40_{self.split}.txt'))}

        if self.use_cache:
            if self.use_fps:
                cache_file = osp.join(self.data_dir, f'modelnet{len(cfg.classes)}_{self.split}_{self.num_points}pts_fps.pkl')
            else:
                cache_file = osp.join(self.data_dir, f'modelnet{len(cfg.classes)}_{self.split}_{self.num_points}pts.pkl')

            if not os.path.exists(cache_file):
                for data_file, class_name in tqdm(self.data_files.items(), desc=f'cache {self.split} dataset...'):
                    label = np.array(self.classes_map[class_name]).astype(np.int32)
                    points = np.loadtxt(data_file, delimiter=',').astype(np.float32)
                    if self.use_fps:
                        points = farthest_point_sample(points, self.num_points)
                    else:
                        points = points[::len(points)//self.num_points + 1, :]
                    self.points.append(points)
                    self.labels.append(label)
                with open(cache_file, 'wb') as f:
                    pickle.dump([self.points, self.labels], f)
            else:
                with open(cache_file, 'rb') as f:
                    self.points, self.labels = pickle.load(f)
        else:
            for data_file, class_name in tqdm(self.data_files.items(), desc=f'loading {self.split} dataset...'):
                label = np.array(self.classes_map[class_name]).astype(np.int32)
                points = np.loadtxt(data_file, delimiter=',').astype(np.float32)
                if self.use_fps:
                    points = farthest_point_sample(points, self.num_points)
                else:
                    points = points[::len(points)//self.num_points + 1, :]
                self.points.append(points)
                self.labels.append(label)

        assert len(self.points) == len(self.labels)

    def __getitem__(self, idx):
        point, label = self.points[idx], self.labels[idx]
        point[:, 0:3] = pc_normalize(point[:, 0:3])
        if not self.use_normals:
            point = point[:, 0:3]
        return point, label

    def __len__(self):
        return len(self.labels)
