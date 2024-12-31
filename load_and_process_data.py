import numpy as np
import os
from torch.utils.data import Dataset
import torch
from concurrent.futures import ThreadPoolExecutor

# 填充函数  
def pad_points(data, target_frames=25, target_points=40):
    data = [frame if frame is not None else [] for frame in data][:target_frames]
    padded_data = np.zeros((target_frames, target_points, 4))
    
    for i, frame in enumerate(data):
        points_count = min(len(frame), target_points)
        points = np.zeros((target_points, 4))
        for j in range(points_count):
            point = frame[j]
            if isinstance(point, dict):
                points[j] = [point.get('x', 0), point.get('y', 0), point.get('doppler', 0), point.get('range', 0)]
        padded_data[i, :points_count] = points[:points_count]
    
    return padded_data

# 单个文件处理函数  
def process_npz_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)['data']
    padded_data = pad_points(data)
    return padded_data

# 多线程加载与处理
def load_and_process_data(data_dir, max_workers=8):
    """
    使用多线程加载和处理数据
    :param data_dir: 数据文件夹路径
    :param max_workers: 最大线程数
    :return: 处理后的数据和标签
    """
    processed_data = []
    labels = []
    
    def process_file(npz_path, label_dir):
        points = process_npz_file(npz_path)
        return points, label_map[label_dir]

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".npz"):
                        npz_path = os.path.join(label_path, file)
                        # 提交任务到线程池
                        tasks.append(executor.submit(process_file, npz_path, label_dir))
        
        # 等待任务完成并收集结果
        for task in tasks:
            points, label = task.result()
            processed_data.append(points)
            labels.append(label)

    return np.array(processed_data), np.array(labels)

# 类别映射字典  
label_map = {
    'ccw': 0,
    'cw': 1,
    'down': 2,
    'left': 3,
    'none': 4,
    'right': 5,
    's': 6,
    'up': 7,
    'x': 8,
    'z': 9
}

class RadarDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def normalize(point_cloud):
    dimensions = [0, 1, 2, 3]
    normalized_point_cloud = point_cloud.copy()

    for dim in dimensions:
        nonzero_mask = normalized_point_cloud[:, :, dim] != 0
        if np.any(nonzero_mask):
            nonzero_values = normalized_point_cloud[nonzero_mask, dim]
            mean = np.mean(nonzero_values)
            std = np.std(nonzero_values)
            normalized_point_cloud[nonzero_mask, dim] = (nonzero_values - mean) / (std + 1e-6)

    return normalized_point_cloud

def main():
    data_dir = './data'
    processed_data, labels = load_and_process_data(data_dir, max_workers=8)
    print("Processed data shape:", processed_data.shape)

if __name__ == "__main__":
    main()


