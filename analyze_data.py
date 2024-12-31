import numpy as np
import os

def analyze_category_data(category_path):
    # 获取目录中的所有npz文件
    npz_files = [f for f in os.listdir(category_path) if f.endswith('.npz')]
    
    # 用于存储每个文件的统计信息
    frames_count_list = []  # 存储每个npz文件的帧数
    points_count_list = []  # 存储每个npz文件中每帧的点数
    
    # 用于存储每个类别中的各个维度数据
    all_x, all_y, all_z = [], [], []
    all_range, all_doppler, all_peak = [], [], []
    
    # 遍历每个 npz 文件
    for npz_file in npz_files:
        npz_path = os.path.join(category_path, npz_file)
        data = np.load(npz_path, allow_pickle=True)['data']

        # 获取数据的帧数
        frames_count = len(data)
        frames_count_list.append(frames_count)
        
        # 遍历每一帧数据
        for frame in data:
            if isinstance(frame, list) and len(frame) > 0:
                points_count = len(frame)  # 每帧中的点数
                points_count_list.append(points_count)
                
                # 统计每个点的 x, y, z, range, doppler, peak 的数据
                for point in frame:
                    all_x.append(point['x'])
                    all_y.append(point['y'])
                    all_z.append(point['z'])
                    all_range.append(point['range'])
                    all_doppler.append(point['doppler'])
                    all_peak.append(point['peak'])
    
    # 计算每个类别的均值
    x_mean = np.mean(all_x)
    y_mean = np.mean(all_y)
    z_mean = np.mean(all_z)
    range_mean = np.mean(all_range)
    doppler_mean = np.mean(all_doppler)
    peak_mean = np.mean(all_peak)
    
    # 计算每个类别中帧数和每帧点数的均值
    avg_frames_count = np.mean(frames_count_list)
    avg_points_count = np.mean(points_count_list)
    
    # 计算帧数和每帧点数的最大值和最小值
    max_frames_count = max(frames_count_list)
    min_frames_count = min(frames_count_list)
    
    max_points_count = max(points_count_list)
    min_points_count = min(points_count_list)
    
    # 打印结果
    print(f"Category: {category_path}")
    print(f"Mean values: x = {x_mean}, y = {y_mean}, z = {z_mean}, range = {range_mean}, "
          f"doppler = {doppler_mean}, peak = {peak_mean}")
    print(f"Average number of frames: {avg_frames_count}")
    print(f"Average number of points per frame: {avg_points_count}")
    print(f"Max/Min number of frames: Max = {max_frames_count}, Min = {min_frames_count}")
    print(f"Max/Min number of points per frame: Max = {max_points_count}, Min = {min_points_count}")
    
    return {
        'mean': {
            'x': x_mean,
            'y': y_mean,
            'z': z_mean,
            'range': range_mean,
            'doppler': doppler_mean,
            'peak': peak_mean
        },
        'avg_frames_count': avg_frames_count,
        'avg_points_count': avg_points_count,
        'max_frames_count': max_frames_count,
        'min_frames_count': min_frames_count,
        'max_points_count': max_points_count,
        'min_points_count': min_points_count
    }

# 遍历data目录下的所有分类
data_dir = './data'
categories = [os.path.join(data_dir, category) for category in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, category))]

category_stats = {}
for category in categories:
    category_stats[category] = analyze_category_data(category)

# 如果需要，可以将 category_stats 保存为一个文件
