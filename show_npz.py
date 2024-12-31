import numpy as np
import matplotlib.pyplot as plt
import os
import math
def main():
    # 设置根目录
    base_dir = './data/right/'

    # 文件名列表
    file_names = ['sample_1.npz', 'sample_100.npz', 'sample_1000.npz', 'sample_2000.npz']

    # 使用根目录和文件名动态生成文件路径
    file_paths = [os.path.join(base_dir, file_name) for file_name in file_names]

    # 提供自定义坐标范围的选项
    use_custom_range = True  # 设置为True来启用自定义范围，否则为False


    # 控制是否显示 Doppler 信息
    show_doppler = False  # 设置为 True 来显示 Doppler 信息，False 来隐藏 Doppler 信息
    show_files_points(file_paths, show_doppler, use_custom_range, None)



def show_files_points(file_paths, show_doppler:False, use_custom_range:False, title:None):
    # 创建一个图形
    plt.figure(figsize=(12, 10))
    # 初始化全局范围变量
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    # 存储所有点的数据
    all_points_data = []
    # 存储每个文件中的frame的颜色
    frame_colors = []
    custom_x_min, custom_x_max = -1, 1
    custom_y_min, custom_y_max = 0, 1.5

    # 计算子图行列数，确保布局能容纳所有文件
    n = len(file_paths)
    rows = int(math.ceil(n ** 0.5))  # 计算行数，使用平方根的上取整
    cols = int(math.ceil(n / rows))  # 计算列数
    # 遍历每个文件路径并同时处理数据和颜色
    for i, file_path in enumerate(file_paths):
        # 加载数据
        data = np.load(file_path, allow_pickle=True)['data']
        print(f"Loaded data from {file_path}: {data}")
        
        # 合并所有帧的点云数据
        all_points = []
        for frame in data:
            if frame is not None:
                all_points.extend(frame)  # 合并每帧的所有点

        all_points_data.append(all_points)

        # 提取 x 和 y 坐标
        x_coords = [point['x'] for point in all_points]
        y_coords = [point['y'] for point in all_points]

        # 更新坐标范围（如果没有设置自定义范围）
        if not use_custom_range:
            x_min = min(x_min, min(x_coords))
            x_max = max(x_max, max(x_coords))
            y_min = min(y_min, min(y_coords))
            y_max = max(y_max, max(y_coords))

        # 为每个文件中的每个frame分配灰度颜色
        frame_colors.append(plt.cm.Greys(np.linspace(0.2, 0.8, len(data))))  # 灰度从浅到深

        # 绘制散点图
        plt.subplot(rows, cols, i + 1)  # 创建 n x n 子图布局  
        
        # 每个file中的frame使用不同的灰度颜色
        for j, frame in enumerate(data):
            if frame is not None:
                # 提取x, y坐标并设置颜色
                frame_x = [point['x'] for point in frame]
                frame_y = [point['y'] for point in frame]
                frame_doppler = [point['doppler'] for point in frame]  # 获取 Doppler 信息

                # 绘制散点
                plt.scatter(frame_x, frame_y, alpha=0.5, s=10, color=frame_colors[i][j], label=f"Frame {j+1}" if j == 0 else "")

                if show_doppler:
                    # 计算该 frame 的平均 Doppler 值
                    avg_doppler = np.mean(frame_doppler)

                    # 在该 frame 的中心显示 Doppler 信息
                    center_x = np.mean(frame_x)
                    center_y = np.mean(frame_y)
                    
                    # 显示 Doppler 信息
                    plt.text(center_x, center_y, f"{avg_doppler:.2f}", fontsize=8, color='red', ha='center', va='center')

        plt.title(f"Point Cloud - Sample {i+1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # 如果使用自定义范围，直接使用自定义范围
        if use_custom_range:
            plt.xlim(custom_x_min, custom_x_max)
            plt.ylim(custom_y_min, custom_y_max)
        else:
            plt.xlim(x_min, x_max)  # 设置统一的 x 坐标范围
            plt.ylim(y_min, y_max)  # 设置统一的 y 坐标范围

        plt.grid(True)
        plt.legend()

    # 调整布局，确保子图不重叠
    if(title):
        plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()