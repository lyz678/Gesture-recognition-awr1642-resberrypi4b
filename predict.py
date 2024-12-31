import onnxruntime
import numpy as np
from load_and_process_data import process_npz_file
from show_npz import show_files_points



# 选择输入数据文件
data_file = './data/right/sample_3.npz'

# 定义类别列表
classes = ['ccw', 'cw', 'down', 'left', 'none', 'right', 's', 'up', 'x', 'z']

# 加载ONNX模型
onnx_model_path = "./models/best_model.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# 获取模型的输入和输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# 从文件名推断真实类别（假设文件夹名即为真实类别）
true_class = data_file.split('/')[-2]  # 提取文件夹名作为真实类别，例如 "right"

# 预处理输入数据
input_data = process_npz_file(data_file)
# 确保输入数据类型为float32
input_data = input_data.astype(np.float32)
# 增加 batch 维度
input_data = np.expand_dims(input_data, axis=0)

# 使用ONNX Runtime进行推理
outputs = session.run([output_name], {input_name: input_data})

# 获取推理结果的类别索引
pred_class_index = np.argmax(outputs[0])
pred_class = classes[pred_class_index]  # 根据索引获取预测类别

# 打印推理结果
print("真实类别 (True):", true_class)
print("预测类别 (Pred):", pred_class)

# 显示点云并在图上标注真实类别和预测类别
show_doppler = False
use_custom_range = True

# 修改 show_files_points 函数的显示逻辑
show_files_points(
    [data_file], 
    show_doppler, 
    use_custom_range,
    title=f"True: {true_class}, Pred: {pred_class}"  # 在图上显示标题
)
