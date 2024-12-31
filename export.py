import torch.onnx
import onnx
from onnxsim import simplify
from models import CNN1DModel, CNN2DModel, LSTMModel, TransformerModel


best_model_path = './models/TransformerModel.17-accuracy0.99.pth'
# 加载最佳模型
model = TransformerModel()  # 'CNN1DModel, CNN2DModel, LSTMModel, TransformerModel'
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 根据模型实际输入来调整大小
target_frames=25
target_points=40
dummy_input = torch.randn(1, target_frames, target_points, 4, device='cpu')

# 导出模型
output_onnx_file = "./models/best_model.onnx"
torch.onnx.export(model,               # 模型
                  dummy_input,         # dummy input (or a tuple for multiple inputs)
                  output_onnx_file,   # 输出文件名
                  export_params=True,  # 导出模型参数
                  opset_version=12,    # ONNX version
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],   # 输入名
                  output_names=['output'], # 输出名
)
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # 批量大小可变
                #                 'output' : {0 : 'batch_size'}})

print(f"Model has been exported to {output_onnx_file}")


# 加载ONNX模型
onnx_model = onnx.load(output_onnx_file)

# 简化模型
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"

# 保存简化后的模型
onnx.save(model_simp, output_onnx_file)
print('finished exporting simplified onnx')