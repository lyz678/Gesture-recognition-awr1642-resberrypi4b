import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, random_split
from models import CNN1DModel, CNN2DModel, TransformerModel
from load_and_process_data import load_and_process_data, RadarDataset
from show_npz import show_files_points
from sklearn.metrics import accuracy_score


# 设置数据路径
data_path = './data'  # 或者你的具体路径
model_path = './models/best_model.pth'  # 模型路径

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



# 加载数据
data, labels = load_and_process_data(data_path, max_workers=8)

# 创建数据集和数据加载器
dataset = RadarDataset(data, labels)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 选择模型
model = TransformerModel()  # 可以根据需要选择 CNN1DModel, CNN2DModel, TransformerModel

# 加载最佳模型
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置模型为评估模式

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 预测和计算混淆矩阵
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)

# 打印准确率
print(f'Accuracy: {accuracy:.2f}')

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

