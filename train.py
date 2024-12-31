import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import CNN1DModel, CNN2DModel, LSTMModel, TransformerModel
from load_and_process_data import load_and_process_data, RadarDataset


# 设置数据路径
data_path = './data'  # 或者你的具体路径

classes = ['ccw','cw','down','left','none','right','s','up','x','z']

# parameters
num_epochs = 40
validation_split = 0.2
num_workers = 8
patience = 5
base_path = 'models/'
log_path = os.path.join(base_path, 'logging.txt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model_path = os.path.join(base_path, 'best_model.pth')



# Configure logging
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
parser.add_argument('--model', type=str, default='TransformerModel', help='CNN1DModel, CNN2DModel, LSTMModel, TransformerModel')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--pretrained', default=False, type=bool, help='True or False')
opt = parser.parse_args()


# 假设 load_and_process_data 返回的数据和标签
data, labels = load_and_process_data(data_path)

# 创建数据集和数据加载器
dataset = RadarDataset(data, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 使用 random_split 来分割数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.bs, shuffle=False)




# 选择模型
if opt.model == 'CNN1DModel':
    model = CNN1DModel()
elif opt.model == 'CNN2DModel':
    model = CNN2DModel()
elif opt.model == 'LSTMModel':
    model = LSTMModel() 
elif opt.model == 'TransformerModel':
    model = TransformerModel()

# 将模型移到设备上
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=int(patience / 4))

# 创建 SummaryWriter
writer = SummaryWriter(log_dir='runs')  # 指定 TensorBoard 日志存储路径

# training and validation loop
best_val_loss = float('inf')
early_stop_counter = 0

logger.info(f'Training started with model: {opt.model}')

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for images, labels in train_progress:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_progress.set_postfix({
            'loss': loss.item(),
            'train_accuracy': train_correct / train_total
        })
    
    train_loss /= train_total
    train_accuracy = train_correct / train_total

    # 在 TensorBoard 上记录训练损失和准确率
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_progress = tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}', unit='batch')
    with torch.no_grad():
        for images, labels in val_progress:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_progress.set_postfix({
                'loss': loss.item(),
                'val_accuracy': val_correct / val_total
            })

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    # 在 TensorBoard 上记录验证损失和准确率
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Accuracy', val_accuracy, epoch)

    scheduler.step(val_loss)

    log_message = (f'Epoch {epoch + 1}/{num_epochs}, '
                   f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, '
                   f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                   f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print(log_message)
    logger.info(log_message)

    # checkpointing
    model_names = f'{base_path}{opt.model}.{epoch:02d}-accuracy{train_accuracy:.2f}.pth'
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), model_names)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping")
        logger.info("Early stopping")
        break

logger.info("Training complete. Best model saved to: " + best_model_path)
print("Training complete. Best model saved to:", best_model_path)

# 关闭 SummaryWriter
writer.close()