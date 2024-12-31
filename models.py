import torch
import torch.nn as nn
import torch.nn.functional as F

# 第1个模型：基于 Conv1D 的 CNN 模型
class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.reshape = nn.Flatten(start_dim=2)  # 将输入 (None, 25, 40, 4) 转换为 (None, 25, 160)
        self.conv1 = nn.Conv1d(in_channels=160, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1536, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.reshape(x)                   # Reshape to (batch_size, 25, 160)
        x = x.permute(0, 2, 1)                # Permute to (batch_size, 160, 25)
        x = F.relu(self.conv1(x))            # Conv1D + ReLU
        x = self.pool(x)                     # MaxPooling1D
        x = F.relu(self.conv2(x))            # Conv1D + ReLU
        x = self.pool(x)                     # MaxPooling1D
        x = F.relu(self.conv3(x))            # Conv1D + ReLU
        x = self.flatten(x)                  # Flatten
        x = self.fc(x)                       # Dense
        x = self.softmax(x)                     # Softmax
        return x






# 第2个模型：基于 Conv2D 的 CNN 模型
class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=2560, out_features=10)  # 根据你的输入大小计算
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))                # Conv2D + ReLU
        x = self.pool(x)                         # MaxPooling2D
        x = F.relu(self.conv2(x))               # Conv2D + ReLU
        x = self.pool(x)                         # MaxPooling2D
        x = F.relu(self.conv3(x))               # Conv2D + ReLU
        x = self.flatten(x)                      # Flatten
        x = self.fc(x)                           # Dense
        x = self.softmax(x)                     # Softmax
        return x


# 第3个模型：基于 LSTM 的模型
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)  # TimeDistributed(Flatten)
        self.lstm1 = nn.LSTM(input_size=160, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256 * 25, 10)  # 展平后全连接层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)  # 假设输入是 (batch, 25, 40, 4)，将最后两维合并
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(x.size(0), -1)  # 展平
        x = self.fc(x)
        x = self.softmax(x)                     # Softmax
        return x




# 第4个模型：基于 Transformer 的模型
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)  # TimeDistributed(Flatten)
        self.dense = nn.Linear(160, 128)  # TimeDistributed(Dense)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Linear(128 * 25, 10)  # 最后展平后连接分类层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)  # 展平 (batch, 25, 40, 4) -> (batch, 25, 160)
        x = F.relu(self.dense(x))  # (batch, 25, 128)
        x = self.transformer(x)  # 送入 Transformer
        x = x.reshape(x.size(0), -1)  # 展平
        x = self.fc(x)  # 分类
        x = self.softmax(x)                     # Softmax
        return x


# 测试两个模型
if __name__ == "__main__":
    # 测试第1个模型
    cnn1d_model = CNN1DModel()
    input_data = torch.randn(1, 25, 40, 4)  # Batch size = 1
    cnn1d_output = cnn1d_model(input_data)
    print("CNN1D 模型输出维度:", cnn1d_output.shape)


    # 测试第2个模型
    cnn2d_model = CNN2DModel()
    cnn2d_output = cnn2d_model(input_data)
    print("CNN2D 模型输出维度:", cnn2d_output.shape)

    # 测试第3个模型
    lstm_model = LSTMModel()
    lstm_output = lstm_model(input_data)
    print("LSTMModel 模型输出维度:", lstm_output.shape)

    # 测试第4个模型
    transformer_model = TransformerModel()
    transformer_output = transformer_model(input_data)
    print("Transformer 模型输出维度:", transformer_output.shape)












