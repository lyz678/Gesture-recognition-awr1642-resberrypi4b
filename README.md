从awr1642中获取毫米波雷达点云数据，用神经网进行手势识别。模型用PyTorch训练，模型达到了 99% 的准确率。并且将模型部署到raspberrypi4b上，实现了移动设备上的实时手势识别。

![Image text](https://assets.raspberrypi.com/static/raspberry-pi-4-labelled-f5e5dcdf6a34223235f83261fa42d1e8.png)

# 安装
```bash
git clone https://github.com/lyz678/Emotion-recogniton-pytorch.git  # clone
cd Emotion-recogniton-pytorch
pip install -r requirements.txt  # install
```

# Gesture Dataset
- 数据集来自 - [data](https://www.dropbox.com/scl/fi/y431rn0eauy2qkiz0y0g2/data.zip?rlkey=punhs9iquojldn6ug2owgnkbv&dl=0) (20k samples - 2k per class) ~120Mb
- 手势分类:
- None (random non-gestures)
- Swipe Up
- Swipe Down
- Swipe Right
- Swipe Left
- Spin Clockwise
- Spin Counterclockwise
- Letter Z
- Letter S
- Letter X
  
- 下载数据集到data文件夹中
  


# 训练和评估模型
```bash
python train.py
```
  
# .pth to .onnx
```bash
python export.py
```

# 预测和评估
```bash
python predict.py
python evalute.py
```
![Image text]([https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/result/demo1.jpg](https://github.com/lyz678/Gesture-recognition-awr1642boost-raspberrypi4b/blob/main/models/confusion.png))

# 连接awr1642实测
```bash
python main.py
```
# 在raspberrypi4b上运行

- 将main.py依赖的文件下载到raspberrypi4b

```bash
python main.py
```
# 1642串口数据读取参考
- https://github.com/ibaiGorordo/AWR1642-Read-Data-Python-MMWAVE-SDK-2
  
# 手势识别参考
- https://github.com/vilari-mickopf/mmwave-gesture-recognition



