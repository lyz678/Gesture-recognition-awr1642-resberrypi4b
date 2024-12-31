import time
import onnxruntime
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from readData_AWR1642 import serialConfig, parseConfigFile, readAndParseData16xx



# Function to update the data and display in the plot
def update(byteBuffer, byteBufferLength, frame_buffer, target_frames : 25, target_points : 40):
    global record
    dataOk = 0
    x = []
    y = []
    # 读取并解析毫米波数据
    dataOk, frameNumber, detObj = readAndParseData16xx(Dataport, configParameters, byteBuffer, byteBufferLength)
    if dataOk:
        record = True

    if dataOk and record and len(detObj["x"]) > 0:
        x = detObj["x"]
        y = detObj["y"]
        # print(len(x))
        # 组装 (x, y, doppler, range) 点云数据
        radar_points = np.column_stack((detObj["x"], detObj["y"], detObj["doppler"], detObj["range"]))
        
        # 如果点数不足补全
        if radar_points.shape[0] < target_points:
            padding = np.zeros((target_points - radar_points.shape[0], 4))  # 4维(x, y, doppler, range)
            radar_points = np.vstack((radar_points, padding))
        
        radar_points = radar_points[:target_points]
        
        # 更新帧缓冲区，先进先出机制
        if len(frame_buffer) >= target_frames:  # 如果已经存储了25帧，移除最早的一帧
            frame_buffer.pop(0)
        frame_buffer.append(radar_points)  # 添加新帧
        print(len(frame_buffer))

        s.setData(x, y)  # 更新绘图
        QApplication.processEvents()  # 更新GUI
    elif record and len(frame_buffer) < target_frames:
        frame_buffer.append(np.zeros((target_points, 4)))

    if len(frame_buffer) == target_frames:
        record = False
        input_data = np.stack(frame_buffer, axis=0)
        input_data = input_data.astype(np.float32)  # 确保输入数据类型为float32
        input_data = np.expand_dims(input_data, axis=0) # 增加 batch 维度
        
        # 使用ONNX Runtime进行推理
        outputs = session.run([output_name], {input_name: input_data})

        # 获取推理结果的类别索引
        pred_class_index = np.argmax(outputs[0])
        pred_class = classes[pred_class_index]  # 根据索引获取预测类别
        print(pred_class)
        
        frame_buffer.clear()  # 清空frame_buffer
        s.setData([], [])  # 清除点云数据
        QApplication.processEvents()  # 更新GUI
    return dataOk

# -------------------------    MAIN   -----------------------------------------  


if __name__ == "__main__":

    # 配置变量
    configFileName = './cfg/profile.cfg'   #profile  1642config
    CLIport = {}
    Dataport = {}
    byteBuffer = np.zeros(2**15,dtype = 'uint8')
    byteBufferLength = 0;
    frame_buffer = []
    record = False

    # 配置串口
    CLIport, Dataport = serialConfig(configFileName)

    # 从配置文件中获取参数
    configParameters = parseConfigFile(configFileName)

    # 启动 Qt 应用程序
    app = QApplication([])

    # 设置绘图
    pg.setConfigOption('background', 'w')
    win = pg.GraphicsLayoutWidget(title="2D scatter plot")
    p = win.addPlot()
    p.setXRange(-1, 1)
    p.setYRange(0, 1.5)
    p.setLabel('left', text='Y position (m)')
    p.setLabel('bottom', text='X position (m)')
    s = p.plot([], [], pen=None, symbol='o')
    win.show()

    # 定义类别列表
    classes = ['ccw', 'cw', 'down', 'left', 'none', 'right', 's', 'up', 'x', 'z']

    # 加载ONNX模型
    onnx_model_path = "./models/best_model.onnx"
    session = onnxruntime.InferenceSession(onnx_model_path)

    # 获取模型的输入和输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    while True:
        try:
            # 更新数据并检查数据是否正常
            dataOk = update(byteBuffer, byteBufferLength, frame_buffer, target_frames = 25, target_points = 40)
            
            time.sleep(0.066)  # 采样频率为15Hz
            
        # 按下 Ctrl + C 停止程序并关闭所有资源
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            win.close()
            break




