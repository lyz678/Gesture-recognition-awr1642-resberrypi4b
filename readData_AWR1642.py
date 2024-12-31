
import serial
import time
import numpy as np
import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QApplication



# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    Dataport = serial.Serial('/dev/ttyACM1', 921600)
    
    # Windows
    #CLIport = serial.Serial('COM3', 115200)
    #Dataport = serial.Serial('COM4', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 2
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;
                
            digOutSampleRate = int(splitWords[11]);
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters
   
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData16xx(Dataport, configParameters, byteBuffer, byteBufferLength):
    # 常量定义
    OBJ_STRUCT_SIZE_BYTES = 12  # 每个对象的数据结构大小（字节）
    BYTE_VEC_ACC_MAX_SIZE = 2**15  # 最大缓存大小
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1  # 表示检测到的点的TLV类型
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2  # 表示范围配置文件的TLV类型
    maxBufferSize = 2**15  # 最大缓存大小
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]  # 魔术字，用于标识数据包开始

    # 初始化变量
    magicOK = 0  # 检查是否读取到魔术字
    dataOK = 0  # 检查数据是否正确读取
    frameNumber = 0  # 帧编号
    detObj = {}  # 存储检测对象数据
    tlv_type = 0  # TLV类型

    # 从串口读取数据
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')  # 将数据转换为无符号8位整型数组
    byteCount = len(byteVec)  # 数据长度

    # 检查缓存是否已满，然后将数据加入缓存
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength += byteCount  # 更新缓存长度

    # 检查缓存中是否有足够的数据
    if byteBufferLength > 16:

        # 查找可能的魔术字位置
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # 确认魔术字是否完整，并记录起始索引
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):  # 检查是否匹配魔术字
                startIdx.append(loc)

        # 检查startIdx是否非空
        if startIdx:

            # 移除魔术字之前的数据
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]), dtype='uint8')
                byteBufferLength -= startIdx[0]  # 更新缓存长度

            # 检查缓存长度是否正确
            if byteBufferLength < 0:
                byteBufferLength = 0

            # 将4字节数据转换为32位数值的权重
            word = [1, 2**8, 2**16, 2**24]

            # 读取整个数据包长度
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # 检查是否读取到完整数据包
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1  # 表示魔术字已找到，且数据完整

    # 如果找到魔术字且数据完整，则开始解析消息
    if magicOK:
        # 初始化权重数组和指针索引
        word = [1, 2**8, 2**16, 2**24]
        idX = 0

        # 读取数据包头部信息
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')  # 软件版本
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)  # 数据包总长度
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')  # 平台信息
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)  # 帧编号
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)  # CPU周期时间
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)  # 检测到的对象数量
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)  # TLV块数量
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)  # 子帧编号
        idX += 4

        # 解析每个TLV块
        for tlvIdx in range(numTLVs):
            try:
                tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)  # TLV类型
                idX += 4
                tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)  # TLV长度
                idX += 4
            except:
                pass

            # 根据TLV类型解析数据
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                word = [1, 2**8]  # 16位数值的权重
                tlv_numObj = np.matmul(byteBuffer[idX:idX + 2], word)  # 对象数量
                idX += 2
                tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX + 2], word)  # 坐标量化格式
                idX += 2

                # 初始化存储对象信息的数组
                rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                peakVal = np.zeros(tlv_numObj, dtype='int16')
                x = np.zeros(tlv_numObj, dtype='int16')
                y = np.zeros(tlv_numObj, dtype='int16')
                z = np.zeros(tlv_numObj, dtype='int16')

                # 读取每个对象的数据
                for objectNum in range(tlv_numObj):
                    rangeIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    peakVal[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    x[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    y[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    z[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2

                # 数据后处理和存储
                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]  # 距离
                dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)] -= 65535
                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]  # 速度
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat

                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx,
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                dataOK = 1  # 数据成功解析

        # 移除已处理的数据
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]), dtype='uint8')
            byteBufferLength -= shiftSize

            # 检查缓存长度是否正确
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, detObj

# ------------------------------------------------------------------

# Function to update the data and display in the plot
def update(dataOk, detObj, showDoppler):
    x = []
    y = []
    doppler_texts = []
    if dataOk and len(detObj["x"]) > 0:
        x = -detObj["x"]
        y = detObj["y"]
        s.setData(x, y)  # Update the plot
        
        # Check if Doppler information should be displayed
        if showDoppler:
            for i in range(len(x)):
                # Add Doppler text beside each point
                doppler_info = f"{detObj['doppler'][i]:.2f} m/s"
                doppler_text = pg.TextItem(doppler_info, color='r', anchor=(0, 0))
                doppler_text.setPos(x[i], y[i])
                p.addItem(doppler_text)  # Add text item to the plot
                
                # Store the text item to remove later
                doppler_texts.append(doppler_text)
        
        QApplication.processEvents()  # Update the GUI
        
    return dataOk

# -------------------------    MAIN   -----------------------------------------  


if __name__ == "__main__":

    # 配置变量
    configFileName = './cfg/profile.cfg'   #profile 1642config
    CLIport = {}
    Dataport = {}
    byteBuffer = np.zeros(2**15,dtype = 'uint8')
    byteBufferLength = 0

    showDoppler = False  # 控制是否显示 Doppler 信息

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
    p.setXRange(-0.5, 0.5)
    p.setYRange(0, 1.5)
    p.setLabel('left', text='Y position (m)')
    p.setLabel('bottom', text='X position (m)')
    s = p.plot([], [], pen=None, symbol='o')
    win.show()

    # 主循环
    detObj = {}

    while True:
        try:
            # Read and parse the received data
            dataOk, frameNumber, detObj = readAndParseData16xx(Dataport, configParameters, byteBuffer, byteBufferLength)
            # 更新数据并检查数据是否正常
            dataOk = update(dataOk, detObj, showDoppler)

            time.sleep(0.03)  # 采样频率为 30Hz
            
        # 按下 Ctrl + C 停止程序并关闭所有资源
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            win.close()
            break





