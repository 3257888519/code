import tkinter
global b #定义全局变量
def sound_recording1() :
    import pyaudio
    import wave
    CHUNK = 1024 # 每个缓冲区的帧数
    FORMAT = pyaudio.paInt16 # 格式
    CHANNELS = 1 # 单声道
    RATE = 44100 # 采样频率
    RECORD_SECONDS = 5 #表示录音4s
    WAVE_OUTPUT_FILENAME = "input.wav"
    p = pyaudio.PyAudio() # 实例化对象
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) # 打开流，传入响应参数
    print("开始录音(测试者讲话)")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("结束录音(测试者讲话)")
    stream.stop_stream() # 关闭流
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') # 打开 wav 文件
    wf.setnchannels(CHANNELS) # 设置声道
    wf.setsampwidth(p.get_sample_size(FORMAT)) # 设置采样位数
    wf.setframerate(RATE) # 设置采样频率
    wf.writeframes(b''.join(frames))
    wf.close()


def sound_recording2() :
    import pyaudio
    import wave
    CHUNK = 1024 # 每个缓冲区的帧数
    FORMAT = pyaudio.paInt16 # 格式
    CHANNELS = 1 # 单声道
    RATE = 44100 # 采样频率
    RECORD_SECONDS = 5 #表示录音4s
    WAVE_OUTPUT_FILENAME = "noise.wav"
    p = pyaudio.PyAudio() # 实例化对象
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) # 打开流，传入响应参数
    print("开始录音(无人讲话时)")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("结束录音(无人讲话时)")
    stream.stop_stream() # 关闭流
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') # 打开 wav 文件
    wf.setnchannels(CHANNELS) # 设置声道
    wf.setsampwidth(p.get_sample_size(FORMAT)) # 设置采样位数
    wf.setframerate(RATE) # 设置采样频率
    wf.writeframes(b''.join(frames))
    wf.close()


def noise_reduction():
#1.读取音频文件，获取音频文件基本信息：采样个数，采样周期，与每个采样的声音信号值。绘制音频时域的：时间/位移图像
    import numpy as np
    import numpy.fft as nf #导入快速傅里叶变换所需模块
    import scipy.io.wavfile as wf
    import matplotlib.pyplot as plt

    #--------------------对噪音的处理------------------------------------------------
    #　读取音频文件
    #　sample_rate：采样频率(每秒采样多少个点)
    #  noised_sigs:　存储音频中每个采样点的采样位移0，1，-1(数字信号)
    sample_rate2, noised_sigs2 = wf.read('noise.wav')
    print(sample_rate2, noised_sigs2.shape)
    times2 = np.arange(noised_sigs2.size) / sample_rate2

    freqs2 = nf.fftfreq(times2.size, times2[1]-times2[0]) #通过采样数与采样周期求得傅里叶变换分解所得曲线的频率序列
    complex_array2 = nf.fft(noised_sigs2) #通过原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    pows2 = np.abs(complex_array2) #获得振幅


    plt.figure('降噪前后', figsize=(13, 6),facecolor='lightgray')
    plt.subplot(233)
    plt.title('Noise', fontsize=16)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.semilogy(freqs2[freqs2>0], pows2[freqs2>0], c='limegreen',label='Noised')
    plt.legend()

    # 寻找噪音能量最大的频率值
    fund_freq2 = freqs2[pows2.argmax()] #argmax返回振幅中最大数的索引
    

    # where函数寻找那些需要抹掉的复数的索引
    noised_indices2 = np.where(freqs2 <= fund_freq2)
    # 复制一个复数数组的副本，避免污染原始数据
    filter_complex_array2 = complex_array2.copy()
    filter_complex_array2[noised_indices2] = 0
    filter_pows2 = np.abs(filter_complex_array2)

    plt.subplot(236)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(freqs2[freqs2 >= 0], filter_pows2[freqs2 >= 0],c='dodgerblue', label='Filter')
    plt.legend()

#--------------------------------对测试音频的处理----------------------------------------------------------

#1.读取音频文件，获取音频文件基本信息：采样个数，采样周期，与每个采样的声音信号值。绘制音频时域的：时间/位移图像
    #　读取音频文件
    #　sample_rate：采样频率(每秒采样多少个点)
    #  noised_sigs:　存储音频中每个采样点的采样位移0，1，-1(数字信号)
    sample_rate1, noised_sigs1 = wf.read('input.wav')
    print(sample_rate1, noised_sigs1.shape)
    times1 = np.arange(noised_sigs1.size) / sample_rate1

    
    plt.subplot(231)
    plt.title('Time Domain', fontsize=16)
    plt.ylabel('Signal', fontsize=12) #设置y坐标轴名称和名称大小
    plt.tick_params(labelsize=10) #表示坐标轴的字体大小
    plt.grid(linestyle=':')
    plt.plot(times1[:150], noised_sigs1[:150],c='orangered', label='Noised') #150表示前150个点
    plt.legend()

#2.基于傅里叶变换，获取音频频域信息，绘制音频频域的：频率/能量图像
    freqs1 = nf.fftfreq(times1.size, times1[1]-times1[0]) #通过采样数与采样周期求得傅里叶变换分解所得曲线的频率序列
    complex_array1 = nf.fft(noised_sigs1) #通过原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    pows1 = np.abs(complex_array1) #获得振幅

    plt.subplot(232)
    plt.title('Frequency Domain', fontsize=16)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.semilogy(freqs1[freqs1>0], pows1[freqs1>0], c='limegreen',label='voice')
    plt.legend()



    fund_freq1 = freqs1[pows1.argmax()]
    print('音频频率最大值：')
    print(fund_freq1)
    print('噪声频率最大值：')
    print(fund_freq2)


#3.将低能噪声去除后绘制音频频域的：频率/能量图像
    # where函数寻找那些需要抹掉的复数的索引
    noised_indices = np.where(freqs1 <= fund_freq2)
    # 复制一个复数数组的副本，避免污染原始数据
    filter_complex_array = complex_array1.copy()
    filter_complex_array[noised_indices] = 0
    filter_pows = np.abs(filter_complex_array)

    plt.subplot(235)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(freqs1[freqs1 >= 0], filter_pows[freqs1 >= 0],c='dodgerblue', label='Filter')
    plt.legend()


#4.基于逆向傅里叶变换，生成新的音频信号，绘制音频时域的：时间/位移图像
    filter_sigs = nf.ifft(filter_complex_array).real #通过一个复数数组（复数的模代表的是振幅，复数的辐角代表初相位）经过逆向傅里叶变换得到合成的函数值数组
                                                     #.real表示获取实部
    plt.subplot(234)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Signal', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(times1[:150], filter_sigs[:150],c='hotpink', label='Filter') #150表示前150个点
    plt.legend()
#5.重新生成去除噪声之后的音频文件
    wf.write('output.wav',sample_rate1,filter_sigs)
    plt.show()

def features():
    #特征值提取
    import matplotlib.pyplot as plt
    import numpy as np
    from entropy import entropy
    from scipy import stats
    import scipy.io.wavfile as wf

    
    sampling_rate, data = wf.read('output.wav')


    frequency_components = np.fft.fft(data)  ##通过原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    frequency_components = np.abs(frequency_components)#获得振幅
    frequency_components = frequency_components[:int(sampling_rate/2)]
    frequency_values = np.linspace(0, len(frequency_components), len(frequency_components))


    #Converting to kHz
    frequency_components = [fc/1000 for fc in frequency_components[:280]]
    frequency_values = frequency_values[:280]



    #声频的平均值
    mean_freq = sum([ frequency_values[i] * frequency_components[i] for i in range(len(frequency_values))])/sum(frequency_components)
    #声频的标准差
    std_freq = np.std(frequency_values)
    #声频的中位数
    median_freq = np.median(frequency_values)
    #声频的第一个四分位数
    q25 = np.quantile(frequency_values, 0.25)
    #声频的第三个四分位数
    q75 = np.quantile(frequency_values, 0.75)
    #声频的四分位间距
    iqr = stats.iqr(frequency_values)

    global b
    #b最后会代入到预测模型中
    b=[mean_freq/1000, std_freq/1000, median_freq/1000, q25/1000, q75/1000, iqr/1000]
    #输出特征值
    print("----------------------------------------")
    print("Mean frequency: %lf" %(mean_freq/1000))
    print("Standard deviation: %lf" %(std_freq/1000))
    print("Median: %lf" %(median_freq/1000))
    print("Q25: %lf" %(q25/1000))
    print("Q75: %lf" %(q75/1000))
    print("IQR: %lf" %(iqr/1000))
    print("----------------------------------------")
    print(b)

def analysis():
    #learn
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier # 包装好的knn算法
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression



    
    # 导入训练集
    col_1names = ['0.059780985', '0.064241268', '0.032026913', '0.015071489', '0.09019344', '0.075121951', '0']
    train = pd.read_csv("train2.csv", header=None, names=col_1names)
    train.head()




    col_2names = ['0.059780985', '0.064241268', '0.032026913', '0.015071489', '0.09019344', '0.075121951']
    col_3names = ['0']
    X_train= train[col_2names]
    y_train=train[col_3names]

    #导入测试集
    col_11names = ['0.193130845', '0.061749731', '0.198130841', '0.134953271', '0.250093458', '0.115140187', '0']
    test = pd.read_csv("test2.csv", header=None, names=col_11names)
    test.head()

    col_21names = ['0.193130845', '0.061749731', '0.198130841', '0.134953271', '0.250093458', '0.115140187']
    col_31names = ['0']

    X_test = test[col_21names]
    y_test = test[col_31names]




    from sklearn.decomposition import PCA
    from sklearn import svm
    import time


    # 特征工程（标准化归一化）(是指数值减去均值，再除以标准差（可以把有量纲量变为无量纲量）)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)


    # using SVM
    C = [100]
    kernels = ['rbf']
    for i in enumerate(C):
       for k in kernels:
            clf1 = svm.SVC(C=i[1], kernel=k)
            clf1.fit(X_train, y_train)
            print('\n')
            print("SVM Accuracy (C={} & kernel={}):".format(i[1], k))
            print(accuracy_score(clf1.predict(X_test), y_test))
       



    #构建KNN模型
    knn = KNeighborsClassifier(n_neighbors=5) # knn中k取5

    knn.fit(X_train, y_train) # fit 拟合


    #结果预测与评分
    y_predict = knn.predict(X_test)
    print('\n')
    print("实际结果", y_test)
    print("预测结果", y_predict)
    print("准确率", knn.score(X_test, y_test))

    global b
    y_predict = knn.predict([b])
    print('\n输入的音频信号为:')
    if y_predict==[0]:
        print('                 男声')
    else:
        print('                 女声')
    print('\n')


#GUI
top=tkinter.Tk(className='gui') 
top.geometry('200x200')
button = tkinter.Button(top)
button['text'] = '     录音     '
button['command'] = sound_recording1
button.pack()


top.geometry('200x200')
button = tkinter.Button(top)
button['text'] = '    录噪音   '
button['command'] = sound_recording2
button.pack()


button = tkinter.Button(top)
button['text'] = '     降噪     '
button['command'] = noise_reduction
button.pack()

button = tkinter.Button(top)
button['text'] = ' 特征值提取'
button['command'] = features
button.pack()


button = tkinter.Button(top)
button['text'] = '  训练 分析 '
button['command'] = analysis
button.pack()
top.mainloop()