import tkinter
def hanshu0() :
    import pyaudio
    import wave
    CHUNK = 1024 # 每个缓冲区的帧数
    FORMAT = pyaudio.paInt16 # 采样位数
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
    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream() # 关闭流
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') # 打开 wav 文件
    wf.setnchannels(CHANNELS) # 声道设置
    wf.setsampwidth(p.get_sample_size(FORMAT)) # 采样位数设置
    wf.setframerate(RATE) # 采样频率设置
    wf.writeframes(b''.join(frames))
    wf.close()
def hanshu1():
#1.读取音频文件，获取音频文件基本信息：采样个数，采样周期，与每个采样的声音信号值。绘制音频时域的：时间/位移图像
    import numpy as np
    import numpy.fft as nf #导入快速傅里叶变换所需模块
    import scipy.io.wavfile as wf
    import matplotlib.pyplot as plt

    #　读取音频文件
    #　sample_rate：采样率
    #  noised_sigs:　存储音频中每个采样点的采样位移
    sample_rate, noised_sigs = wf.read('input.wav')
    print(sample_rate, noised_sigs.shape)
    times = np.arange(noised_sigs.size) / sample_rate

    plt.figure('Filter', facecolor='lightgray')
    plt.subplot(221)
    plt.title('Time Domain', fontsize=16)
    plt.ylabel('Signal', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(times[:178], noised_sigs[:178],c='orangered', label='Noised')
    plt.legend()

#2.基于傅里叶变换，获取音频频域信息，绘制音频频域的：频率/能量图像
    #　傅里叶变换后，绘制频域图像
    freqs = nf.fftfreq(times.size, times[1]-times[0]) #通过采样数与采样周期求得傅里叶变换分解所得曲线的频率序列
    complex_array = nf.fft(noised_sigs) #通过原函数值的序列j经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    pows = np.abs(complex_array)

    plt.subplot(222)
    plt.title('Frequency Domain', fontsize=16)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.semilogy(freqs[freqs>0], pows[freqs>0], c='limegreen',label='Noised')
    plt.legend()


#3.将低能噪声去除后绘制音频频域的：频率/能量图像
    # 寻找能量最大的频率值
    fund_freq = freqs[pows.argmax()]
    # where函数寻找那些需要抹掉的复数的索引
    noised_indices = np.where(freqs != fund_freq)
    # 复制一个复数数组的副本，避免污染原始数据
    filter_complex_array = complex_array.copy()
    filter_complex_array[noised_indices] = 0
    filter_pows = np.abs(filter_complex_array)

    plt.subplot(224)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(freqs[freqs >= 0], filter_pows[freqs >= 0],c='dodgerblue', label='Filter')
    plt.legend()


#4.基于逆向傅里叶变换，生成新的音频信号，绘制音频时域的：时间/位移图像
    filter_sigs = nf.ifft(filter_complex_array).real
    plt.subplot(223)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Signal', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(times[:178], filter_sigs[:178],c='hotpink', label='Filter')
    plt.legend()
#5.重新生成去除噪声之后的音频文件
    wf.write('output.wav',sample_rate,filter_sigs)
    

    #特征值提取
    import librosa as librosa
    import matplotlib.pyplot as plt
    import numpy as np
    from entropy import entropy
    from scipy import stats



    data, sampling_rate = librosa.load('output.wav')


    frequency_components = np.fft.fft(data)  #快速傅里叶变换
    frequency_components = np.abs(frequency_components)
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


    # 特征工程（标准化即归一化）(是指数值减去均值，再除以标准差（可以把有量纲量变为无量纲量）)
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


    
    y_predict = knn.predict([b])
    if y_predict==[0]:
        print('男')
    else:
        print('女')
    print('\n')


#GUI
top=tkinter.Tk(className='gui') 
top.geometry('200x200')
button = tkinter.Button(top)
button['text'] = 'sound recording'
button['command'] = hanshu0
button.pack()


button = tkinter.Button(top)
button['text'] = 'analysis'
button['command'] = hanshu1
button.pack()
top.mainloop()