
# 本文演示代码用于滤出图像中的低频信号。
import numpy as np
from PIL import Image
from numpy.fft import fft, ifft

def filterImage(srcImage):
    # 打开图像文件并获取数据
    srcIm = Image.open(srcImage)
    srcArray = np.fromstring(srcIm.tobytes(), dtype=np.int8)

    # 傅里叶变换并滤除低频信号
    result = fft(srcArray)
    result = np.where(np.absolute(result)<9e3, 0, result)
    # 傅里叶反变换，保留实部
    result = ifft(result)
    result = np.int8(np.real(result))

    # 转换为图像
    im = Image.frombytes(srcIm.mode, srcIm.size, result)
    im.show()

filterImage('lion.jpg')