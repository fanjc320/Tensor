import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
from matplotlib import animation

plt.axis([0,100,0,1])
plt.ion()

xs = [0,0]
ys = [1,1]

def DyImg0():
    for i in range(100):
        y = np.random.random()
        xs[0]=xs[1]
        ys[0]=ys[1]
        xs[1]=i
        ys[1]=y
        plt.plot(xs,ys)
        plt.pause(0.1)

def DyImg1():
    fig,ax=plt.subplots()
    y1=[]
    for i in range(50):
        y1.append(i)
        ax.cla()
        ax.bar(y1,label='test',height=y1,width=0.3)
        ax.legend()
        plt.pause(0.1)

# DyImg0()
# DyImg1()


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    print("init-----")
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    print("updata-----")
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()

input()

# def update(i):
#     label = 'timestep {0}'.format(i)
#     print(label)
#     # 更新直线和x轴（用一个新的x轴的标签）。
#     # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
#     line.set_ydata(x - 5 + i)
#     ax.set_xlabel(label)
#     return line, ax
#
# if __name__ == '__main__':
#     # FuncAnimation 会在每一帧都调用“update” 函数。
#     # 在这里设置一个10帧的动画，每帧之间间隔200毫秒
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
#     # if len(sys.argv) > 1 and sys.argv[1] == 'save':
#     #     anim.save('line.gif', dpi=80, writer='imagemagick')
#     # else:
#     #     # plt.show() 会一直循环播放动画
#     plt.show()


