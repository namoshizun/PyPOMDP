from matplotlib import pyplot as plt
import numpy as np

def make(lower,higher):
    x = np.linspace(lower, higher,100)
    y1,y2,y3 =fn1(x), fn2(x), fn3(x)

    plt.plot(x,y1, label='go top')
    plt.plot(x,y2, label='go left')
    plt.plot(x,y3, label='go right')
    plt.legend(loc='lower right')

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0).reshape(-1) + 0
    plt.plot(x[idx], y1[idx], 'ro')

    idx = np.argwhere(np.diff(np.sign(y2 - y3)) != 0).reshape(-1) + 0
    plt.plot(x[idx], y2[idx], 'ro')

    idx = np.argwhere(np.diff(np.sign(y1 - y3)) != 0).reshape(-1) + 0
    plt.plot(x[idx], y1[idx], 'ro')

    plt.ylim(ymin=0.)
    plt.show()

r1,r2,r3=10,20,40
fn1=lambda x: r1 + 4*r1*(x-2) - 1
fn2=lambda x: r2 + 4*r2*(x-4) - 3
fn3=lambda x: r3 + 4*r3*(x-5) - 4
make(2,7)
plt.show()