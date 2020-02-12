import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from method import *
import figure2 as F

# 点群の中から図形にフィットする点のインデックスを返す
def CountPoints(figure, points, normals, epsilon, alpha):

    X, Y, Z = Disassemble(points)

    #|f(x,y,z)|<εを満たす点群だけにする
    D = figure.f_rep(X, Y, Z)
    index_1 = np.where(np.abs(D)<epsilon)

    #次にcos-1(|nf*ni|)<αを満たす点群だけにする
    T = np.arccos(np.abs(np.sum(figure.normal(X,Y,Z) * normals, axis=1)))
    # (0<T<pi/2のはずだが念のため絶対値をつけてる)
    index_2 = np.where(np.abs(T)<alpha)
    
    #どちらも満たすindexを残す
    index = list(filter(lambda x: x in index_2[0], index_1[0]))

    return index, len(index)

def PlaneDict(points, normals, epsilon, alpha):

    X, Y, Z = Disassemble(points)

    n = points.shape[0]
    N = 5000
    # ランダムに3点ずつN組抽出
    points_set = points[np.array([np.random.choice(n, 3, replace=False) for i in range(N)]), :]
    #points_set = points[np.random.choice(n, size=(int((n-n%3)/3), 3), replace=False), :]
    
    #print("points:{}".format(points_set.shape))

    # 分割
    # [a1, b1, c1] -> [a1] [b1, c1]
    a0, a1 = np.split(points_set, [1], axis=1)

    # a2 = [[b1-a1], ...,[bn-an]]
    #      [[c1-a1], ...,[cn-an]]
    a2 = np.transpose(a1-a0, (1,0,2))

    # n = (b-a) × (c-a)
    n = np.cross(a2[0], a2[1])

    # 単位ベクトルに変換
    n = norm(n)

    # d = n・a
    a0 = np.reshape(a0, (a0.shape[0],3))
    d = np.sum(n*a0, axis=1)

    # パラメータ
    # p = [nx, ny, nz, d]
    d = np.reshape(d, (d.shape[0],1))
    p = np.concatenate([n, d], axis=1)

    # 平面生成
    Planes = [F.plane(p[i]) for i in range(p.shape[0])]

    # フィットしている点の数を数える
    Scores = [CountPoints(Planes[i], points, normals, epsilon, alpha)[1] for i in range(p.shape[0])]

    print(p[Scores.index(max(Scores))])

    return Planes[Scores.index(max(Scores))]

# 入力：点群、法線
# 出力：最適平面のパラメータ、フィット点のインデックス
def RANSAC(points, normals, epsilon=0.05, alpha=np.pi/8):

    X, Y, Z = Disassemble(points)

    # 平面検出
    figure = PlaneDict(points, normals, epsilon, alpha)
    
    # フィット点を抽出
    index, num = CountPoints(figure, points, normals, epsilon, alpha)

    return figure, index, num
