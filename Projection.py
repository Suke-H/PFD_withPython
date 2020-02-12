import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import numpy as np

# zenrin
from method2d import *

# main
import figure2 as F
from method import *

def Plane2DProjection(points, plane):
    N = points.shape[0]

    # 平面のパラメータ
    a, b, c, d = plane.p
    n = np.array([a, b, c])

    # 法線方向に点を動かして平面に落とし込む
    # f(p0 + t n) = 0 <=> t = f(p0)/(a^2+b^2+c^2)
    X, Y, Z = Disassemble(points)
    t = plane.f_rep(X,Y,Z) / (a**2+b**2+c**2)

    # p = p0 + t n
    tn = np.array([t[i]*n for i in range(N)])
    plane_points = points + tn

    # 新しい原点を適当に選んだ1点にする
    O = plane_points[0]
    # 適当な2点のベクトルを軸の1つにする
    u = norm(plane_points[1] - O)
    # v = u × n
    v = norm(np.cross(u, n))
    # UV座標に変換
    UVvector = np.array([[np.dot((plane_points[i]-O), u), np.dot((plane_points[i]-O), v)]for i in range(N)])

    # reshape
    #UVvector = np.reshape(UVvector, (N, 2))

    # 平面に落とした点の"3次元"座標、"2次元"座標, u, v, O 出力
    return plane_points, UVvector, u, v, O

def Plane3DProjection(points2d, para, u, v, O):

    # 三次元に射影
    uv = np.array([u, v])
    points3d = np.dot(points2d, uv) + np.array([O for i in range(points2d.shape[0])])

    # 中心座標を3次元に射影する
    # 中心座標だけ抽出
    para2d = para[:]
    center2d = np.array([para2d[0], para2d[1]])
    # center2d = []
    # center2d.append(para2d.pop(0))
    # center2d.append(para2d.pop(0))
    # 中心座標を3次元射影
    center3d = np.dot(center2d, uv) + O
    # パラメータ = 中心座標(3d) + 座標抜きパラメータ
    # para3d = np.concatenate([center3d, para2d])

    return center3d, points3d


# def Plane3DProjection2(para, u, v, O):

#     # 三次元に射影
#     N = np.array([u, v])
#     N_inv = np.linalg.inv(N)

#     W = f()




#     points3d = np.dot(points2d, uv) + np.array([O for i in range(points2d.shape[0])])

#     # 中心座標を3次元に射影する
#     # 中心座標だけ抽出
#     para2d = para[:]
#     center2d = np.array([para2d[0], para2d[1]])
#     # center2d = []
#     # center2d.append(para2d.pop(0))
#     # center2d.append(para2d.pop(0))
#     # 中心座標を3次元射影
#     center3d = np.dot(center2d, uv) + O
#     # パラメータ = 中心座標(3d) + 座標抜きパラメータ
#     # para3d = np.concatenate([center3d, para2d])

#     return center3d, points3d