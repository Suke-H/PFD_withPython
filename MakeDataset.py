import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from method3d import *
from method2d import *
import figure2d as F2
import figure3d as F3
from Projection import Plane3DProjection

import open3d

def RandomCircle(x_min=-100, x_max=100, y_min=-100, y_max=100, r_min=0, r_max=20):
    """ ランダムな円生成 """

    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    r = Random(r_min, r_max)

    return F2.circle([x, y, r])

def RandomTriangle(x_min=-100, x_max=100, y_min=-100, y_max=100, r_min=0, r_max=20):
    """ ランダムな正三角形生成 """
    
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    r = Random(r_min, r_max)
    t = Random(0, np.pi*2/3)

    return F2.tri([x, y, r, t])

def RandomRectangle(x_min=-100, x_max=100, y_min=-100, y_max=100, w_min=0, w_max=20, h_min=0, h_max=20):
    """ ランダムな長方形生成 """
    
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    w = Random(w_min, w_max)
    h = Random(h_min, h_max)
    t = Random(0, np.pi/2)

    return F2.rect([x, y, w, h, t])

def RandomPlane(high=1000):
    """ ランダムな平面生成 """

    # 法線作成
    n = np.array([0, 0, 0])
    while LA.norm(n) == 0:
        n = np.random.rand(3)

    n = n / LA.norm(n)
    a, b, c = n

    # d作成
    d = Random(-high, high)

    return F3.plane([a, b, c, d])

def MakeSign2D(sign_type, scale):
    """
    いろんな種類の標識作成
    sign_type: 
    0: 半径0.3mの円
    1: 1辺0.8mの正三角形
    2: 1辺0.9mの正方形
    3. 1辺0.45mのひし形(てか正方形)
    4. 1辺が0.05～1のどれかの長方形

    scale: sign_typeのスケールを標準とした倍率
    """

    # 標識作成
    if sign_type == 0:
        r = 0.3*scale
        fig = F2.circle([0,0,r])
        fig_type = 0
        AABB = np.array([-0.35, 0.35, -0.35, 0.35])*scale

    elif sign_type == 1:
        r = 0.8/np.sqrt(3)*scale
        fig = F2.tri([0,0,r,Random(0,np.pi*2/3)])
        fig_type = 1
        AABB = np.array([-0.45, 0.45, -0.45, 0.45])*scale
        
    elif sign_type == 2:
        r = 0.9*scale
        fig = F2.rect([0,0,r,r,Random(0,np.pi/2)])
        fig_type = 2
        l = 0.9*np.sqrt(2)/2
        AABB = np.array([-l*1.1, l*1.1, -l*1.1, l*1.1])*scale

    elif sign_type == 3:
        r = 0.45*scale
        fig = F2.rect([0,0,r,r,Random(0,np.pi/2)])
        fig_type = 2
        l = 0.45*np.sqrt(2)/2
        AABB = np.array([-l*1.1, l*1.1, -l*1.1, l*1.1])*scale

    else:
        w = Random(0.5,2)
        h = Random(0.5,2)
        fig = F2.rect([0,0,w,h,Random(0,np.pi/2)])
        fig_type = 2
        l = np.sqrt(w**2+h**2)*np.sqrt(2)/2
        AABB = np.array([-l*1.1, l*1.1, -l*1.1, l*1.1])*scale

    return fig, fig_type, AABB

def MakeSign3D(sign_type, scale, density, noise_rate, low=-100, high=100, grid_step=50):
    """
    図形の点群＋ランダムに生成したAABB内にノイズ作成

    sign_type, scale: MakeSign2D参照
    density: 点密度(個/m^2)
    noise_rate: 全点群数に対するノイズ点群の割合
    
    """

    ## 平面図形設定 + 点群作成 ##
    fig2d, fig_type, AABB2d = MakeSign2D(sign_type, scale)
    fig_size = int(density*fig2d.CalcArea()//1)
    points2d = InteriorPoints2d(fig2d.f_rep, AABB2d, fig_size, grid_step=grid_step)

    # ノイズ点群の数と全点群数を出しておく
    noise_size = int(noise_rate/(1-noise_rate)*fig_size//1)

    ## 平面ランダム生成 ##
    plane = RandomPlane()

    ## 平面に2D点群を射影 ##
    # 平面上の2点をランダムに定める
    a, b, c, d = plane.p
    ox, oy, ax, ay = Random(low, high), Random(low, high), Random(low, high), Random(low, high)
    oz = (d - a*ox - b*oy) / c
    az = (d - a*ax - b*ay) / c
    O = np.array([ox, oy, oz])
    A = np.array([ax, ay, az])
    # uを定める
    u = norm(A - O)
    # 平面のnよりv算出
    n = np.array([a, b, c])
    v = norm(np.cross(u, n))
    # 平面3d射影
    _, points3d = Plane3DProjection(points2d, fig2d.p, u, v, O)

    ## ノイズ用AABBを、図形点群のAABBの1~1.5倍増しに作成 ##
    max_p, min_p, _ = buildAABB3d(points3d)
    xmax, ymax, zmax = max_p
    xmin, ymin, zmin = min_p

    p = [Random(1, 1.5) for i in range(6)]

    xmax = xmax + (xmax - xmin)/2 * p[0]
    xmin = xmin - (xmax - xmin)/2 * p[1]
    ymax = ymax + (ymax - ymin)/2 * p[2]
    ymin = ymin - (ymax - ymin)/2 * p[3]
    zmax = zmax + (zmax - zmin)/2 * p[4]
    zmin = zmin - (zmax - zmin)/2 * p[5]

    max_p = np.array([xmax, ymax, zmax])
    min_p = np.array([xmin, ymin, zmin])

    ## ノイズ生成、図形点群と結合 ##
    if noise_size == 0:
        # ノイズなし
        points = points3d[:, :]

    else:
        # AABBにランダムにノイズ生成
        noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(noise_size)])
        # 平面点群とノイズの結合 
        points = np.concatenate([points3d, noise])

    # 図形点群はTrue, ノイズはFalseでラベル付け
    trueIndex = np.array([True if i < fig_size else False for i in range(points.shape[0])])

    return fig2d.p, fig_type, AABB2d, plane.p, points, trueIndex, u, v, O
