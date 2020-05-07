import numpy as np
import numpy.linalg as LA
import itertools
import random
import time
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm(normal):
    """ ベクトルの正規化 """

     #ベクトルが一次元のとき
    if len(normal.shape)==1:
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
            
        else:
            return normal / np.linalg.norm(normal)

    #ベクトルが二次元
    else:
        #各法線のノルムをnormに格納
        norm = np.linalg.norm(normal, ord=2, axis=1)

        #normが0の要素は1にする(normalをnormで割る際に0除算を回避するため)
        norm = np.where(norm==0, 1, norm)

        #normalの各成分をノルムで割る
        norm = np.array([np.full(3, norm[i]) for i in range(len(norm))])
        return normal / norm

def Disassemble3d(XYZ):
    """
    点群データなどをx, y, zに分解する
    [x1, y1, z1]         [x1, x2, ..., xn]
        :        ->      [y1, y2, ..., yn]
    [xn, yn, zn]         [z1, z2, ..., zn]

    """
    XYZ = XYZ.T[:]
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]

    return X, Y, Z

def line3d(a, b):
    """ 線分abの2D点群生成 """

    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    z = a[2]*t + b[2]*(1-t)

    return x, y, z

def buildAABB3d(points):
    """
    3D点群のAABB(Axis Aligned Bounding Box)生成

    max_p: 最大の頂点[xmax, ymax]
    min_p: 最小の頂点[xmin, ymin]
    l: AABBの対角線の長さ
    """
    #なんとこれで終わり
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    l = np.sqrt((max_p[0]-min_p[0])**2 + (max_p[1]-min_p[1])**2 + (max_p[2]-min_p[2])**2)

    return max_p, min_p, l

def buildOBB3d(points):
    """
    3D点群のOBB(Oriented Bounding Box)生成

    max_p: 最大の頂点[xmax, ymax]
    min_p: 最小の頂点[xmin, ymin]
    l: OBBの対角線の長さ
    """

    #分散共分散行列Sを生成
    S = np.cov(points, rowvar=0, bias=1)

    #固有ベクトルを算出
    w,svd_vector = LA.eig(S)

    # 固有値が小さい順に固有ベクトルを並べる
    svd_vector = svd_vector[np.argsort(w)]

    #print(S)
    #print(svd_vector)
    #print("="*50)

    #正規直交座標にする(=直行行列にする)
    #############################################
    u = np.asarray([svd_vector[i] / np.linalg.norm(svd_vector[i]) for i in range(3)])

    #点群の各点と各固有ベクトルとの内積を取る
    #P V^T = [[p1*v1, p1*v2, p1*v3], ... ,[pN*v1, pN*v2, pN*v3]]
    inner_product = np.dot(points, u.T)
    
    #各固有値の内積最大、最小を抽出(max_stu_point = [s座標max, tmax, umax])
    max_stu_point = np.amax(inner_product, axis=0)
    min_stu_point = np.amin(inner_product, axis=0)

    #xyz座標に変換・・・単位ベクトル*座標
    #max_xyz_point = [[xs, ys, zs], [xt, yt, zt], [xu, yu, zu]]
    max_xyz_point = np.asarray([u[i]*max_stu_point[i] for i in range(3)])
    min_xyz_point = np.asarray([u[i]*min_stu_point[i] for i in range(3)])

    #対角線の長さ
    vert_max = min_xyz_point[0] + min_xyz_point[1] + max_xyz_point[2]
    vert_min = max_xyz_point[0] + max_xyz_point[1] + min_xyz_point[2]
    l = np.linalg.norm(vert_max-vert_min)

    return max_xyz_point, min_xyz_point, l

def MakePoints3D(fn, bbox=(-2.5,2.5), grid_step=50, down_rate = 0.5, epsilon=0.05):
    """
    図形の3D点群生成

    fn: F-Rep
    bbox: 描画する範囲
    grid_step: グリッドの数。ここで点群数を調整
    down_rate: 点群を間引く割合
    epsilon: 点群と図形との許容距離

    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    # 点群X, Y, Z, pointsを作成
    x = np.linspace(xmin, xmax, grid_step)
    y = np.linspace(ymin, ymax, grid_step)
    z = np.linspace(zmin, zmax, grid_step)

    X, Y, Z = np.meshgrid(x, y, z)

    # 格子点X, Y, Zをすべてfnにぶち込んでみる
    W = np.array([[fn(X[i][j], Y[i][j], Z[i][j]) for j in range(grid_step)] for i in range(grid_step)])
    # 変更前
    #W = fn(X, Y, Z)

    # Wが0に近いインデックスを取り出す
    index = np.where(np.abs(W)<=epsilon)
    index = [(index[0][i], index[1][i], index[2][i]) for i in range(len(index[0]))]

    # ランダムにダウンサンプリング
    index = random.sample(index, int(len(index)*(1-down_rate)//1))

    # 格子点から境界面(fn(x,y,z)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])
    pointZ = np.array([Z[i] for i in index])

    # points作成([[x1,y1,z1],[x2,y2,z2],...])    
    points = np.stack([pointX, pointY, pointZ])
    points = points.T

    return points

def OBBViewer3d(ax, max_p, min_p):
    """ OBB描画 """

    #直積：[smax, smin]*[tmax, tmin]*[umax, umin] <=> 頂点
    s_axis = np.vstack((max_p[0], min_p[0]))
    t_axis = np.vstack((max_p[1], min_p[1]))
    u_axis = np.vstack((max_p[2], min_p[2]))

    products = np.asarray(list(itertools.product(s_axis, t_axis, u_axis)))
    vertices = np.sum(products, axis=1)

    #各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))


    #頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y, z = line3d(vertices[i], vertices[j])
                ax.plot(x,y,z,marker=".",color="orange")

    #OBBの頂点の1つ
    vert_max = min_p[0] + min_p[1] + max_p[2]
    vert_min = max_p[0] + max_p[1] + min_p[2]

    #xyzに分解
    Xmax, Ymax, Zmax = Disassemble3d(max_p)
    Xmin, Ymin, Zmin = Disassemble3d(min_p)


    #頂点なども描画
    ax.plot(Xmax,Ymax,Zmax,marker="X",linestyle="None",color="red")
    ax.plot(Xmin,Ymin,Zmin,marker="X",linestyle="None",color="blue")
    ax.plot([vert_max[0], vert_min[0]],[vert_max[1], vert_min[1]],[vert_max[2], vert_min[2]],marker="o",linestyle="None",color="black")

def AABBViewer3d(ax, max_p, min_p):
    """ AABB描画 """

    # [xmax, xmin]と[ymax, ymin]の直積 <=> 頂点
    x_axis = [max_p[0], min_p[0]]
    y_axis = [max_p[1], min_p[1]]
    z_axis = [max_p[2], min_p[2]]

    vertices = np.asarray(list(itertools.product(x_axis, y_axis, z_axis)))

    # 各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))

    # 頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y, z = line3d(vertices[i], vertices[j])
                ax.plot(x,y,z,marker=".",color="orange")
