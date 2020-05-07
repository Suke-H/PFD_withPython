from numpy import linalg as LA
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import itertools

import figure2d as F2

def Random(a, b):
    """ aからbまでのランダムな実数値を返す """
    return (b - a) * np.random.rand() + a


def Disassemble2d(XY):
    """
    点群データをx, yに分解する

    [x1, y1]         [x1, x2, ..., xn]
       :       ->    [y1, y2, ..., yn]
    [xn, yn]    

    """
    X, Y = XY.T[:]

    return X, Y

def Composition2d(X, Y):
    """
    x, yを点群データに構成する

    [x1, x2, ..., xn]        [x1, y1]         
    [y1, y2, ..., yn]    ->     : 
                             [xn, yn]    

    """
    points = np.stack([X, Y])
    points = np.transpose(points, (1,0))

    return points

def line2d(a, b):
    """ 線分abの2D点群生成 """

    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)

    return x, y
    
def norm(vector):
    """ ベクトルの正規化 """

     # ベクトルが一次元のとき
    if len(vector.shape)==1:
        if LA.norm(vector) == 0:
            return vector
            
        else:
            return vector / np.linalg.norm(vector)

    # ベクトルが二次元
    else:
        # 各法線のノルムをnormに格納
        norm = LA.norm(vector, ord=2, axis=1)

        # normが0の要素は1にする(vectorをnormで割る際に0除算を回避するため)
        norm = np.where(norm==0, 1, norm)

        # vectorの各成分をノルムで割る
        norm = np.array([np.full(2, norm[i]) for i in range(len(norm))])
        return vector / norm

def ContourPoints2d(fn, AABB=None, AABB_size=1, grid_step=1000, down_rate = 0, epsilon=0.01):
    """
    図形の境界線の2D点群を生成
    
    fn: F-Rep
    AABB: 描画する範囲(指定してなければbbox)
    AABB_size: 入力したAABBの範囲の拡大率
    grid_step: グリッドの数。ここで点群数を調整
    down_rate: 点群を間引く割合
    epsilon: 点群と図形との許容距離

    """

    if AABB is None:
        bbox=(-2.5,2.5)
        xmin, xmax, ymin, ymax= bbox*2
    else:
        xmin, xmax, ymin, ymax= AABB
        #AABBの各辺がAABB_size倍されるように頂点を変更
        xmax = xmax + (xmax - xmin)/2 * AABB_size
        xmin = xmin - (xmax - xmin)/2 * AABB_size
        ymax = ymax + (ymax - ymin)/2 * AABB_size
        ymin = ymin - (ymax - ymin)/2 * AABB_size

    #点群X, Y, pointsを作成
    x = np.linspace(xmin, xmax, grid_step)
    y = np.linspace(ymin, ymax, grid_step)

    X, Y= np.meshgrid(x, y)

    # 格子点X, Yをすべてfnにぶち込んでみる
    W = np.array([fn(X[i], Y[i]) for i in range(grid_step)])
    # 変更前
    #W = fn(X, Y)

    #Ｗが0に近いインデックスを取り出す
    index = np.where(np.abs(W)<=epsilon)
    index = [(index[0][i], index[1][i]) for i in range(len(index[0]))]

    #ランダムにダウンサンプリング
    index = random.sample(index, int(len(index)*(1-down_rate)//1))

    #格子点から境界面(fn(x,y)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])

    #points作成([[x1,y1],[x2,y2],...])   
    points = Composition2d(pointX, pointY)

    return points

def InteriorPoints2d(fn, AABB, sampling_size, grid_step=50):
    """
    図形の内部に2D点群生成

    fn: F-Rep
    AABB: 描画する範囲
    sampling_size: 生成する点群の数
    grid_step: グリッドの数。ここで点群数を調整

    """
    xmin, xmax, ymin, ymax = AABB

    while True:

        # 点群X, Y, Z, pointsを作成
        x = np.linspace(xmin, xmax, grid_step)
        y = np.linspace(ymin, ymax, grid_step)

        X, Y = np.meshgrid(x, y)

        # 格子点X, Yをすべてfnにぶち込んでみる
        W = np.array([fn(X[i], Y[i]) for i in range(grid_step)])
        # 変更前
        # W = fn(X, Y, Z)

        # Ｗ>=0のインデックスを取り出す
        index = np.where(W >= 0)
        index = [(index[0][i], index[1][i]) for i in range(len(index[0]))]
        # print(index)

        # indexがsampling_size分なかったらgrid_stepを増やしてやり直し
        if len(index) >= sampling_size:
            break

        grid_step += 10

    # サンプリング
    index = random.sample(index, sampling_size)

    # 格子点から境界面(fn(x,y,z)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])

    # points作成([[x1,y1,z1],[x2,y2,z2],...])
    points = Composition2d(pointX, pointY)

    return points

def buildAABB2d(points):
    """
    2D点群のAABB(Oriented Bounding Box)生成

    max_p: 最大の頂点[xmax, ymax]
    min_p: 最小の頂点[xmin, ymin]
    l: AABBの対角線の長さ
    """

    # (x,y)の最大と最小をとる
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    # 面積算出
    area = abs((max_p[0]-min_p[0]) * (max_p[1]-min_p[1]))

    return max_p, min_p, LA.norm(max_p - min_p), area

def buildOBB2d(points):
    """
    3D点群のOBB(Oriented Bounding Box)生成

    max_p: 最大の頂点[xmax, ymax]
    min_p: 最小の頂点[xmin, ymin]
    l: OBBの対角線の長さ
    """

    # 分散共分散行列Sを生成
    S = np.cov(points, rowvar=0, bias=1)

    # 固有ベクトルを算出
    w,svd_vector = LA.eig(S)

    # 固有値が小さい順に固有ベクトルを並べる
    svd_vector = svd_vector[np.argsort(w)]

    # 正規直交座標にする(=直行行列にする)
    # u = [sの単位ベクトル, tの単位ベクトル]になる
    u = np.asarray([svd_vector[i] / np.linalg.norm(svd_vector[i]) for i in range(2)])

    # 点群の各点と各固有ベクトルとの内積を取る
    # P V^T = [[p1*v1, p1*v2], ... ,[pN*v1, pN*v2]]
    inner_product = np.dot(points, u.T)
    
    # 各固有値の内積最大、最小を抽出(max_st_point = [s座標max, tmax])
    max_st_point = np.amax(inner_product, axis=0)
    min_st_point = np.amin(inner_product, axis=0)

    # xyz座標に変換・・・単位ベクトル*座標
    # max_xyz_point = [[xs, ys], [xt, yt]]
    max_xy_point = np.asarray([u[i]*max_st_point[i] for i in range(2)])
    min_xy_point = np.asarray([u[i]*min_st_point[i] for i in range(2)])

    #################################################################

    # 図形作成処理

    # s, t軸の単位ベクトル
    s, t = u
    # 法線: s, -s, t, -s
    # c: 法線と1点(smaxなど)との内積
    lx1 = F2.line([s[0], s[1], np.dot(max_xy_point[0], s)])
    lx2 = F2.line([-s[0], -s[1], -np.dot(min_xy_point[0], s)])
    ly1 = F2.line([t[0], t[1], np.dot(max_xy_point[1], t)])
    ly2 = F2.line([-t[0], -t[1], -np.dot(min_xy_point[1], t)])
    OBB = F2.inter(lx1, F2.inter(lx2, F2.inter(ly1, ly2)))

    # 対角線の長さ算出
    vert_max = min_xy_point[0] + min_xy_point[1]
    vert_min = max_xy_point[0] + max_xy_point[1]
    l = np.linalg.norm(vert_max-vert_min)

    # 面積算出
    area = abs((max_st_point[0]-min_st_point[0]) * (max_st_point[1]-min_st_point[1]))

    return max_xy_point, min_xy_point, OBB, l, area

def AABBViewer2d(max_p, min_p):
    """ AABB描画 """

    # [xmax, xmin]と[ymax, ymin]の直積 <=> 頂点
    x_axis = [max_p[0], min_p[0]]
    y_axis = [max_p[1], min_p[1]]

    vertices = np.asarray(list(itertools.product(x_axis, y_axis)))

    # 各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit)))

    # 頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y = line2d(vertices[i], vertices[j])
                plt.plot(x,y,marker=".",color="red")

def OBBViewer2d(max_p, min_p):
    """ OBB描画 """

    # 直積：[smax, smin]*[tmax, tmin] <=> 頂点
    s_axis = np.vstack((max_p[0], min_p[0]))
    t_axis = np.vstack((max_p[1], min_p[1]))

    products = np.asarray(list(itertools.product(s_axis, t_axis)))
    vertices = np.sum(products, axis=1)

    # 各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit)))

    # 頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y = line2d(vertices[i], vertices[j])
                plt.plot(x,y,marker=".",color="orange")

def PlotContour2d(hull, color="red"):
    """ 2D輪郭点を描画 """

    # 点プロット
    X, Y = Disassemble2d(hull)
    plt.plot(X, Y, marker=".",linestyle="None",color=color)

    # hullを[0,1,2,..n] -> [1,2,...,n,0]の順番にしたhull2作成
    hull2 = list(hull[:])
    a = hull2.pop(0)
    hull2.append(a)
    hull2 = np.array(hull2)

    # hull2を利用して線を引く
    for a, b in zip(hull, hull2):
        LX, LY = line2d(a, b)
        plt.plot(LX, LY, color=color)
