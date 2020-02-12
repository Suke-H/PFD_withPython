import numpy as np
import numpy.linalg as LA
import itertools
import random

#seabornはimportしておくだけでもmatplotlibのグラフがきれいになる
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import NearestNeighbors

"""
def AND(f1, f2):
    return lambda x,y,z: f1(x,y,z) + f2(x,y,z) - np.sqrt(f1(x,y,z)**2 + f2(x,y,z)**2)

def OR(f1, f2):
    return lambda x,y,z: f1(x,y,z) + f2(x,y,z) + np.sqrt(f1(x,y,z)**2 + f2(x,y,z)**2)

def NOT(f):
    return lambda x,y,z: -f(x,y,z)
"""

def norm(normal):
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


#pointsからpのk近傍点のindexのリストを返す
def K_neighbor(points, p, k):
    #points[i]とpointsの各点とのユークリッド距離を格納
    distances = np.sum(np.square(points - p), axis=1)

    #距離順でpointsをソートしたときのインデックスを格納
    sorted_index = np.argsort(distances)

    return sorted_index[:k]

def K_neighbor2(points, k):
    # scikit-learnより全ての点群のk近傍のインデックスを受け取る
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(points)
    _, indices = nn.kneighbors(points)

    # 自分もk近傍に含んじゃってるので自分を消す処理
    mask = indices != np.arange(indices.shape[0])[:,np.newaxis]
    mask[:,-1] &= np.logical_not(mask.all(axis=1))
    shape = (indices.shape[0], indices.shape[1] - 1)

    return indices[mask].reshape(shape)


#点群データなどをx, y, zに分解する

#[x1, y1, z1]         [x1, x2, ..., xn]
#      :        ->    [y1, y2, ..., yn]
#[xn, yn, zn]         [z1, z2, ..., zn]
def Disassemble(XYZ):
    XYZ = XYZ.T[:]
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]

    return X, Y, Z

def line(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    z = a[2]*t + b[2]*(1-t)

    return x, y, z

###OBB生成####
def buildOBB(points):
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

    """
    max_index = 
    print(max_index)
    max_point = np.asarray([points[max_index[i]] for i in range(3)])

    min_index = np.argmin(inner_product, axis=0)
    min_point = np.asarray([points[min_index[i]] for i in range(3)])
    """
    #対角線の長さ
    vert_max = min_xyz_point[0] + min_xyz_point[1] + max_xyz_point[2]
    vert_min = max_xyz_point[0] + max_xyz_point[1] + min_xyz_point[2]
    l = np.linalg.norm(vert_max-vert_min)

    return max_xyz_point, min_xyz_point, l

###AABB生成####
def buildAABB(points):
    #なんとこれで終わり
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    l = np.sqrt((max_p[0]-min_p[0])**2 + (max_p[1]-min_p[1])**2 + (max_p[2]-min_p[2])**2)

    return max_p, min_p, l
    

def MakePoints(fn, bbox=(-2.5,2.5), grid_step=50, down_rate = 0.5, epsilon=0.05):
    #import time
    #start = time.time()
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    #点群X, Y, Z, pointsを作成
    x = np.linspace(xmin, xmax, grid_step)
    y = np.linspace(ymin, ymax, grid_step)
    z = np.linspace(zmin, zmax, grid_step)

    X, Y, Z = np.meshgrid(x, y, z)

    # 格子点X, Y, Zをすべてfnにぶち込んでみる
    W = np.array([[fn(X[i][j], Y[i][j], Z[i][j]) for j in range(grid_step)] for i in range(grid_step)])
    # 変更前
    #W = fn(X, Y, Z)

    #Ｗが0に近いインデックスを取り出す
    index = np.where(np.abs(W)<=epsilon)
    index = [(index[0][i], index[1][i], index[2][i]) for i in range(len(index[0]))]
    #print(index)

    #ランダムにダウンサンプリング
    index = random.sample(index, int(len(index)*down_rate//1))


    #格子点から境界面(fn(x,y,z)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])
    pointZ = np.array([Z[i] for i in index])

    #points作成([[x1,y1,z1],[x2,y2,z2],...])    
    points = np.stack([pointX, pointY, pointZ])
    points = points.T

    #end = time.time()
    #print("time:{}s".format(end-start))

    return points

def ViewerInit(points, X, Y, Z, normals=[]):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #点群を描画
    ax.plot(X,Y,Z,marker="o",linestyle='None',color="white")

    """
    if len(normals) != 0:
        #法線を描画
        U, V, W = Disassemble(normals)
        ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True, color="blue")
    """

    max_p, min_p, _ = buildOBB(points)

    #OBBを描画
    OBBViewer(ax, max_p, min_p)

    return ax

#点群を入力としてOBBを描画する
def OBBViewer(ax, max_p, min_p):

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
                x, y, z = line(vertices[i], vertices[j])
                ax.plot(x,y,z,marker=".",color="orange")

    #OBBの頂点の1つ
    vert_max = min_p[0] + min_p[1] + max_p[2]
    vert_min = max_p[0] + max_p[1] + min_p[2]

    #xyzに分解
    Xmax, Ymax, Zmax = Disassemble(max_p)
    Xmin, Ymin, Zmin = Disassemble(min_p)


    #頂点なども描画
    ax.plot(Xmax,Ymax,Zmax,marker="X",linestyle="None",color="red")
    ax.plot(Xmin,Ymin,Zmin,marker="X",linestyle="None",color="blue")
    ax.plot([vert_max[0], vert_min[0]],[vert_max[1], vert_min[1]],[vert_max[2], vert_min[2]],marker="o",linestyle="None",color="black")

#点群を入力としてAABBを描画する
def AABBViewer(ax, max_p, min_p):

    # [xmax, xmin]と[ymax, ymin]の直積 <=> 頂点
    x_axis = [max_p[0], min_p[0]]
    y_axis = [max_p[1], min_p[1]]
    z_axis = [max_p[2], min_p[2]]

    vertices = np.asarray(list(itertools.product(x_axis, y_axis, z_axis)))

    #各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))


    #頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y, z = line(vertices[i], vertices[j])
                ax.plot(x,y,z,marker=".",color="orange")


# ラベルの色分け
def LabelViewer(ax, points, label_list, max_label):

    colorlist = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

    # ラベルの数
    label_num = np.max(label_list)

    # ラベルなしの点群を白でプロット

    X, Y, Z = Disassemble(points[np.where(label_list == 0)])
    ax.plot(X, Y, Z, marker=".",linestyle="None",color="white")


    for i in range(1, label_num+1):
        #同じラベルの点群のみにする
        same_label_points = points[np.where(label_list == i)]

        print("{}:{}".format(i, same_label_points.shape[0]))

        #plot
        X, Y, Z = Disassemble(same_label_points)
        if i == max_label:       
            ax.plot(X, Y, Z, marker="o",linestyle="None",color=colorlist[i%len(colorlist)])
        else:
            ax.plot(X, Y, Z, marker=".",linestyle="None",color=colorlist[i%len(colorlist)])

#陰関数のグラフ描画
#fn  ...fn(x, y, z) = 0の左辺
#AABB_size ...AABBの各辺をAABB_size倍する
def plot_implicit(ax, fn, points=None, AABB_size=2, bbox=(-2.5,2.5), contourNum=30):

    if points is not None:
        #AABB生成
        max_p, min_p = buildAABB(points)

        xmax, ymax, zmax = max_p[0], max_p[1], max_p[2]
        xmin, ymin, zmin = min_p[0], min_p[1], min_p[2]

        #AABBの各辺がAABB_size倍されるように頂点を変更
        xmax = xmax + (xmax - xmin)/2 * AABB_size
        xmin = xmin - (xmax - xmin)/2 * AABB_size
        ymax = ymax + (ymax - ymin)/2 * AABB_size
        ymin = ymin - (ymax - ymin)/2 * AABB_size
        zmax = zmax + (zmax - zmin)/2 * AABB_size
        zmin = zmin - (zmax - zmin)/2 * AABB_size

    else:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    A_X = np.linspace(xmin, xmax, 100) # resolution of the contour
    A_Y = np.linspace(ymin, ymax, 100)
    A_Z = np.linspace(zmin, zmax, 100)
    B_X = np.linspace(xmin, xmax, 15) # number of slices
    B_Y = np.linspace(ymin, ymax, 15)
    B_Z = np.linspace(zmin, zmax, 15)
    #A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B_Z: # plot contours in the XY plane
        X,Y = np.meshgrid(A_X, A_Y)
        Z = fn(X,Y,z)
        ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B_Y: # plot contours in the XZ plane
        X,Z = np.meshgrid(A_X, A_Z)
        Y = fn(X,y,Z)
        ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B_X: # plot contours in the YZ plane
        Y,Z = np.meshgrid(A_Y, A_Z)
        X = fn(x,Y,Z)
        ax.contour(X+x, Y, Z, [x], zdir='x')

    #(拡大した)AABBの範囲に制限
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

def plot_normal(ax, figure, X, Y, Z):
    #図形の方程式から点群を作る
    #points, X, Y, Z = MakePoints(figure.f_rep, epsilon=0.01)

    #法線
    normals = figure.normal(X, Y, Z)
    U, V, W = Disassemble(normals)

    #法線を描画
    ax.quiver(X, Y, Z, U, V, W,  length=0.1,color='red', normalize=True)