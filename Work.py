import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import time
import os

from method2d import *
from method3d import *
from PreProcess import NormalEstimate
from Ransac import Ransac
from Projection import Plane2DProjection, Plane3DProjection
from GA import EntireGA
from MakeDataset import MakePointSet, MakePointSet3D, MakeSign3D
from MakeOuterFrame import MakeOuterFrame
from CrossNum import CheckCrossNum
from Score import CalcScore

def SelectIndex(index1, index2):
    """
    点群-> [平面検出] -> index1 -> [外枠生成] -> index2

    index1, 2には生き残った点群のインデックスが格納されてるので、
    最終的に生き残るインデックスのリストを出力

    # 入力点群: [x0,x1,x2,x3,x4,x5,x6,x7]
    # index1:  [ 1, 1, 1, 1, 0, 1, 1, 0]
    #                     ↓
    # 抽出:    [x0,x1,x2,x3,   x5,　x6 ]
    #                     ↓
    # index2:  [ 1, 1, 1, 1,    0, 1   ]
    # 抽出:    [x0,x1,x2,x3,     　x6  ]
    #                     ↓
    # 出力: 　 [ 1, 1, 1, 1, 0, 0, 1, 0]

    """

    result = []
    j = 0
    for i in index1:
        if i == False:
            result.append(False)

        else:
            if index2[j] == True:
                result.append(True)
            else:
                result.append(False)
            j+=1

    return np.array(result)


def ConfusionLabeling(trueIndex, optiIndex):

    """
    trueIndex, optiIndex には点群の各点に対応して 0:ノイズ、1:実際の点群 が格納されている
    (trueIndex: 正解, optiIndex: 推定結果)

    trueIndex, optiIndexにより混合行列を生成
    (点群の各点に対応してTP:1, TN:2, FP:3, FN:4を格納したリストを出力)

       true  opti
    TP: 1  ->  1
    TF: 0  ->  0
    FP: 0  ->  1
    FN: 1  ->  0
    """

    confusionIndex = []

    for (true, opti) in zip(trueIndex, optiIndex):
        if true==True and opti==True:
            confusionIndex.append(1)
        elif true==False and opti==False:
            confusionIndex.append(2)
        elif true==False and opti==True:
            confusionIndex.append(3)
        elif true==True and opti==False:
            confusionIndex.append(4)

    return np.array(confusionIndex)

def View(points3d, trueIndex,  # 点群
        para2d, fig_type, plane_para, u, v, O, AABB2d, # 正解図形
        opti_para2d, opti_fig_type, opti_plane_para, opti_u, opti_v, opti_O, opti_AABB2d): # 検出図形

    """ 点群、正解図形、検出図形を表示 """

    # 点群プロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    X, Y, Z = Disassemble3d(points3d[trueIndex])
    NX, NY, NZ = Disassemble3d(points3d[trueIndex==False])
    ax.plot(X, Y, Z, marker=".", linestyle='None', color="orange")
    ax.plot(NX, NY, NZ, marker=".", linestyle='None', color="blue")

    # 正解図形、検出図形の境界線を描画

    # 正解図形
    if fig_type == 0:
        fig = F2.circle(para2d)
    elif fig_type == 1:
        fig = F2.tri(para2d)
    else:
        fig = F2.rect(para2d)

    goal2d = ContourPoints2d(fig.f_rep, AABB2d, grid_step=1000)
    center, goal3d = Plane3DProjection(goal2d, para2d, u, v, O)
    GX, GY, GZ = Disassemble3d(goal3d)

    # 検出図形
    if opti_fig_type == 0:
        opti_fig = F2.circle(opti_para2d)
    elif opti_fig_type == 1:
        opti_fig = F2.tri(opti_para2d)
    else:
        opti_fig = F2.rect(opti_para2d)

    opti2d = ContourPoints2d(opti_fig.f_rep, opti_AABB2d, grid_step=1000)
    opti_center, opti3d = Plane3DProjection(opti2d, opti_para2d, opti_u, opti_v, opti_O)
    OX, OY, OZ = Disassemble3d(opti3d)

    # プロット
    ax.plot(GX, GY, GZ, marker=".", linestyle='None', color="red")
    ax.plot(OX, OY, OZ, marker=".", linestyle='None', color="dodgerblue")
    ax.plot([center[0]], [center[1]], [center[2]], marker="o", linestyle='None', color="red")
    ax.plot([opti_center[0]], [opti_center[1]], [opti_center[2]], marker="o", linestyle='None', color="dodgerblue")

    plt.show()
    plt.close()   

def simulation(sign_type, scale, density, noise_rate, error_rate, error_step, out_path):
    """  シミュレーションにより生成した点群から平面図形検出 """

    # 一度実行したことあればiだけ定義
    # (iには何度目かの情報が入る)
    if os.path.exists(out_path + "contour"):
        i = len(glob(out_path + "contour/**"))

    # 初めての場合は一度フォルダ作成
    # (out_pathのフォルダを作成し、その中に複数のフォルダを作成する。)
    else:
        i = 0

        # out_pathにフォルダが存在しないなら作成
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        os.mkdir(out_path+"origin")
        os.mkdir(out_path+"dil")
        os.mkdir(out_path+"close")
        os.mkdir(out_path+"open")
        os.mkdir(out_path+"add")
        os.mkdir(out_path+"contour")
        os.mkdir(out_path+"GA")

    # シミュレーションにより点群生成
    center, para2d, plane_para, points3d, AABB3d, AABB2d, trueIndex, u, v, O = MakeSign3D(sign_type, scale, density, noise_rate, error_rate, error_step)

    if sign_type == 0:
        fig_type = 0
    elif sign_type == 1:
        fig_type = 1
    else:
        fig_type = 2

    if fig_type != 2:
        items = ["type", "pos", "size", "angle", "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"]

    else:
        items = ["type", "pos", "size1", "size2", "angle", "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"]

    rec_list = []

    # AABB(Align-Axis Bounding Box)の対角線の長さ抽出
    _, _, l = buildAABB3d(points3d)

    # 法線推定
    normals = NormalEstimate(points3d)

    start = time.time()

    # 平面検出
    opti_plane, fit_index, _ = Ransac(points3d, normals, epsilon=0.01*l, alpha=np.pi/8)

    end = time.time()
    print("平面検出:{}m".format((end-start)/60))

    # 平面にフィットした点のインデックス(points3dに対応したインデックスのリスト)
    index1 = np.array([True if i in fit_index else False for i in range(points3d.shape[0])])

    # 点群2D変換
    _, points2d, opti_u, opti_v, opti_O = Plane2DProjection(points3d[fit_index], opti_plane)
    opti_plane_para = opti_plane.p

    start = time.time()

    # 外枠生成
    out_points, out_area = MakeOuterFrame(points2d, out_path, i, 
                                        dilate_size=50, close_size=20, open_size=70, add_size=20)

    end = time.time()
    print("外枠生成:{}m".format((end-start)/60))

    # 輪郭抽出に失敗したら終了
    if out_points is None:
        rec_list.extend(["×" for i in range(len(items))])

        return items, rec_list

    # 外枠内の点群だけにする
    index2 = np.array([CheckCrossNum(points2d[i], out_points) for i in range(points2d.shape[0])])
    points2d = points2d[index2]

    # GAにより最適パラメータ出力
    best, opti_fig_type = EntireGA(points2d, out_points, out_area, CalcScore, out_path, i)


    # 検出図形の中心座標を3次元に射影
    opti_para2d = best.figure.p
    opti_center, _ = Plane3DProjection(points2d, opti_para2d, opti_u, opti_v, opti_O)
    print(opti_fig_type, opti_para2d)

    #########################################################################

    # 2D点群でのAABB生成
    max_p, min_p, _, _, _ = buildAABB2d(points2d)
    opti_AABB2d = [min_p[0], max_p[0], min_p[1], max_p[1]]

    # 表示
    View(points3d, trueIndex,  # 点群
        para2d, fig_type, plane_para, u, v, O, AABB2d, # 正解図形
        opti_para2d, opti_fig_type, opti_plane_para, # 検出図形
        opti_u, opti_v, opti_O, opti_AABB2d)

    # 図形の種類、位置、大きさ、平面、形の5つの指標で評価

    # 図形の種類：図形の種類が一致しているか

    # 一致してなかったら(type=-1)これ以上評価しない
    if fig_type != opti_fig_type:
        rec_list.append(-1)
        rec_list.extend(["×" for i in range(len(items)-1)])

        return items, rec_list

    # 一致した場合(type=1)評価を続ける
    rec_list.append(1)

    # 位置: 3次元座標上の中心座標の距離
    pos = LA.norm(center - opti_center)
    rec_list.append(pos)

    # 大きさ: 円と三角形ならr, 長方形なら長辺と短辺
    if fig_type != 2:
        size = abs(para2d[2] - opti_para2d[2])
        rec_list.append(size)

    else:
        if para2d[2] > para2d[3]:
            long_edge, short_edge = para2d[2], para2d[3]
        else:
            long_edge, short_edge = para2d[3], para2d[2]

        if opti_para2d[2] > opti_para2d[3]:
            opti_long_edge, opti_short_edge = opti_para2d[2], opti_para2d[3]
        else:
            opti_long_edge, opti_short_edge = opti_para2d[3], opti_para2d[2]

        size1 = abs(long_edge - opti_long_edge)
        size2 = abs(short_edge - opti_short_edge)
        rec_list.append(size1)
        rec_list.append(size2)

    # 平面：平面の法線の角度
    n_goal = np.array([plane_para[0], plane_para[1], plane_para[2]])
    n_opt = np.array([opti_plane_para[0], opti_plane_para[1], opti_plane_para[2]])
    angle = np.arccos(np.dot(n_opt, n_goal))
    angle = angle / np.pi * 180
    rec_list.append(angle)

    # 形: 混合行列で見る
    X, Y = Disassemble2d(points2d)
    index3 = (best.figure.f_rep(X, Y) >= 0)
    estiIndex = SelectIndex(index1, SelectIndex(index2, index3))
    confusionIndex = ConfusionLabeling(trueIndex, estiIndex)

    TP = np.count_nonzero(confusionIndex==1)
    TN = np.count_nonzero(confusionIndex==2)
    FP = np.count_nonzero(confusionIndex==3)
    FN = np.count_nonzero(confusionIndex==4)

    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)
    F_measure = 2*prec*rec/(prec+rec)

    rec_list.extend([TP, TN, FP, FN, acc, prec, rec, F_measure])

    return items, rec_list

# test3D(1, 1, 500, 10, out_path="data/EntireTest/test2/")
# write_data3D(0, 500, 0.2, 0.4, 0.005, 20, "data/dataset/3D/4/")
# write_any_data3d(20, "data/dataset/3D/square0.45_noise/")
# use_any_data3D("data/dataset/3D/square0.45_noise/", "data/EntireTest/square0.45_noise/")
# CheckView(4, 0, 0, "data/dataset/3D/0_500_test/", "data/EntireTest/testtest/")
# use_any_data3D("data/dataset/rect0.9_noise/", "data/result/rect0.9_noise/")

sign_type, scale, density, noise_rate, error_rate, error_step  = 1, 1, 2500, 0.2, 0, 0
out_path = "data/result/sim2/"
items, rec_list = simulation(sign_type, scale, density, noise_rate, error_rate, error_step, out_path)
print(items, rec_list)
for item, rec in zip(items, rec_list):
    print("{}: {}".format(item, rec))