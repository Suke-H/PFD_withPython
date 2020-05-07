import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import time
import os

from method2d import *
from method3d import *
from PreProcess import NormalEstimate
from Ransac import PlaneDetect
from Projection import Plane2DProjection, Plane3DProjection
from GA import EntireGA
from MakeDataset import MakeSign3D
from MakeOuterFrame import MakeOuterFrame
from Score import CalcScore

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

    # print用
    goal_list = []
    opti_list = []

    # 正解図形
    if fig_type == 0:
        fig = F2.circle(para2d)
        items = ["半径", "中心座標"]
        goal_list.append(para2d[2])
        
    elif fig_type == 1:
        fig = F2.tri(para2d)
        items = ["半径", "中心座標"]
        goal_list.append(para2d[2])

    else:
        fig = F2.rect(para2d)
        items = ["短辺", "長辺", "中心座標"]
        goal_list.append(para2d[2])
        goal_list.append(para2d[3])

    goal2d = ContourPoints2d(fig.f_rep, AABB2d, grid_step=1000)
    center, goal3d = Plane3DProjection(goal2d, para2d, u, v, O)
    goal_list.append(center)
    GX, GY, GZ = Disassemble3d(goal3d)

    # 検出図形
    if opti_fig_type == 0:
        opti_fig = F2.circle(opti_para2d)
        opti_list.append(opti_para2d[2])

    elif opti_fig_type == 1:
        opti_fig = F2.tri(opti_para2d)
        opti_list.append(opti_para2d[2])

    else:
        opti_fig = F2.rect(opti_para2d)
        opti_list.append(opti_para2d[2])
        opti_list.append(opti_para2d[3])

    opti2d = ContourPoints2d(opti_fig.f_rep, opti_AABB2d, grid_step=1000)
    opti_center, opti3d = Plane3DProjection(opti2d, opti_para2d, opti_u, opti_v, opti_O)
    opti_list.append(opti_center)
    OX, OY, OZ = Disassemble3d(opti3d)

    # パラメータをprint
    for (item, goal, opti) in zip(items, goal_list, opti_list):
        print("{}: 正解図形{}, 検出図形{} (mm)".format(item, goal, opti))

    # 平面の角度差 算出
    n_goal = np.array([plane_para[0], plane_para[1], plane_para[2]])
    n_opt = np.array([opti_plane_para[0], opti_plane_para[1], opti_plane_para[2]])
    angle = np.arccos(np.dot(n_opt, n_goal))
    angle = angle / np.pi * 180

    # 0と180が角度差最小、90が最大なのでmin(angle, 180-angle)を表示
    angle = min(angle, 180-angle)
    print("2つの図形のある平面の角度差:{}°\n".format(angle))
    
    print("点群、正解図形、検出図形を表示")
    print("(オレンジが点群、赤が正解図形、青が検出図形)")

    # プロット
    ax.plot(GX, GY, GZ, marker=".", linestyle='None', color="red")
    ax.plot(OX, OY, OZ, marker=".", linestyle='None', color="dodgerblue")
    ax.plot([center[0]], [center[1]], [center[2]], marker="o", linestyle='None', color="red")
    ax.plot([opti_center[0]], [opti_center[1]], [opti_center[2]], marker="o", linestyle='None', color="dodgerblue")

    plt.show()
    plt.close()   

def simulation(sign_type, scale, density, noise_rate, out_path):
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
        os.mkdir(out_path+"view_data")

    os.mkdir(out_path+"view_data/"+str(i))

    # シミュレーションにより点群生成
    para2d, fig_type, AABB2d, plane_para, points3d, trueIndex, u, v, O = MakeSign3D(sign_type, scale, density, noise_rate)
    print("点群数: {}".format(points3d.shape[0]))

    if sign_type == 0:
        fig_type = 0
    elif sign_type == 1:
        fig_type = 1
    else:
        fig_type = 2

    # AABB(Axis Aligned Bounding Box)の対角線の長さ抽出
    _, _, l = buildAABB3d(points3d)

    # 法線推定
    normals = NormalEstimate(points3d)

    print("平面検出 開始")
    start = time.time()

    # 平面検出
    opti_plane, fit_index, fit_num = PlaneDetect(points3d, normals, epsilon=0.01*l, alpha=np.pi/8)

    print("平面にフィットした点群数: {}".format(fit_num))
    end = time.time()
    
    print("平面検出 終了\ntime: {}s".format(end-start))
    print("\n==========================================\n")

    # 点群2D変換
    _, points2d, opti_u, opti_v, opti_O = Plane2DProjection(points3d[fit_index], opti_plane)
    opti_plane_para = opti_plane.p

    print("外枠生成 開始")
    start = time.time()

    # 外枠生成
    out_points, points2d, out_area = MakeOuterFrame(points2d, out_path, i, 
                                    dilate_size=50, close_size=20, open_size=70, add_size=20)

    end = time.time()
    print("外枠生成 終了\ntime: {}s".format(end-start))
    print("\n==========================================\n")

    # GAにより最適パラメータ出力
    best, opti_fig_type = EntireGA(points2d, out_points, out_area, CalcScore, out_path, i)


    # 検出図形の中心座標を3次元に射影
    opti_para2d = best.figure.p
    #########################################################################

    # 2D点群でのAABB生成
    max_p, min_p, _, _ = buildAABB2d(points2d)
    opti_AABB2d = [min_p[0], max_p[0], min_p[1], max_p[1]]

    print("\n==========================================\n")

    # 表示
    View(points3d, trueIndex,  # 点群
        para2d, fig_type, plane_para, u, v, O, AABB2d, # 正解図形
        opti_para2d, opti_fig_type, opti_plane_para, # 検出図形
        opti_u, opti_v, opti_O, opti_AABB2d)

    # 後で表示するために保存
    np.save(out_path+"view_data/"+str(i)+"/points3d", points3d)
    np.save(out_path+"view_data/"+str(i)+"/trueIndex", trueIndex)

    np.save(out_path+"view_data/"+str(i)+"/para2d", np.array(para2d))
    np.save(out_path+"view_data/"+str(i)+"/plane_para", np.array(plane_para))
    np.save(out_path+"view_data/"+str(i)+"/u", u)
    np.save(out_path+"view_data/"+str(i)+"/v", v)
    np.save(out_path+"view_data/"+str(i)+"/O", O)
    np.save(out_path+"view_data/"+str(i)+"/AABB2d", np.array(AABB2d))

    np.save(out_path+"view_data/"+str(i)+"/opti_para2d", np.array(opti_para2d))
    np.save(out_path+"view_data/"+str(i)+"/opti_plane_para", np.array(opti_plane_para))
    np.save(out_path+"view_data/"+str(i)+"/opti_u", opti_u)
    np.save(out_path+"view_data/"+str(i)+"/opti_v", opti_v)
    np.save(out_path+"view_data/"+str(i)+"/opti_O", opti_O)
    np.save(out_path+"view_data/"+str(i)+"/opti_AABB2d", np.array(opti_AABB2d))

def review(root_path, i):
    """
    一度平面図形検出した結果を3Dで再表示する

    root_path: out_pathと同じパスにする
    i: 何番目の検出結果かを選択
    """
    # 読み込み
    points3d = np.load(root_path+"view_data/"+str(i)+"/points3d.npy")
    trueIndex = np.load(root_path+"view_data/"+str(i)+"/trueIndex.npy")

    para2d = np.load(root_path+"view_data/"+str(i)+"/para2d.npy")
    plane_para = np.load(root_path+"view_data/"+str(i)+"/plane_para.npy")
    u = np.load(root_path+"view_data/"+str(i)+"/u.npy")
    v = np.load(root_path+"view_data/"+str(i)+"/v.npy")
    O = np.load(root_path+"view_data/"+str(i)+"/O.npy")
    AABB2d = np.load(root_path+"view_data/"+str(i)+"/AABB2d.npy")

    opti_para2d = np.load(root_path+"view_data/"+str(i)+"/opti_para2d.npy")
    opti_plane_para = np.load(root_path+"view_data/"+str(i)+"/opti_plane_para.npy")
    opti_u = np.load(root_path+"view_data/"+str(i)+"/opti_u.npy")
    opti_v = np.load(root_path+"view_data/"+str(i)+"/opti_v.npy")
    opti_O = np.load(root_path+"view_data/"+str(i)+"/opti_O.npy")
    opti_AABB2d = np.load(root_path+"view_data/"+str(i)+"/opti_AABB2d.npy")

    if len(para2d) == 3:
        fig_type = 0
    elif len(para2d) == 4:
        fig_type = 1
    else:
        fig_type = 2

    if len(opti_para2d) == 3:
        opti_fig_type = 0
    elif len(opti_para2d) == 4:
        opti_fig_type = 1
    else:
        opti_fig_type = 2

    # 表示
    View(points3d, trueIndex,  # 点群
        para2d, fig_type, plane_para, u, v, O, AABB2d, # 正解図形
        opti_para2d, opti_fig_type, opti_plane_para, # 検出図形
        opti_u, opti_v, opti_O, opti_AABB2d)

def main():
    """
    シミュレーション点群を生成し、それを入力に平面図形を検出

    <引数>
    sign_type: 
    0: 半径0.3mの円
    1: 1辺0.8mの正三角形
    2: 1辺0.9mの正方形
    3. 1辺0.45mのひし形(てか正方形)
    4. 1辺が0.05～1のどれかの長方形

    scale: sign_typeのスケールを標準とした倍率
    noise_rate: 全点群数に対するノイズ点群の割合

    out_path: 出力先のフォルダパス
    
    <出力>
    origin: 元画像(2D点群を画像に変換したもの)
    dil: 膨張演算
    open: オープニング演算
    close: クロージング演算
    add: 膨張演算
    contour: 外枠の生成結果(オレンジが点群、赤が外枠)
    GA: GA図形検出の結果(オレンジが点群、赤が推定図形)
    view_data: 3Dグラフで再表示するための保存フォルダ

    """

    # 出力先のフォルダパス
    out_path = "data/sim2/"

    sign_type, scale, density, noise_rate = 0, 1, 2500, 0.2

    # 平面図形検出
    simulation(sign_type, scale, density, noise_rate, out_path)

    # # 再表示(第二引数：何番目の検出結果を表示するか(0始まり))
    # review(out_path, 2)

if __name__ == "__main__":
    main()
