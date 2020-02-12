import numpy as np
import csv
from glob import glob
import re

from method import *
from method2d import *
from Projection import Plane3DProjection

# def CheckAnyView(root_path, out_root_path, fig_type, out_fig_type):

#     dir_paths = sorted(glob(root_path + "**"),\
#                         key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

#     out_paths = sorted(glob(out_root_path + "**"),\
#                         key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

#     for i, (dir_path, out_path) in enumerate(zip(dir_paths, out_paths)):

#         CheckView(dir_path, out_path, fig_type, out_fig_type)

def CheckView(dir_path, out_path, miss_list=[]):
    center_list = np.load(dir_path+"center.npy")
    para2d_list = np.load(dir_path+"para2d.npy")
    points3d_list = np.load(dir_path+"points3d.npy")
    AABB2d_list = np.load(dir_path+"AABB2d.npy")
    trueIndex_list = np.load(dir_path+"trueIndex.npy")
    plane_list = np.load(dir_path+"plane.npy")
    u_list = np.load(dir_path+"u.npy")
    v_list = np.load(dir_path+"v.npy")
    O_list = np.load(dir_path+"O.npy")

    opti_para2d_list = np.load(out_path+"opti_para2d.npy")
    opti_AABB2d_list = np.load(out_path+"opti_AABB2d.npy")
    opti_plane_list = np.load(out_path+"opti_plane.npy")
    opti_u_list = np.load(out_path+"opti_u.npy")
    opti_v_list = np.load(out_path+"opti_v.npy")
    opti_O_list = np.load(out_path+"opti_O.npy")

    j = 0

    for i in range(opti_para2d_list.shape[0]):

        while j in miss_list:
            j += 1

        if len(para2d_list[i]) == 3:
            fig_type = 0
        elif len(para2d_list[i]) == 4:
            fig_type = 1
        else:
            fig_type = 2

        if len(opti_para2d_list[i]) == 3:
            opti_fig_type = 0
        elif len(opti_para2d_list[i]) == 4:
            opti_fig_type = 1
        else:
            opti_fig_type = 2

        pos = ViewTest(points3d_list[j], trueIndex_list[j],  # 点群
                    para2d_list[j], fig_type, plane_list[j], u_list[j], v_list[j], O_list[j], AABB2d_list[j], # 正解図形
                    opti_para2d_list[i], opti_fig_type, opti_plane_list[i], opti_u_list[i], opti_v_list[i], opti_O_list[i], opti_AABB2d_list[i])# 検出図形

        with open(out_path+"pos.csv", 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([pos])

        j += 1

def ViewTest(points3d, trueIndex,  # 点群
            para2d, fig_type, plane_para, u, v, O, AABB2d, # 正解図形
            opti_para2d, opti_fig_type, opti_plane_para, opti_u, opti_v, opti_O, opti_AABB2d): # 検出図形

    
    # プロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    X, Y, Z = Disassemble(points3d[trueIndex])
    NX, NY, NZ = Disassemble(points3d[trueIndex==False])
    ax.plot(X, Y, Z, marker=".", linestyle='None', color="orange")
    ax.plot(NX, NY, NZ, marker=".", linestyle='None', color="blue")

    # 図形境界線作成
    if fig_type == 0:
        fig = F.circle(para2d)
    elif fig_type == 1:
        fig = F.tri(para2d)
    else:
        fig = F.rect(para2d)

    if opti_fig_type == 0:
        opti_fig = F.circle(opti_para2d)
    elif opti_fig_type == 1:
        opti_fig = F.tri(opti_para2d)
    else:
        opti_fig = F.rect(opti_para2d)

    goal2d = ContourPoints(fig.f_rep, AABB2d, grid_step=1000)
    center, goal3d = Plane3DProjection(goal2d, para2d, u, v, O)
    opti2d = ContourPoints(opti_fig.f_rep, opti_AABB2d, grid_step=1000)
    opti_center, opti3d = Plane3DProjection(opti2d, opti_para2d, opti_u, opti_v, opti_O)

    print(np.linalg.norm(center-opti_center))

    GX, GY, GZ = Disassemble(goal3d)
    OX, OY, OZ = Disassemble(opti3d)

    ax.plot(GX, GY, GZ, marker=".", linestyle='None', color="red")
    ax.plot(OX, OY, OZ, marker=".", linestyle='None', color="dodgerblue")
    ax.plot([center[0]], [center[1]], [center[2]], marker="o", linestyle='None', color="red")
    ax.plot([opti_center[0]], [opti_center[1]], [opti_center[2]], marker="o", linestyle='None', color="dodgerblue")

    plt.show()
    plt.close()

    return np.linalg.norm(center-opti_center)*1000

    

# CheckView(0, "data/dataset/3D/SET_NOISE/1/", "data/EntireTest/SET_NOISE/1/", 0, 0)

# CheckView("data/基本形/dataset/triangle/1/", "data/基本形/result/triangle/2000/", miss_list=[])
CheckView("data/ノイズ/dataset/circle_noise/5/", "data/ノイズ/result/circle_noise/5/", miss_list=[])