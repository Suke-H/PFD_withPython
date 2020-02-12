import numpy as np
import matplotlib.pyplot as plt

from method import *
from method2d import *
from Projection import Plane3DProjection

def TransThetaVector(p_goal, p_opt, fig_type, u_goal, v_goal, O_goal, u_opt, v_opt, O_opt):
    # 正三角形
    if fig_type==1:
        # 120度で循環させる
        t_goal = cycle(p_goal[4], np.pi*(2/3))
        t_opt = cycle(p_opt[4], np.pi*(2/3))

    # 長方形
    elif fig_type==2:
        # 基準(0度)は長辺が底辺にあるようにしたい
        # 基本wを底辺にしてるが、hの方が大きかったら90度増やすことでhを底辺にする

        if p_goal[3] < p_goal[4]:
            t_goal = cycle(p_goal[5]+np.pi/2, np.pi)

        else:
            t_goal = cycle(p_goal[5], np.pi)

        if p_opt[3] < p_opt[4]:
            t_opt = cycle(p_opt[5]+np.pi/2, np.pi)

        else:
            t_opt = cycle(p_opt[5], np.pi)
    
    print("t_goal:{}".format(t_goal))
    print("t_opt:{}".format(t_opt))

    # θをベクトルに変換
    # θ_optはoptのuv座標からgoalのuv座標へ変換する
    t_vec_goal = norm(np.array([np.cos(t_goal), np.sin(t_goal)]))
    t_vec_opt = norm(np.array([np.cos(t_opt), np.sin(t_opt)]))

    # xyz座標に変換
    t_xyz = t_vec_opt[0]*u_opt + t_vec_opt[1]*v_opt + O_opt

    return t_vec_goal, np.array([np.dot((t_xyz-O_goal), u_goal), np.dot((t_xyz-O_goal), v_goal)])

# fig_opt: 出力の図形
#    |
# p_goal: 入力の図形の3dパラメータ
# p_opt: 出力の図形の3dパラメータ

# この2つを比較
def Analysis(fig_opt, p_goal, fig_type, 
            input2d, input3d, 
            u_goal, v_goal, n_goal, O_goal, u_opt, v_opt, n_opt, O_opt):
    max_p, min_p, _, _, _ = buildAABB2d(input2d)
    AABB = [min_p[0], max_p[0], min_p[1], max_p[1]]
    out2d = InteriorPoints(fig_opt.f_rep, AABB, 500)
    p_opt, out3d = Plane3DProjection(out2d, fig_opt, u_opt, v_opt, O_opt)

    # プロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    X, Y, Z = Disassemble(input3d)
    OX, OY, OZ = Disassemble(out3d)
    ax.plot(X, Y, Z, marker=".", linestyle='None', color="red")
    ax.plot(OX, OY, OZ, marker=".", linestyle='None', color="blue")

    plt.show()

    # 比較
    # 位置・大きさ・平面のn・θベクトル(円は除く)
    pos = np.sqrt((p_opt[0]-p_goal[0])**2 + (p_opt[1]-p_goal[1])**2 + (p_opt[2]-p_goal[2])**2)
    n_angle = np.arccos(np.dot(n_opt, n_goal))

    print("pos:{}".format(pos))
    print("n_angle:{}".format(n_angle))

    if fig_type!=2:
        size = abs(p_opt[3] - p_goal[3])
        print("size:{}".format(size))
    else:
        size1 = abs(p_opt[3] - p_goal[3])
        size2 = abs(p_opt[4] - p_goal[4])
        print("size1:{}".format(size1))
        print("size2:{}".format(size2))

    if fig_type!=0:
        # θはベクトルにして角度を出す
        t_goal, t_opt = TransThetaVector(p_goal, p_opt, fig_type, u_goal, v_goal, O_goal, u_opt, v_opt, O_opt)

        print("t_vec_goal:{}".format(t_goal))
        print("t_vec_opt:{}".format(t_opt))

        theta = np.arccos(np.dot(t_goal, t_opt))
        print("theta:{}".format(theta))

    
    