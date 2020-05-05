import numpy as np

from method2d import *
import figure2d as F

def CheckIB(fig, fig_type, max_p, min_p, l):
    """ AABB内に中心座標がある かつ AABBに対して図形が小さすぎないし大きすぎないかチェック """
    # 円
    if fig_type==0:
        x, y, r = fig
        w, h = l/2, l/2
    # 正三角形
    elif fig_type==1:
        x, y, r, _= fig
        w, h = l/2, l/2
    # 長方形
    elif fig_type==2:
        x, y, w, h, _ = fig
        r = l/2

    if (min_p[0] < x < max_p[0]) and (min_p[1] < y < max_p[1]) and (l*0.2 < r < l) and (l*0.2 < w < l) and (l*0.2 < h < l):
        return True

    else:
        return False

def CalcScore(points, out_contour, out_area, figure, flag=False):
    """
    Score = inの密度 - outの密度
    ただしAABBに対して図形(=inの面積)が小さすぎ&大きすぎたら大幅減点
    """
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    # AABB内にあるのかチェック
    if not CheckIB(figure.p, fig, max_p, min_p, l):
        return -100

    ########################################################

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W>=0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    
    Cout = points.shape[0] - Cin

    # Aout = out_shapeの面積 - Ain
    Aout = out_area - Ain

    if Aout < 0:
        return -100

    if flag==True:
        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin/Ain - Cout/Aout