import numpy as np
import matplotlib.pyplot as plt

from method2d import *

# 点pが輪郭点contour内にあるかの判定
def CheckClossNum(p, contour):
    # 輪郭点の辺をつくるため、
    # 輪郭点を[0,1,2,..n] -> [1,2,...,n,0]の順にした配列を作成
    order = [i for i in range(1, contour.shape[0])]
    order.append(0)
    contour2 = contour[order, :]

    # l: pから伸ばした左にx軸平行な半直線
    # 各辺とlの交差数をカウントする
    #judge_list = []
    crossCount = 0
    #for p, a, b in zip(points, contour, contour2):
    for a, b in zip(contour, contour2):
        # a,bのy座標がどちらもpのy座標より小さい, 大きい, a,bのx座標がどちらもpのx座標より大きい => 交差しない
        if (a[1]<p[1] and b[1]<p[1]) or (a[1]>p[1] and b[1]>p[1]) or (a[0]>p[0] and b[0]>p[0]):
            continue
        # a,bのx座標がどちらもpのx座標より小さい =>　交差する
        if a[0]<p[0] and b[0]<p[0]:
            crossCount+=1
            continue
        # lが直線として,辺との交点cのx座標を求める
        cx = (p[1]*(a[0]-b[0]) + a[1]*b[0] - a[0]*b[1]) / (a[1]-b[1])
        # cx < pxなら交差する
        if cx < p[0]:
            crossCount+=1

    # 交差数が偶数なら外、奇数なら内
    if crossCount%2 == 0:
        return False
    else:
        return True


# pointsの各点が輪郭点contour内にあるかの判定
def CheckClossNum2(points, contour):
    # 輪郭点の辺をつくるため、
    # 輪郭点を[0,1,2,..n] -> [1,2,...,n,0]の順にした配列を作成
    order = [i for i in range(1, contour.shape[0])]
    order.append(0)
    contour2 = contour[order, :]

    #ori_points = points[:, :]

    # 各点の半直線と各辺の交差を記録
    judge_list = np.zeros((contour.shape[0], points.shape[0]))
    #for p, a, b in zip(points, contour, contour2):
    for i, (a, b) in enumerate(zip(contour, contour2)):

        #points = ori_points[:, :]
        check_list = np.array([True for k in range(points.shape[0])])

        # a,bのy座標がどちらもpのy座標より小さい, 大きい, a,bのx座標がどちらもpのx座標より大きい => 交差しない
        drop_list = np.where(((a[1]<points[:, 1]) & (b[1]<points[:, 1])) | ((a[1]>points[:, 1]) & (b[1]>points[:, 1]))\
             | ((a[0]>points[:, 0]) & (b[0]>points[:, 0])), False, True)

        #print(drop_index)

        # 脱落
        check_list = check_list * drop_list

        # a,bのx座標がどちらもpのx座標より小さい =>　交差する
        closs_list = np.where((a[0]<points[:, 0]) & (b[0]<points[:, 0]), True, False)
        # チェックリストから抜けてる点は無視する
        closs_list = closs_list*check_list
        # 交差を記録
        judge_list[i, closs_list] = 1
        # 脱落
        drop_list = np.logical_not(closs_list)
        check_list = check_list*drop_list

        # lが直線として,辺との交点cのx座標を求める
        closs_list = np.where((points[:, 1]*(a[0]-b[0]) + a[1]*b[0] - a[0]*b[1]) / (a[1]-b[1]) < points[:, 0], True, False)
        # チェックリストから抜けてる点は無視する
        closs_list = closs_list*check_list
        # 交差を記録
        judge_list[i, closs_list] = 1

    inout_judge = np.sum(judge_list, axis=0)
    inout_judge = np.where(inout_judge % 2 == 1, True, False)

    return inout_judge


# 点pが輪郭点contour内にあるかの判定
def CheckClossNum3(p, contour):
    # 輪郭点の辺をつくるため、
    # 輪郭点を[0,1,2,..n] -> [1,2,...,n,0]の順にした配列を作成
    order = [i for i in range(1, contour.shape[0])]
    order.append(0)
    contour2 = contour[order, :]

    # l: pから右に伸ばした半直線
    # 各辺とlの交差数をカウントする
    crossCount = 0
    for a, b in zip(contour, contour2):
        # ルール1,2,3
        if (a[1]<=p[1] and b[1]>p[1]) or (a[1]>p[1] and b[1]<=p[1]):

            # ルール4: cx > pxなら交差する
            #print("a:{}, b:{}, p:{}".format(a,b,p))
            cx = (p[1]*(a[0]-b[0]) + a[1]*b[0] - a[0]*b[1]) / (a[1]-b[1])
            if cx > p[0]:
                crossCount+=1

    # 交差数が偶数なら外、奇数なら内
    if crossCount%2 == 0:
        return False
    else:
        return True

# import time

# n = 2000
# x = (np.random.rand(n) - 0.5)*2.5
# y = (np.random.rand(n) - 0.5)*2.5
# x1, y1 = [], []
    
# # define a polygon
# for i in range(5):
#     x1.append(np.cos(i*2.*np.pi/5.))
#     y1.append(np.sin(i*2.*np.pi/5.))

# # x1 = [-1,-1,1,1]
# # y1 = [1,-1,-1,1]

# # x = (np.random.rand(4) - 0.5)*2.5
# # y = np.array(y1[:])

# points = Composition2d(x, y)
# contour = Composition2d(x1, y1)

# print(points.shape)

# start = time.time()

# inside = np.array([CheckClossNum3(points[i], contour) for i in range(points.shape[0])])
# #inside = CheckClossNum2(points, contour)

# print(inside)

# end = time.time()

# print("time:{}s".format(end-start))

# x1.extend([x1[0]])
# y1.extend([y1[0]])

# plt.plot(x[inside], y[inside], marker=".",linestyle="None",color="red")
# plt.plot(x[inside==False], y[inside==False], marker=".",linestyle="None",color="black")
# plt.plot(x1, y1)
# plt.savefig("data/inpoly.png")
# plt.show()