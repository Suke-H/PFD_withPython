import numpy as np
import re
from glob import glob
import csv
import os

import figure2d as F
from method2d import *
from TransPix import MakeOuterFrame
from MakeDataset import MakePointSet, MakeSign2D
from ClossNum import CheckClossNum3

def ConfusionLabeling(trueIndex, optiIndex):
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

def write_contour_dataset(fig_type, num, dir_path="data/dataset/tri/"):

    fig_list = []
    AABB_list = []
    trueIndex_list = []

    os.mkdir(dir_path)
    os.mkdir(dir_path+"points")

    for i in range(num):
        rate = Random(0.5, 1)
        N_rate = Random(0.5, 1)
        N = int(300*N_rate//1)
        # 2d図形点群作成
        fig, points, AABB, trueIndex = MakePointSet(fig_type, N, rate=rate)

        fig_list.append(fig.p)
        AABB_list.append(AABB)
        trueIndex_list.append(trueIndex)
        np.save(dir_path+"points/"+str(i), np.array(points))

    np.save(dir_path+"fig", np.array(fig_list))
    np.save(dir_path+"AABB", np.array(AABB_list))
    np.save(dir_path+"trueIndex", np.array(trueIndex_list))

def test_contour(fig_type, num, dir_path, out_path, csv_name):

    # 読み込み
    fig_list = np.load(dir_path+"fig.npy")
    AABB_list = np.load(dir_path+"AABB.npy")
    trueIndex_list = np.load(dir_path+"trueIndex.npy")
    points_paths = sorted(glob(dir_path + "points/**.npy"),\
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

    # 動作確認の画像出力用フォルダ作成
    os.mkdir(out_path)
    os.mkdir(out_path+"origin")
    os.mkdir(out_path+"dil")
    os.mkdir(out_path+"close")
    os.mkdir(out_path+"open")
    os.mkdir(out_path+"add")
    os.mkdir(out_path+"contour")

    csv_path = out_path + csv_name

    with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["AABB", "fig", "num", "rate=fig/AABB", "num*rate", "num/rate",\
                #  "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"
                ])

    for i in range(num):
        print("epoch:{}".format(i))

        # 読み込み
        points = np.load(points_paths[i])
        fig_p = fig_list[i]
        AABB = AABB_list[i]
        trueIndex = trueIndex_list[i]

        # 外枠作成
        out_points, out_area = MakeOuterFrame(points, out_path, i, 
                                dilate_size=50, close_size=20, open_size=70, add_size=20)

        # 輪郭抽出に失敗したら失敗を記録して次へ
        if out_points is None:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([-1])

            continue

        # 外枠内の点群だけにする
        inside = np.array([CheckClossNum3(points[i], out_points) for i in range(points.shape[0])])
        points = points[inside]

        ####################################################

        if fig_type==0:
            fig = F.circle(fig_p)
        elif fig_type==1:
            fig = F.tri(fig_p)
        else:
            fig = F.rect(fig_p)


        # AABBの面積、figの面積、pointsの数を記録
        umin, umax, vmin, vmax = AABB
        AABBArea = abs((umax-umin)*(vmax-vmin))
        figArea = fig.CalcArea()
        pointNum = points.shape[0]
        rate = figArea/AABBArea

        rec_list = [AABBArea, figArea, pointNum, rate, pointNum*rate, pointNum/rate]

        # # 混合行列の記録
        # confusionIndex = ConfusionLabeling(trueIndex, inside)

        # TP = np.count_nonzero(confusionIndex==1)
        # TN = np.count_nonzero(confusionIndex==2)
        # FP = np.count_nonzero(confusionIndex==3)
        # FN = np.count_nonzero(confusionIndex==4)

        # acc = (TP+TN)/(TP+TN+FP+FN)
        # prec = TP/(TP+FP)
        # rec = TP/(TP+FN)
        # F_measure = 2*prec*rec/(prec+rec)

        # rec_list.extend([TP, TN, FP, FN, acc, prec, rec, F_measure])

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rec_list)


def Record(fig_type, dir_path):

    # データセット読み込み
    fig_list = np.load(dir_path+"fig.npy")
    AABB_list = np.load(dir_path+"AABB.npy")
    outArea_list = np.load(dir_path+"outArea.npy")

    print("fig:{}".format(np.array(fig_list).shape))
    print("AABB:{}".format(np.array(AABB_list).shape))
    print("outArea:{}".format(np.array(outArea_list).shape))

    # points, outPointsはまずパスを読み込み
    points_paths = sorted(glob(dir_path + "points/**.npy"),\
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))
    outPoints_paths = sorted(glob(dir_path + "outPoints/**.npy"),\
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

    print(points_paths)
    print(outPoints_paths)

    num = len(fig_list)

    for i in range(num):

        # points, outPoints読み込み
        points = np.load(points_paths[i])
        outPoints = np.load(outPoints_paths[i])
        # 他も参照
        fig_p = fig_list[i]
        AABB = AABB_list[i]
        outArea = outArea_list[i]

        if fig_type==0:
            fig = F.circle(fig_p)
        elif fig_type==1:
            fig = F.tri(fig_p)
        else:
            fig = F.rect(fig_p)

        # AABBの面積、figの面積、pointsの数を記録
        umin, umax, vmin, vmax = AABB
        AABBArea = abs((umax-umin)*(vmax-vmin))
        figArea = fig.CalcArea()
        pointNum = points.shape[0]
        rate = figArea/AABBArea

        with open(dir_path+"rect_re.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([AABBArea, figArea, pointNum, rate, pointNum*rate, pointNum/rate])


def test(points, i):
    # out_points, out_area = MakeOuterFrame(sign2d, path=dir_path+"contour/"+str(i)+".png")
    out_points, out_area = MakeOuterFrame(points, "data/Contour/test/", i,
                            dilate_size=30, close_size=40, open_size=50, add_size=50)

# write_contour_dataset(2, 50, dir_path="data/Contour/rect300/")
test_contour(1, 50, dir_path="data/Contour/tri300/", out_path="data/Contour/triLast2/", csv_name="triLast.csv")