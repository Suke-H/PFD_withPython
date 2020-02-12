from GAtest import *
import open3d
import os

import time

def write_any_data3d(num, root_path):
    sign_type_set = np.array([3])
    scale_set = np.array([1])
    density_set = np.array([3000])
    noise_rate_set = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    error_rate_set = np.array([0])
    error_step_set = np.array([0])
    # # noise_rate_set = np.array([0.2, 0.4])
    # # error_rate_set = np.array([0.2, 0.4])
    # # error_step_set = np.array([0.001, 0.005])

    para_set = np.array(list(itertools.product(sign_type_set, scale_set, density_set, noise_rate_set, error_rate_set, error_step_set)))

    # para_set = np.array([[0, 1, 2500, 0.1, 0, 0], [0, 1, 2500, 0.3, 0, 0], [0, 1, 2500, 0.5, 0, 0], \
                        # [1, 1, 2500, 0.1, 0, 0], [1, 1, 2500, 0.3, 0, 0], [1, 1, 2500, 0.5, 0, 0],\
                        # [2, 1, 1000, 0.1, 0, 0], [2, 1, 1000, 0.3, 0, 0], [2, 1, 1000, 0.5, 0, 0]])

    for i, para in enumerate(para_set):
        sign_type, scale, density, noise_rate, error_rate, error_step = para

        write_data3D(int(sign_type), scale, density, noise_rate, error_rate, error_step, num, root_path+str(i)+"/")


def write_data3D(sign_type, scale, density, noise_rate, error_rate, error_step, num, dir_path):

    # 保存リストとフォルダ作成

    # os.mkdir(dir_path+str(sign_type)+"_"+str(N))
    os.mkdir(dir_path)

    path_w = dir_path+"para.txt"
    para_config = np.array([sign_type, scale, density, noise_rate, error_rate, error_step])
    np.savetxt(path_w, para_config, fmt="%0.3f", delimiter=",")

    center_list = []
    para2d_list = []
    plane_list = []
    points3d_list = []
    AABB2d_list = []
    trueIndex_list = []
    u_list = []
    v_list = []
    O_list = []

    # パラメータセットの各要素でnum個作成
    for i in range(num):

        center, para2d, plane_para, points3d, AABB3d, AABB2d, trueIndex, u, v, O = MakeSign3D(sign_type, scale, density, noise_rate, error_rate, error_step)

        center_list.append(center)
        para2d_list.append(para2d)
        points3d_list.append(points3d)
        AABB2d_list.append(AABB2d)
        trueIndex_list.append(trueIndex)
        plane_list.append(plane_para)
        u_list.append(u)
        v_list.append(v)
        O_list.append(O)

    np.save(dir_path+"center", np.array(center_list))
    np.save(dir_path+"para2d", np.array(para2d_list))
    np.save(dir_path+"points3d", np.array(points3d_list))
    np.save(dir_path+"AABB2d", np.array(AABB2d_list))
    np.save(dir_path+"trueIndex", np.array(trueIndex_list))
    np.save(dir_path+"plane", np.array(plane_list))
    np.save(dir_path+"u", np.array(u_list))
    np.save(dir_path+"v", np.array(v_list))
    np.save(dir_path+"O", np.array(O_list))
   
def use_any_data3D(root_path, out_root_path):

    start = time.time()

    folder_paths = sorted(glob(root_path + "**"),\
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

    # a = [i for i in range(len(folder_paths))]

    # for i, folder in zip(a[from_num:to_num], folder_paths[from_num:to_num]):
    for i, folder in enumerate(folder_paths):
        print("="*60)
        print("folder:{}".format(i))

        with open(folder+"/para.txt") as f:
            sign_type = f.readline()
            sign_type = int(float(sign_type))

            use_data3D(sign_type, folder+"/", out_root_path+str(i)+"/")
        
    end = time.time()
    print("total_time:{}m".format((end-start)/60))


def use_data3D(sign_type, dir_path, out_path):

    os.mkdir(out_path)
    os.mkdir(out_path+"origin")
    os.mkdir(out_path+"origin2")
    os.mkdir(out_path+"dil")
    os.mkdir(out_path+"close")
    os.mkdir(out_path+"open")
    os.mkdir(out_path+"add")
    os.mkdir(out_path+"contour")
    os.mkdir(out_path+"outPoints")
    os.mkdir(out_path+"points")
    os.mkdir(out_path+"GA")

    center_list = np.load(dir_path+"center.npy")
    para2d_list = np.load(dir_path+"para2d.npy")
    points3d_list = np.load(dir_path+"points3d.npy")
    AABB2d_list = np.load(dir_path+"AABB2d.npy")
    trueIndex_list = np.load(dir_path+"trueIndex.npy")
    plane_list = np.load(dir_path+"plane.npy")
    u_list = np.load(dir_path+"u.npy")
    v_list = np.load(dir_path+"v.npy")
    O_list = np.load(dir_path+"O.npy")

    opti_para2d_list = []
    opti_plane_list = []
    opti_u_list = []
    opti_v_list = []
    opti_O_list = []
    opti_AABB2d_list = []


    if sign_type == 0:
        fig_type = 0
    elif sign_type == 1:
        fig_type = 1
    else:
        fig_type = 2

    if fig_type != 2:
        rec_list = ["type", "pos", "size", "angle", "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"]

    else:
        rec_list = ["type", "pos", "size1", "size2", "angle", "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"]

    with open(out_path+"test.csv", 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rec_list)

    ###############################################################################


    for i, (center, para2d, points3d, AABB2d, trueIndex, plane_para, u, v, O) in \
        enumerate(zip(center_list, para2d_list, points3d_list, AABB2d_list, trueIndex_list, plane_list, u_list, v_list, O_list)):

        print("epoch:{}".format(i))
        # 2d図形点群作成
        #center, para2d, plane_para, points3d, AABB, trueIndex = MakePointSet3D(fig_type, 500, rate=0.8)

        start = time.time()

        _, _, l = buildAABB(points3d)

        # 法線推定
        normals = NormalEstimate(points3d)

        # 平面検出, 2d変換
        points2d, opti_plane, opti_u, opti_v, opti_O, index1 = PlaneDetect(points3d, normals, epsilon=0.01*l, alpha=np.pi/8)

        opti_plane_para = opti_plane.p

        end = time.time()
        # print("平面検出:{}m".format((end-start)/60))

        start = time.time()

        # 外枠作成
        # out_points, out_area = MakeOuterFrame2(points2d, out_path, i, 
        #                         dilate_size1=30, close_size1=20, open_size1=50, add_size1=50,
        #                         dilate_size2=30, close_size2=0, open_size2=50, add_size2=5, goalDensity=10000)

        out_points, out_area = MakeOuterFrame(points2d, out_path, i, 
                              dilate_size=50, close_size=20, open_size=70, add_size=20)

        end = time.time()
        # print("輪郭抽出:{}m".format((end-start)/60))

        # 輪郭抽出に失敗したら終了
        if out_points is None:
            with open(out_path+"test.csv", 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([-2])

            continue

        # 外枠内の点群だけにする
        index2 = np.array([CheckClossNum3(points2d[i], out_points) for i in range(points2d.shape[0])])
        #inside = CheckClossNum2(sign2d, out_points)
        points2d = points2d[index2]
        max_p, min_p, _, _, _ = buildAABB2d(points2d)
        opti_AABB2d = [min_p[0], max_p[0], min_p[1], max_p[1]]

        start = time.time()

        # GAにより最適パラメータ出力
        #best = GA(sign)
        # print("GA開始")
        best_circle = EntireGA(points2d, out_points, out_area, CalcIoU1, out_path+"/GA/circle"+str(i)+".png", fig_type, 
                                fig=[0], n_epoch=100, N=100, add_num=30, half_reset_num=10, all_reset_num=5)

        best_tri = EntireGA(points2d, out_points, out_area, CalcIoU1, out_path+"/GA/tri"+str(i)+".png", fig_type, 
                                fig=[1], n_epoch=300, N=100, add_num=30, half_reset_num=15, all_reset_num=9)

        best_rect = EntireGA(points2d, out_points, out_area, CalcIoU1, out_path+"/GA/rect"+str(i)+".png", fig_type, 
                                fig=[2], n_epoch=600, N=100, add_num=30, half_reset_num=30, all_reset_num=10)

        people_list = [best_circle, best_tri, best_rect]
        score_list = [best_circle.score, best_tri.score, best_rect.score]

        print(score_list)

        max_index = score_list.index(max(score_list))
        best = people_list[max_index]

        end = time.time()
        # print("GA:{}m".format((end-start)/60))

        # 検出図形の中心座標を3次元に射影
        opti_para2d = best.figure.p
        opti_center, _ = Plane3DProjection(points2d, opti_para2d, opti_u, opti_v, opti_O)
        
        #######################################################################

        # 検出結果を保存

        opti_para2d_list.append(opti_para2d)
        opti_plane_list.append(opti_plane_para)
        opti_u_list.append(opti_u)
        opti_v_list.append(opti_v)
        opti_O_list.append(opti_O)
        opti_AABB2d_list.append(opti_AABB2d)

        #########################################################################

        # 評価
        
        rec_list = []

        # 図形の種類が一致しているか
        if len(opti_para2d) == 3: 
            opti_fig_type = 0
        elif len(opti_para2d) == 4:
            opti_fig_type = 1
        else:
            opti_fig_type = 2

        # 一致してなかったらこれ以上評価しない
        if fig_type != opti_fig_type:
            rec_list.append(-1)

            with open(out_path+"test.csv", 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(rec_list)

            continue

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

        # 平面の法線の角度
        n_goal = np.array([plane_para[0], plane_para[1], plane_para[2]])
        n_opt = np.array([opti_plane_para[0], opti_plane_para[1], opti_plane_para[2]])
        angle = np.arccos(np.dot(n_opt, n_goal))
        angle = angle / np.pi * 180
        rec_list.append(angle)

        # 形: 混合行列で見る
        X, Y = Disassemble2d(points2d)
        index3 = (best.figure.f_rep(X, Y) >= 0)

        # print(index1.shape, np.count_nonzero(index1))
        # print(index2.shape, np.count_nonzero(index2))
        # print(index3.shape, np.count_nonzero(index3))

        estiIndex = SelectIndex(index1, SelectIndex(index2, index3))

        # print(estiIndex.shape, np.count_nonzero(estiIndex))

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

        with open(out_path+"test.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rec_list)


    np.save(out_path+"opti_para2d", np.array(opti_para2d_list))
    np.save(out_path+"opti_plane", np.array(opti_plane_list))
    np.save(out_path+"opti_u", np.array(opti_u_list))
    np.save(out_path+"opti_v", np.array(opti_v_list))
    np.save(out_path+"opti_O", np.array(opti_O_list))
    np.save(out_path+"opti_AABB2d", np.array(opti_AABB2d_list))


# test3D(1, 1, 500, 10, out_path="data/EntireTest/test2/")
# write_data3D(0, 500, 0.2, 0.4, 0.005, 20, "data/dataset/3D/4/")
# write_any_data3d(20, "data/dataset/3D/square0.45_noise/")
use_any_data3D("data/dataset/3D/square0.45_noise/", "data/EntireTest/square0.45_noise/")
# CheckView(4, 0, 0, "data/dataset/3D/0_500_test/", "data/EntireTest/testtest/")