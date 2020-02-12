from GAtest import *
import open3d

def use_data3d(fig_type, num, dir_path="data/dataset/3D/tri/", out_path="data/PlaneTest/"):
    # データセット読み込み
    para3d_list = np.load(dir_path+"para3d.npy")
    para2d_list = np.load(dir_path+"para2d.npy")
    points3d_list = np.load(dir_path+"points3d.npy")
    AABB3d_list = np.load(dir_path+"AABB3d.npy")
    trueIndex_list = np.load(dir_path+"trueIndex.npy")
    size_list = np.load(dir_path+"size.npy")

    #num = para2d_list.shape[0]

    for i, (para3d, para2d, points3d, AABB3d, trueIndex, size) in\
        enumerate(zip(para3d_list, para2d_list, points3d_list, AABB3d_list, trueIndex_list, size_list)):

        print("epoch:{}".format(i))

        _, _, l = buildOBB(points3d)

        # 法線推定
        normals = NormalEstimate(points3d)

        # 平面検出, 2d変換
        points2d, plane, u, v, O, estiIndex = PlaneDetect(points3d, normals, epsilon=0.01*l, alpha=np.pi/12)

        confutionIndex = ConfusionLabeling(trueIndex, estiIndex)

        TP = np.count_nonzero(confutionIndex==1)
        TN = np.count_nonzero(confutionIndex==2)
        FP = np.count_nonzero(confutionIndex==3)
        FN = np.count_nonzero(confutionIndex==4)

        acc = (TP+TN)/(TP+TN+FP+FN)
        prec = TP/(TP+FN)
        rec = TP/(TP+FP)
        F_measure = 2*prec*rec/(prec+rec)

        with open(out_path+"0.01_pi12_tri.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([TP, TN, FP, FN, acc, prec, rec, F_measure])
            
def write_data3d(fig_type, num, dir_path="data/dataset/3D/tri/"):
    para3d_list = []
    para2d_list = []
    points3d_list = []
    AABB3d_list = []
    trueIndex_list = []
    size_list = []

    plane_list = []

    for i in range(num):
        rate = Random(0.5, 1)
        # 3D図形点群作成
        para3d, para2d, plane_p, points3d, AABB3d, trueIndex = MakePointSet3D(fig_type, 500, rate=rate)

        para3d_list.append(para3d)
        para2d_list.append(para2d)
        points3d_list.append(points3d)
        AABB3d_list.append(AABB3d)
        trueIndex_list.append(trueIndex)
        size_list.append(int(points3d.shape[0]*rate//1))

        plane_list.append(plane_p)

    print("para3d:{}".format(np.array(para3d_list).shape))
    print("para2d:{}".format(np.array(para2d_list).shape))
    print("points3d:{}".format(np.array(points3d_list).shape))
    print("AABB3d:{}".format(np.array(AABB3d_list).shape))
    print("trueIndex:{}".format(np.array(trueIndex_list).shape))
    print("size:{}".format(np.array(size_list).shape))
    print("plane:{}".format(np.array(plane_list).shape))


    np.save(dir_path+"para3d", np.array(para3d_list))
    np.save(dir_path+"para2d", np.array(para2d_list))
    np.save(dir_path+"points3d", np.array(points3d_list))
    np.save(dir_path+"AABB3d", np.array(AABB3d_list))
    np.save(dir_path+"trueIndex", np.array(trueIndex_list))
    np.save(dir_path+"size", np.array(size_list))

    np.save(dir_path+"plane", np.array(plane_list))

#write_data3d(1, 50, dir_path="data/dataset/3D/tri2/")
use_data3d(1, 50)