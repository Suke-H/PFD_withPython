import numpy as np
import open3d

from method import *

def ViewPLY(path):
    # 読み込み
    pointcloud = open3d.read_point_cloud(path)

    # ダウンサンプリング
    # 法線推定できなくなるので不採用
    pointcloud = open3d.voxel_down_sample(pointcloud, 10)

    # 法線推定
    open3d.estimate_normals(
        pointcloud,
        search_param = open3d.KDTreeSearchParamHybrid(
        radius = 20, max_nn = 30))

    # 法線の方向を視点ベースでそろえる
    open3d.orient_normals_towards_camera_location(
        pointcloud,
        camera_location = np.array([0., 10., 10.], dtype="float64"))

    # 可視化
    open3d.draw_geometries([pointcloud])

    # numpyに変換
    points = np.asarray(pointcloud.points)
    normals = np.asarray(pointcloud.normals)

    print("points:{}".format(points.shape[0]))

    X, Y, Z = Disassemble(points)

    # OBB生成
    _, _, length = buildOBB(points)
    print("OBB_length: {}".format(length))

    return points, X, Y, Z, normals, length

ViewPLY("../data/triangle.ply")