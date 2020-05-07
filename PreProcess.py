import numpy as np
import open3d
from method3d import *

def NormalEstimate(points):
    """ 法線推定 """

    #点群をnp配列⇒open3d形式に
    pointcloud = open3d.PointCloud()
    pointcloud.points = open3d.Vector3dVector(points)

	# 法線推定
    open3d.estimate_normals(
    	pointcloud,
    	search_param = open3d.KDTreeSearchParamHybrid(
    	radius = 5, max_nn = 30))

	# 法線の方向を視点ベースでそろえる
    open3d.orient_normals_towards_camera_location(
        pointcloud,
        camera_location = np.array([0., 10., 10.], 
        dtype="float64"))

	#nキーで法線表示
	#open3d.draw_geometries([pointcloud])

	#法線をnumpyへ変換
    normals = np.asarray(pointcloud.normals)
    
    return normals
