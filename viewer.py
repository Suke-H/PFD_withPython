import numpy as np
import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from method import MakePoints
from method2d import Random
import figure2 as F

plane = F.plane([0,0,1,1])
bbox = (0, 2)
AABB = [0, 2, 0, 2, 1.1, 1.3]
n_size = 100

points = MakePoints(plane.f_rep, bbox=bbox, grid_step=30)
p_size = points.shape[0]
print(p_size)

xmin, xmax, ymin, ymax, zmax, zmin = AABB
noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(n_size)])

points = np.concatenate([points, noise])
size = points.shape[0]

#点群をnp配列⇒open3d形式に
pointcloud = open3d.PointCloud()
pointcloud.points = open3d.Vector3dVector(points)

# color
colors3 = np.asarray(pointcloud.colors)
print(colors3.shape)
colors3 = np.array([[255, 130, 0] if i < p_size else [0, 0, 255] for i in range(size)]) / 255
pointcloud.colors = open3d.Vector3dVector(colors3)

# 法線推定
open3d.estimate_normals(
	pointcloud,
	search_param = open3d.KDTreeSearchParamHybrid(
	radius = 5, max_nn = 100))

# 法線の方向を視点ベースでそろえる
open3d.orient_normals_towards_camera_location(
    pointcloud,
    camera_location = np.array([0., 10., 10.], 
    dtype="float64"))

#nキーで法線表示
open3d.draw_geometries([pointcloud])