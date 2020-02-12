import numpy as np
import matplotlib.pyplot as plt

# main
from method import *
from PreProcess import NormalEstimate
from ransac import RANSAC
from Projection import Plane2DProjection, Plane3DProjection

# zenrin
import figure2d as F
from IoUtest import CalcIoU, CalcIoU2, LastIoU
from GA import *
from MakeDataset import MakePointSet, MakePointSet3D
from TransPix import MakeOuterFrame
from ClossNum import CheckClossNum, CheckClossNum2, CheckClossNum3

# PlaneViewer("data/FC_00652.jpg_0.txt", "data/FC_00652")
# PlaneViewer("data/点群データFC_00587.jpg_0.txt", "data/FC_00587")

fig_type = 1
para3d, sign3d, AABB3d = MakePointSet3D(fig_type, 500, rate=0.8)

sign2d, plane, u, v, O = PlaneDetect(sign3d)

out_points, out_area = MakeOuterFrame(sign2d, path="data/last/test.png")

print("="*50)
print(para3d)
print(AABB3d)
print(out_area)
print("="*50)

# 外枠内の点群だけにする
inside = np.array([CheckClossNum3(points[i], out_contour) for i in range(points.shape[0])])
#inside = CheckClossNum3(sign2d, out_points)
sign2d = sign2d[inside]

# GAにより最適パラメータ出力
#best = GA(sign)
best = EntireGA(sign2d, out_points, out_area)
print("="*50)

print(best[fig_type].figure.p)
optiFig2d = best[fig_type].figure

# max_p, min_p, _, _, _ = buildAABB2d(sign2d)
# AABB = [min_p[0], max_p[0], min_p[1], max_p[1]]
# optiSign2d = InteriorPoints(optiFig2d.f_rep, AABB, 500)
# optiPara3d, optiSign3d = Plane3DProjection(optiSign2d, optiFig2d, u, v, O)

# print("opti:{}".format(optiPara3d))
# print("truth:{}".format(para3d))

# # プロット準備
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# X, Y, Z = Disassemble(sign3d)
# OX, OY, OZ = Disassemble(optiSign3d)
# ax.plot(X, Y, Z, marker=".", linestyle='None', color="red")
# ax.plot(OX, OY, OZ, marker=".", linestyle='None', color="blue")

# plt.show()