import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from method import Disassemble

fig_type = 2

# goal座標
u_goal = np.array([1,0,0])
v_goal = np.array([0,1,0])
n_goal = np.array([0,0,1])
O_goal = np.array([0,0,0])
# opt座標
u_opt = np.array([1/np.sqrt(2),1/np.sqrt(2),0])
v_opt = np.array([-1/np.sqrt(2),1/np.sqrt(2),0])
n_opt = np.array([0,0,1])
O_opt = np.array([0,0,0])

uu = np.dot(u_goal, u_opt)
uv = np.dot(u_goal, v_opt)
un = np.dot(u_goal, n_opt)
vu = np.dot(v_goal, u_opt)
vv = np.dot(v_goal, v_opt)
vn = np.dot(v_goal, n_opt)
nu = np.dot(n_goal, u_opt)
nv = np.dot(n_goal, v_opt)
nn = np.dot(n_goal, n_opt)

P = np.array([[uu, vu, nu], [uv, vv, nv], [un, vn, nn]])

goal = np.stack([u_goal, v_goal, n_goal])
X1, Y1, Z1 = Disassemble(goal)
opt = np.stack([u_opt, v_opt, n_opt])
X2, Y2, Z2 = Disassemble(opt)

XX = np.dot(np.linalg.inv(P), goal.T)
XX = XX.T

X3, Y3, Z3 = Disassemble(XX)

#法線を描画
fig_plt = plt.figure()
ax = Axes3D(fig_plt)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.quiver(O_goal, O_goal, O_goal, X1, Y1, Z1, length=1,color='red', normalize=True)
ax.quiver(O_opt, O_opt, O_opt, X2, Y2, Z2, length=1,color='blue', normalize=True)
ax.quiver(O_opt, O_opt, O_opt, X3, Y3, Z3, length=0.1,color='green', normalize=True)

plt.show()

    
    