import numpy as np

from method2d import *
import figure2d as F

def MarkPoints(figure, points, epsilon):

    X, Y = Disassemble2d(points)

    #|f(x,y)|<εを満たす点群だけにする
    D = figure.f_rep(X, Y)
    index = np.where(np.abs(D)<epsilon)

    return len(index[0]), index

def CircleDict(points):
    n = points.shape[0]
    #N = 5000
    # ランダムに3点ずつN組抽出
    #index = np.array([np.random.choice(n, 3, replace=False) for i in range(N)])
    index = np.random.choice(n, size=(int((n-n%3)/3), 3), replace=False)
    points_set = points[index, :]

    num = points_set.shape[0]

    # 省略
    DET = lambda v1, v2: np.linalg.det(np.stack([v1, v2]))
    DOT = lambda v1, v2: np.dot(v1, v2)

    c_list = lambda p1, p2, p3: np.array([DOT(norm(p1-p3),(p1+p3)/2), DOT(norm(p2-p3),(p2+p3)/2)])
    center = lambda p1, p2, p3: \
            np.array([DET(c_list(p1,p2,p3), norm(p2-p3)) / DET(norm(p1-p3), norm(p2-p3)),\
                    DET(norm(p1-p3), c_list(p1,p2,p3)) / DET(norm(p1-p3), norm(p2-p3))])
    radius = lambda p1, c: np.linalg.norm(p1-c)

    c = np.array([center(points_set[i][0], points_set[i][1], points_set[i][2]) for i in range(num)])
    r = np.array([radius(points_set[i][0], c[i]) for i in range(num)])

    # パラメータ
    # p = [x0, y0, z0, r]
    r = np.reshape(r, (num,1))
    p = np.concatenate([c, r], axis=1)

    # 円生成
    Circles = [F.circle(p[i]) for i in range(num)]

    # フィットしている点の数を数える
    Scores = [MarkPoints(Circles[i], points, epsilon=0.01)[0] for i in range(num)]

    print(p[Scores.index(max(Scores))], max(Scores))

    return p[Scores.index(max(Scores))], Circles[Scores.index(max(Scores))]

C1 = F.circle([0,0,1])
points1= ContourPoints(C1.f_rep, grid_step=450, epsilon=0.01, down_rate = 0.5)
print("points:{}".format(points1.shape[0]))

X, Y = Disassemble2d(points1)
plt.plot(X, Y, marker=".",linestyle="None",color="blue")

x, figure = CircleDict(points1)

plot_implicit2d(figure.f_rep, points1)
plt.show()