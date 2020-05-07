import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from method3d import *

class sphere:
    """ 球面のF-Rep表現のクラス """

    def __init__(self, p):
        """
        パラメータ: p = [x0, y0, z0, r]

        中心座標: p0 = [x0, y0, z0]
        半径: r
        """
        self.p = p

    def f_rep(self, x, y, z):
        """ 球の方程式: f(x,y,z) = r - √(x-x0)^2 + (y-y0)^2 + (z-z0)^2 """
        return self.p[3] - np.sqrt((x-self.p[0])**2 + (y-self.p[1])**2 + (z-self.p[2])**2)

    def normal(self, x, y, z):
        """ 球の法線: [x-x0, y-y0, z-z0] """
        normal = np.array([x-self.p[0], y-self.p[1], z-self.p[2]])

        # 二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        # [[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

class plane:
    """ 平面のF-Rep表現のクラス """

    def __init__(self, p):
        """
        パラメータ: p = [a, b, c, d]

        法線: n = [a, b, c](単位ベクトル)
        原点との垂直距離: d
        """
        self.p = p

    def f_rep(self, x, y, z):
        """ 平面の関数:f = d - (n1*x+n2*y+n3*z) """
        return self.p[3] - (self.p[0]*x + self.p[1]*y + self.p[2]*z)

    def normal(self, x, y, z):
        """ 平面の法線: [n1, n2, n3] """
        normal = np.array([self.p[0], self.p[1], self.p[2]])
        
        #[[x,y,z],[x,y,z],...]のかたちにする
        normal = np.array([normal for i in range(x.shape[0])])

        return norm(normal)

class cylinder:
    """ 円柱面のF-Rep表現のクラス """

    def __init__(self, p):
        """
        パラメータ: p = [x0, y0, z0, a, b, c, r]

        円柱の軸の1点: p0 = [x0, y0, z0]
        方向ベクトル: v = [a, b, c]
        半径: r ( = |(p-p0) × v| )
        """
        self.p = p

    def f_rep(self, x, y, z):
        """ 円柱の関数 """
        (x0, y0, z0, a, b, c, r) = self.p
        return r - np.sqrt(( b*(z-z0)-c*(y-y0))**2 + \
            (c*(x-x0)-a*(z-z0))**2  + (a*(y-y0)-b*(x-x0))**2 )

    def normal(self, x, y, z):
        """ 円柱の法線 """
        (x0, y0, z0, a, b, c, r) = self.p
        normal = np.array([c*(c*(x-x0)-a*(z-z0)) - b*(a*(y-y0)-b*(x-x0)), \
            -c*(b*(z-z0)-c*(y-y0)) + a*(a*(y-y0)-b*(x-x0)), \
            b*(b*(z-z0)-c*(y-y0)) - a*(c*(x-x0)-a*(z-z0))])

        # 二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        # [[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

class cone:
    """ 円錐面のF-Rep表現のクラス """

    def __init__(self, p):
        """
        パラメータ: p = [x0, y0, z0, a, b, c, θ]

        円錐の頂点: p0 = [x0, y0, z0]
        方向ベクトル: v = [a, b, c]
        角度(頂角の半分): θ ( (p-p0)・v = |p-p0||v|cosθ )
        """
        self.p = p

    # 円錐の関数
    def f_rep(self, x, y, z):
        """ 円錐の関数 """
        (x0, y0, z0, a, b, c, t) = self.p
        return -np.cos(t) + (a*(x-x0)+b*(y-y0)+c*(z-z0)) / \
            np.sqrt(((x-x0)**2+(y-y0)**2+(z-z0)**2) * (a**2+b**2+c**2))

    # 円錐の法線
    def normal(self, x, y, z):
        """ 円錐の法線 """
        (x0, y0, z0, a, b, c, t) = self.p
        normal = np.array([a*(a*(x-x0)+b*(y-y0)+c*(z-z0)) - (a**2+b**2+c**2)*np.cos(t)**2*(x-x0), \
            b*(a*(x-x0)+b*(y-y0)+c*(z-z0)) - (a**2+b**2+c**2)*np.cos(t)**2*(y-y0), \
            c*(a*(x-x0)+b*(y-y0)+c*(z-z0)) - (a**2+b**2+c**2)*np.cos(t)**2*(z-z0)])

        # 二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        # [[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

class AND:
    """ intersection(=論理積, and)のクラス """
    
    def __init__(self, fig1, fig2):
        self.fig1 = fig1
        self.fig2 = fig2

    def f_rep(self, x, y, z):
        """ AND(f1,f2) = f1 + f2 - √f1^2 + f2^2 """
        return self.fig1.f_rep(x,y,z) + self.fig2.f_rep(x,y,z) - np.sqrt(self.fig1.f_rep(x,y,z)**2 + self.fig2.f_rep(x,y,z)**2)

    def normal(self, x, y, z):
        """ ∇AND = ∇f1 + ∇f2 - (f1∇f1+f2∇f2)/√f1^2+f2^2 """
        f1, f2, n1, n2 = self.fig1.f_rep(x,y,z), self.fig2.f_rep(x,y,z), self.fig1.normal(x,y,z), self.fig2.normal(x,y,z)
        normal = np.array([n1[i] + n2[i] - (f1[i]*n1[i] + f2[i]*n2[i]) / np.sqrt(f1[i]**2 + f2[i]**2) for i in range(len(f1))])
        
        return norm(normal)

class OR:
    """ union(=論理和, or)のクラス """

    def __init__(self, fig1, fig2):
        self.fig1 = fig1
        self.fig2 = fig2

    def f_rep(self, x, y, z):
        """ OR(f1,f2) = f1 + f2 + √f1^2 + f2^2 """
        return self.fig1.f_rep(x,y,z) + self.fig2.f_rep(x,y,z) + np.sqrt(self.fig1.f_rep(x,y,z)**2 + self.fig2.f_rep(x,y,z)**2)

    def normal(self, x, y, z):
        """ ∇OR = ∇f1 + ∇f2 + (f1∇f1+f2∇f2)/√f1^2+f2^2 """
        f1, f2, n1, n2 = self.fig1.f_rep(x,y,z), self.fig2.f_rep(x,y,z), self.fig1.normal(x,y,z), self.fig2.normal(x,y,z)
        normal = np.array([n1[i] + n2[i] + (f1[i]*n1[i] + f2[i]*n2[i]) / np.sqrt(f1[i]**2 + f2[i]**2) for i in range(len(f1))])

        return norm(normal)

class NOT:
    """ not(否定)のクラス """

    def __init__(self, fig):
        self.fig = fig

    def f_rep(self, x, y, z):
        """ NOT(f) = -f """
        return -self.fig.f_rep(x,y,z)

    def normal(self, x, y, z):
        """ ∇NOT = -∇f """
        normal = -self.fig.normal(x,y,z)
        
        return norm(normal)
