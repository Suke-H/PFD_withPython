import numpy as np

class circle:
    """ 円のF-Rep表現のクラス """

    def __init__(self, p):
        """ パラメータ: p = [x0, y0, r] """
        self.p = p

    def f_rep(self, x, y):
        """ 円の方程式: f(x,y) = r - √(x-x0)^2 + (y-y0)^2 """
        x0, y0, r = self.p

        return r - np.sqrt((x-x0)**2 + (y-y0)**2)

    def CalcArea(self):
        """ 面積: S = pi*r^2 """
        return np.pi * self.p[2]**2

class line:
    """ 直線のF-Rep表現のクラス """

    def __init__(self, p):
        """ パラメータ: p = [a, b, c] """
        self.p = p

    def f_rep(self, x, y):
        """ 直線の方程式: f(x,y) = c - (ax + by) """
        a, b, c = self.p

        return c - (a*x + b*y)

class tri:
    """ 正三角形のF-Rep表現のクラス """

    def __init__(self, p):
        """
        パラメータ
        p = [x0, y0, r, t]
        x0, y0: 中心
        r: 半径(=中心から辺への垂直距離)
        t: 中心から反時計回りへの回転の角度(/rad)
        """
        self.p = p

    def f_rep(self, x, y):
        """　正三角形: spin(inter(l1,l2,l3), x0, y0, t)　"""
        x0, y0, r, t = self.p
        s = np.sqrt(3)/2
        # 3辺作成
        l1 = line([0,-1,-y0+r/2])
        l2 = line([s,0.5,s*x0+0.5*y0+r/2])
        l3 = line([-s,0.5,-s*x0+0.5*y0+r/2])
        # intersectionで正三角形作成
        tri = inter(l1, inter(l2, l3))
        # 回転
        tri = spin(tri, x0, y0, t)

        return tri.f_rep(x, y)

    def CalcArea(self):
        """ 面積: S = √3/3 * r^2 """
        return 3*np.sqrt(3)/4 * self.p[2]**2

    def CalcVertices(self):
        """ 頂点3つの座標を算出 """
        x0, y0, r, t = self.p
        s = np.sqrt(3)/2

        # θ=0としたときの3頂点の座標
        v1 = np.array([x0, y0+r])
        v2 = np.array([x0-s*r, y0-0.5*r])
        v3 = np.array([x0+s*r, y0-0.5*r])

        # 回転させる
        P = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        P = np.linalg.inv(P)
        p0 = [x0, y0]
        v1 = np.dot(v1-p0, P) + p0
        v2 = np.dot(v2-p0, P) + p0
        v3 = np.dot(v3-p0, P) + p0

        return np.stack([v1, v2, v3])

class rect:
    """ 長方形のF-Rep表現のクラス """

    def __init__(self, p):
        """
        パラメータ
        p = [x0, y0, w, h, t]
        x0, y0: 中心
        w, h: 幅、高さ
        t: 中心から反時計回りへの回転の角度(/rad)
        """
        self.p = p

    def f_rep(self, x, y):
        """ 長方形: spin(inter(l1,l2,l3,l4), x0, y0, t) """
        x0, y0, w, h, t = self.p
        # 4辺作成
        l1 = line([0,1,y0+h/2])
        l2 = line([0,-1,-y0+h/2])
        l3 = line([-1,0,-x0+w/2])
        l4 = line([1,0,x0+w/2])
        # intersectionで長方形作成
        rect = inter(l1, inter(l2, inter(l3, l4)))
        # 回転
        rect = spin(rect, x0, y0, t)

        return rect.f_rep(x, y)

    def CalcArea(self):
        """ 面積: S = w*h """
        return self.p[2] * self.p[3]

    def CalcVertices(self):
        """ 頂点4つの座標を算出 """

        x0, y0, w, h, t = self.p

        # θ=0としたときの4頂点の座標
        v1 = np.array([x0+w/2, y0+h/2])
        v2 = np.array([x0+w/2, y0-h/2])
        v3 = np.array([x0-w/2, y0+h/2])
        v4 = np.array([x0-w/2, y0-h/2])

        # 回転させる
        P = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        P = np.linalg.inv(P)
        p0 = [x0, y0]
        v1 = np.dot(v1-p0, P) + p0
        v2 = np.dot(v2-p0, P) + p0
        v3 = np.dot(v3-p0, P) + p0
        v4 = np.dot(v4-p0, P) + p0

        return np.stack([v1, v2, v3, v4])

class spin:
    """ 回転のクラス """
    
    def __init__(self, fig, x0, y0, t):
        #fig: 回転させる図形
        self.fig = fig
        #x0: 回転の中心
        self.x0 = np.array([x0, y0])
        # t: 点x0を中心に反時計回りに回転させる角度(/rad)
        self.t = t
        # P: 回転行列
        self.P = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        # P_inv: Pの逆行列
        self.P_inv = np.linalg.inv(self.P)

    
    def f_rep(self, x, y):
        """
        X = P(x-x0) + x0     で回転させることができるため、
        x = P_inv(X-x0) + x0 をf(x)に代入する
        """

        # xとyをまとめて[[x1,x2,...],[y1,y2,...]]の形にする
        p0 = np.concatenate([[x],[y]])
        
        # x0を[[x0,x0,...],[y0,y0,...]]にする
        x0 = np.array([[self.x0[0] for i in range(len(x))], [self.x0[1] for i in range(len(x))]])

        # x = P_inv(X-x0) + x0
        x, y = np.dot(self.P_inv, p0-x0) + x0

        # f(x)に代入
        return self.fig.f_rep(x, y)   

class inter:
    """ intersection(=論理積, and)のクラス """

    def __init__(self, fig1, fig2):
        self.fig1 = fig1
        self.fig2 = fig2

    def f_rep(self, x, y):
        """ inter(f1,f2) = f1 + f2 - √f1^2 + f2^2 """
        f1 = self.fig1.f_rep(x,y)
        f2 = self.fig2.f_rep(x,y)

        return f1 + f2 - np.sqrt(f1**2 + f2**2)

class union:
    """ union(=論理和, or)のクラス """

    def __init__(self, fig1, fig2):
        self.fig1 = fig1
        self.fig2 = fig2

    def f_rep(self, x, y):
        """ union(f1,f2) = f1 + f2 + √f1^2 + f2^2 """
        f1 = self.fig1.f_rep(x,y)
        f2 = self.fig2.f_rep(x,y)
        
        return f1 + f2 + np.sqrt(f1**2 + f2**2)
    