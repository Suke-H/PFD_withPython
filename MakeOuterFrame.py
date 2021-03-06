import numpy as np
import cv2 as cv

from method2d import *

def TransPix(points, dir_path, num, x_pix=1000):
    """ 2D点群を画像に変換 """

    # AABBの横長dx, 縦長dyを出す
    max_p, min_p, _, _ = buildAABB2d(points)
    # print(max_p, min_p)
    dx = abs(max_p[0] - min_p[0])
    dy = abs(max_p[1] - min_p[1])

    # 画像ではxを1000に固定する -> yはint(dy/dx*1000)
    px, py = x_pix, int(dy/dx*x_pix)

    # print("px, py = {}, {}".format(px, py))

    # 原点はmin_p
    cx, cy = min_p

    # pixel内に点があれば255, なければ0の画像作成
    pix = np.zeros((py, px))
    X, Y = Disassemble2d(points)

    for i in range(py):
        # if i % 100 == 0:
        #     print("{}回目".format(i))
        for j in range(px):
            x1 = (cx + dx/px*j <= X).astype(int)
            x2 = (X <= cx + dx/px*(j+1)).astype(int)
            y1 = (cy + dy/py*i <= Y).astype(int)
            y2 = (Y <= cy + dy/py*(i+1)).astype(int)
            judge = x1 * x2 * y1 * y2

            if np.any(judge == 1):
                pix[i][j] = 255

    pix = np.array(pix, dtype=np.int)
    pix = np.flipud(pix)

    RGB_pix = np.array([pix, pix, pix])
    RGB_pix = np.transpose(RGB_pix, (1,2,0))

    # originPath = 'data/Contour/origin.png'
    originPath = dir_path+'origin/'+str(num)+'.png'

    cv.imwrite(originPath, RGB_pix)

    return dx, dy, px, py, cx, cy, originPath

def Morphology(dir_path, originPath, i, dilate_size=25, erode_size=10, close_size=30, open_size=30, add_size=35):
    """ モルフォロジー処理により図形点群の外枠を生成 """

     # ファイルを読み込み
    pix = cv.imread(originPath, cv.IMREAD_COLOR)
    # 画像の大きさ取得
    height, width, _ = pix.shape
    image_size = height * width

    # グレースケール変換
    dst = cv.cvtColor(pix, cv.COLOR_RGB2GRAY)

    # モルフォロジー
    dilate_kernel = np.ones((dilate_size,dilate_size),np.uint8)
    dst = cv.dilate(dst, dilate_kernel, iterations = 1)
    cv.imwrite(dir_path+'dil/'+str(i)+'.png', dst)

    close_kernel = np.ones((close_size, close_size),np.uint8)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, close_kernel)
    cv.imwrite(dir_path+'close/'+str(i)+'.png', dst)

    open_kernel = np.ones((open_size,open_size),np.uint8)
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, open_kernel)
    cv.imwrite(dir_path+'open/'+str(i)+'.png', dst)

    add_kernel = np.ones((add_size,add_size),np.uint8)
    dst = cv.dilate(dst, add_kernel, iterations = 1)
    cv.imwrite(dir_path+'add/'+str(i)+'.png', dst)

    # 輪郭を抽出
    # dst, contours, hierarchy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst, contours, hierarchy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # 面積最大の輪郭点を取り出す
    max_area = 0
    max_contour = None
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)

        # print("area:{} <> max_area:{}".format(area, max_area))

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is None:
        return None

    # contourの形式はなんか変なのでシンプルなnumpyに変換
    max_contour = np.array(max_contour, dtype=np.float32).reshape([len(max_contour), 2])
    
    # 凸包をとる
    hull = cv.convexHull(max_contour)
    hull = np.array(hull).reshape([len(hull), 2])

    return hull

def TransPoints(contour, dx, dy, px, py, cx, cy):
    """
    画像->2D点群
    画像座標の輪郭点を点群座標に変換
    """
    points = np.zeros(contour.shape)

    for i, p in enumerate(contour):
        points[i] = [cx + dx/px*(2*p[0]+1)/2, cy + dy/py*(2*py-2*p[1]-1)/2]

    return points

def CheckCrossNum(p, contour):
    """ 点pが輪郭点contour内にあるかの判定 """

    # 輪郭点の辺をつくるため、
    # 輪郭点を[0,1,2,..n] -> [1,2,...,n,0]の順にした配列を作成
    order = [i for i in range(1, contour.shape[0])]
    order.append(0)
    contour2 = contour[order, :]

    # l: pから右に伸ばした半直線
    # 各辺とlの交差数をカウントする
    crossCount = 0
    for a, b in zip(contour, contour2):
        # ルール1,2,3
        if (a[1]<=p[1] and b[1]>p[1]) or (a[1]>p[1] and b[1]<=p[1]):

            # ルール4: cx > pxなら交差する
            #print("a:{}, b:{}, p:{}".format(a,b,p))
            cx = (p[1]*(a[0]-b[0]) + a[1]*b[0] - a[0]*b[1]) / (a[1]-b[1])
            if cx > p[0]:
                crossCount+=1

    # 交差数が偶数なら外、奇数なら内
    if crossCount%2 == 0:
        return False
    else:
        return True

def MakeOuterFrame(points, dir_path, i,  dilate_size=25, close_size=30, open_size=30, add_size=35):
    """ 2D点群から外枠の輪郭点を生成 """

    # 点群->画像
    dx, dy, px, py, cx, cy, originPath = TransPix(points, dir_path, i)

    # 画像からモルフォロジーを利用して、領域面積最大の輪郭点抽出
    contour = Morphology(dir_path, originPath, i, dilate_size=dilate_size, close_size=close_size, open_size=open_size, add_size=add_size)

    if contour is None:
        return None, 0

    # 画像座標の輪郭点を点群に変換
    contour_points = TransPoints(contour, dx, dy, px, py, cx, cy)

    area = cv.contourArea(np.array(contour_points, dtype=np.float32))

    X1, Y1 = Disassemble2d(points)
    plt.plot(X1, Y1, marker="o",linestyle="None",color="orange")

    PlotContour2d(contour_points)

    plt.savefig(dir_path+'contour/'+str(i)+'.png')
    plt.close()

    # 輪郭抽出に失敗したら終了
    if contour_points is None:
        return None, None, 0

    # 外枠内の点群だけにする
    inside_index = np.array([CheckCrossNum(points[i], contour_points) for i in range(points.shape[0])])
    inside_points = points[inside_index]

    return contour_points, inside_points, area
