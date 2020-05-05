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
