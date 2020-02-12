import numpy as np

def loadOBJ(path):

    vertices = []

    #一行ずつ読み込み
    for line in open(path, 'r', encoding="utf-8_sig"):
     
        #行を分解(v -3.00 2.1 1.4など)
        vals = line.split()

        if(len(vals)!=0):
            #先頭がvならverticesに保存
            if(vals[0]=="v"):
                v = [float(i) for i in vals[1:4]]
                vertices.append(v)

    #verticesを転置させて、XYZ座標のlistに分解
    vertices = np.asarray(vertices)

    # V = vertices.T[:]

    # X = V[0]
    # Y = V[1]
    # Z = V[2]

    return vertices