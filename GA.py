import numpy as np
import copy
import matplotlib.pyplot as plt
import csv

from method2d import *
import figure2d as F
from IoUtest import *

class person:
    def __init__(self, fig_type, figure):
        self.fig_type = fig_type
        self.figure = figure
        self.score = 0
        self.scoreFlag = False
        self.area = figure.CalcArea()

    def Profile(self):
        print("fig_type: {}".format(self.fig_type))
        print("para: {}".format(self.figure.p))
        print("score: {}".format(self.score))

def EntireGA(points, out_points, out_area, score_f, imgPath, fig_type,
    fig=[0,1,2], n_epoch=300, N=100, add_num=0, save_num=5, tournament_size=10, 
    cross_rate=1, path=None, half_reset_num=10000, all_reset_num=10000):

    # reset指数
    half_list = [0 for i in range(len(fig))]
    all_list = [0 for i in range(len(fig))]

    # 前世代のスコア最大値
    prev_score_list = [0 for i in range(len(fig))]

    # 全個体初期化前の1位の記録
    records = [[] for i in range(len(fig))]

    # 最終結果の図形保存
    result_list = []

    # AABB生成
    max_p, min_p, _, l, _ = buildAABB2d(points)

    # 図形の種類ごとにN人クリーチャー作成
    group = np.array([CreateRandomPopulation(N, max_p, min_p, l, fig[i]) for i in range(len(fig))])

    for epoch in range(n_epoch):
        #print("epoch:{}".format(epoch))

        for i, f in enumerate(fig):

            people = group[i]

            # 新しいクリーチャー追加
            #people = np.concatenate([people, CreateRandomPopulation(add_num, max_p, min_p, l, f)])
            # スコア順に並び替え
            people, _ = Rank(score_f, people, points, out_points, out_area)
            # 上位n人は保存
            next_people = people[:save_num]
            
            # csv?に保存
            #if path:
                #SaveCSV(group[0][0], epoch, path)

            # 次世代がN人超すまで
            # トーナメント選択->交叉、突然変異->保存
            # を繰り返す
            while len(next_people) < N:
                # トーナメントサイズの人数出場
                entry = np.random.choice(people, tournament_size, replace=False)

                # 図形の種類によって"Crossover"で取る個体の数変更
                if f == 0:
                    num = 4
                elif f == 1:
                    num = 5
                elif f == 2:
                    num = 6

                # 上位num+1人選択
                winners, _ = Rank(score_f, entry, points, out_points, out_area)[:num+1]
                # 突然変異させる人を選択
                mutate_index = np.random.choice(num+1)
                # それ以外を交叉
                cross_child = Crossover2(np.delete(winners, mutate_index), f, max_p, min_p, l, cross_rate=cross_rate)
                # 突然変異
                #mutate_child = Mutate(winners[mutate_index], max_p, min_p, l, rate=mutate_rate)

                # 次世代に子を追加
                next_people = np.append(next_people, cross_child)
                
            ##### RESET処理 ############################################################################
            people, score_list = Rank(score_f, next_people, points, out_points, out_area)
            current_score = score_list[0]
        
            # スコアが変わらないようならhalf_nを増やす
            if prev_score_list[i] >= current_score:
                half_list[i] += 1

                # 半初期化する状況、かつ半初期化したらall_nが上限に達するというときに、1位を記録に残しておいて全て初期化
                if all_list[i] == all_reset_num-1 and half_list[i] == half_reset_num:
                    #print("全初期化")
                    records[i].append(people[0])
                    people = CreateRandomPopulation(N, max_p, min_p, l, fig[i])

                    half_list[i] = 0
                    all_list[i] = 0

                # half_nが上限に達したら(1位以外の)半数をランダムに初期化
                if half_list[i] == half_reset_num:
                    #print("半初期化")
                    reset_index = np.random.choice([i for i in range(1, N)], int(N/2), replace=False)
                    #reset_index = np.random.choice(N, int(N/2), replace=False)
                    # もし1位も消すなら記録に残しておく
                    #if 0 in reset_index:
                    #    records[i].append(people[0])
                    people = np.delete(people, reset_index)
                    new_people = CreateRandomPopulation(int(N/2), max_p, min_p, l, fig[i])
                    people = np.concatenate([people, new_people])

                    half_list[i] = 0
                    all_list[i] += 1

            # スコアが上がったらhalfもallも0に
            else:
                half_list[i] = 0
                all_list[i] = 0

            #print("{} -> {} : ({},{})".format(prev_score_list[i], current_score, half_list[i], all_list[i]))

            # 現在のスコアを前のスコアとして、終わり
            prev_score_list[i] = current_score

            ######################################################################################

            group[i] = people
        
        #途中経過表示
        # if epoch % 100 == 0:
        #     print("{}回目成果".format(int(epoch/100)))

        #     for i in range(len(fig)):
        #         _, score_list = Rank(score_f, group[i], points, out_points, out_area)
        #         print(score_list[:10])
        #         print(group[i][0].figure.p)
        #         #DrawFig(points, group[i][0], out_points, out_area)
    
        
    # 最終結果表示
    for i in range(len(fig)):
        
        # 最終世代   
        people, score_list = Rank(score_f, group[i], points, out_points, out_area)
        # print(score_list[:10])

        # 記録した図形を呼び出す
        if len(records[i]) != 0:
            record_score_list = [records[i][j].score for j in range(len(records[i]))]
            record_score = max(record_score_list)
            record_fig = records[i][record_score_list.index(max(record_score_list))]

            # 最終世代の一位と記録図形の一位を比較
            if score_list[0] >= record_score:
                # print("最終世代1位の勝ち")
                # print(people[0].figure.p)
                result_list.append(people[0])
            else:
                # print("記録1位の勝ち")
                # print(record_fig.figure.p)
                result_list.append(record_fig)

        else:
            # print("最終世代1位の不戦勝")
            # print(people[0].figure.p)
            result_list.append(people[0])

        # 描画
        # DrawFig(points, result_list[i], out_points, out_area, imgPath)

    score_list = [result_list[i].score for i in range(len(fig))]
    # print(result_list[0].figure.p, score_list[0])
    # print(result_list[1].figure.p, score_list[1])
    # print(result_list[2].figure.p, score_list[2])
    max_index = score_list.index(max(score_list))
    #print("種類:{}".format(max_index))
    DrawFig(points, result_list[max_index], out_points, out_area, imgPath)

    return result_list[max_index]


def CreateRandomPerson(fig_type, max_p, min_p, l):
    # 円
    if fig_type==0:
        #print("円")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(0.2*l, 2/3*l)

        figure = F.circle([x,y,r])

    # 正三角形
    elif fig_type==1:
        #print("正三角形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(0.2*l, 2/3*l)
        # 0 < t < pi*2/3
        t = Random(0, np.pi*2/3)

        figure = F.tri([x,y,r,t])

    # 長方形
    elif fig_type==2:
        #print("長方形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < w,h < l
        w = Random(0.2*l, l)
        h = Random(0.2*l, l)
        # 0 < t < pi
        t = Random(0, np.pi)

        figure = F.rect([x,y,w,h,t])

    return person(fig_type, figure)
    
def CreateRandomPopulation(num, max_p, min_p, l, fig):
    # ランダムに図形の種類を選択し、遺伝子たちを生成
    population = np.array([CreateRandomPerson(fig, max_p, min_p, l) for i in range(num)])

    return population

def Score(score_f, person, points, out_points, out_area):
    # scoreFlagが立ってなかったらIoUを計算
    if person.scoreFlag == False:
        #person.score = CalcIoU(points, person.figure)
        #person.score = CalcIoU2(points, person.figure)
        #person.score = CalcIoU3(points, out_points, out_area, person.figure)
        person.score = score_f(points, out_points, out_area, person.figure)
        person.scoreFlag = True

    return person.score

def Rank(score_f, people, points, out_points, out_area):
    # リストにスコアを記録していく
    score_list = [Score(score_f, people[i], points, out_points, out_area) for i in range(len(people))]
    # Scoreの大きい順からインデックスを読み上げ、リストに記録
    index_list = sorted(range(len(score_list)), reverse=True, key=lambda k: score_list[k])
    # index_listの順にPeopleを並べる
    return np.array(people)[index_list], np.array(score_list)[index_list]

# 図形パラメータのどれかを変更(今のところ図形の種類は変えない)
def Mutate(person, max_p, min_p, l, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:
        #personに直接書き込まないようコピー
        person = copy.deepcopy(person)
        # 図形パラメータの番号を選択
        index = np.random.choice([i for i in range(len(person.figure.p))])
        # 図形の種類にそって、選択したパラメータをランダムに変更

        # 円
        if person.fig_type == 0:
            # x
            if index == 0:
                person.figure.p[index] = Random(min_p[0], max_p[0])
            # y
            elif index == 1:
                person.figure.p[index] = Random(min_p[1], max_p[1])
            # r
            else:
                person.figure.p[index] = Random(l/10, 2/3*l)

        # 正三角形
        elif person.fig_type == 1:
            # x
            if index == 0:
                person.figure.p[index] = Random(min_p[0], max_p[0])
            # y
            elif index == 1:
                person.figure.p[index] = Random(min_p[1], max_p[1])
            # r
            elif index == 2:
                person.figure.p[index] = Random(l/10, 2/3*l)
            # t
            else:
                person.figure.p[index] = Random(0, np.pi*2/3)

        # 長方形
        elif person.fig_type == 2:
            # x
            if index == 0:
                person.figure.p[index] = Random(min_p[0], max_p[0])
            # y
            elif index == 1:
                person.figure.p[index] = Random(min_p[1], max_p[1])
            # w
            elif index == 2:
                person.figure.p[index] = Random(l/10, l)
            # h
            elif index == 3:
                person.figure.p[index] = Random(l/10, l)
            # t
            else:
                person.figure.p[index] = Random(0, np.pi/2)

    return person

# 同じ図形同士なら、場所を選択して交叉
# [x1,y1,r1][x2,y2,r2] -> 1を選択 -> [x1,y2,r2][x2,y1,r1]

# 違う図形同士なら、共通するパラメータを1つだけ交換
# [x1,y1,r][x2,y2,w,h,t] -> [x,y]が共通 -> 1を選択 -> [x1,y2,r][x2,y1,w,h,t]
# [x1,y1,r,t1][x2,y2,w,h,t2] -> [x,y,t]が共通 -> 2を選択 -> [x1,y1,r,t2][x2,y2,w,h,t1]
def Crossover(person1, person2, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:
        # personに直接書き込まないようコピー
        person1, person2 = copy.deepcopy(person1), copy.deepcopy(person2)

        f1, f2, p1, p2 = person1.fig_type, person2.fig_type, person1.figure.p, person2.figure.p

        # 同じ図形なら
        if f1 == f2:
            # 図形パラメータの番号を選択
            index = np.random.choice([i for i in range(len(p1))])
            # 同じ番号の場所を交代
            p1[index], p2[index] = p2[index], p1[index]

        # 違う図形なら
        else:
            print("error")
            # 円と正三角形 or 円と長方形
            if f1 == 0 or f2 == 0:
                # 図形パラメータの番号を選択
                index = np.random.choice([i for i in range(2)])
                # 同じ番号の場所を交代
                p1[index], p2[index] = p2[index], p1[index]

            # 正三角形と長方形
            else:
                # 図形パラメータの番号を選択
                index = np.random.choice([i for i in range(3)])

                # indexが0か1(=xかy)だったら無難に交換
                if index in [0,1]:
                    p1[index], p2[index] = p2[index], p1[index]

                # indexが2(=t)だったらまあ頑張って交換
                else:
                    if f1 == 1:
                        if p2[4] > np.pi*2/3:
                            p2[4] -= np.pi*2/3
                        p1[3], p2[4] = p2[4], p1[3]
                    else:
                        if p1[4] > np.pi*2/3:
                            p1[4] -= np.pi*2/3
                        p1[4], p2[3] = p2[3], p1[4]

    return person1, person2

# BLX-a
def BLX(x1, x2, xmin, xmax, alpha):
    r = Random(-alpha, 1+alpha)
    x = r*x1 + (1-r)*x2

    if any(xmin < x) and any(x < xmax):
        return x
        
    else:
        return BLX(x1, x2, xmin, xmax, alpha)

# ブレンド交叉を採用
def Crossover2(parents, fig, max_p, min_p, l, cross_rate):

    if np.random.rand() >= cross_rate:
        return np.random.choice(parents)

    # n: パラメータの数, x: n+1人の親のパラメータのリスト
    n = len(parents[0].figure.p)
    x = np.array([parents[i].figure.p for i in range(n+1)])

    # g: xの重心
    g = np.sum(x, axis=0) / n

    alpha = np.sqrt(n+2)

    # p, cを定義
    p, c = np.empty((0,n)), np.empty((0,n))
    p = np.append(p, [g + alpha*(x[0] - g)], axis=0)
    c = np.append(c, [[0 for i in range(n)]], axis=0)

    for i in range(1, n+1):
        r = Random(0, 1)**(1/i)
        p = np.append(p, [g + alpha*(x[i] - g)], axis=0)
        c = np.append(c, [r*(p[i-1]-p[i] + c[i-1])], axis=0)
        #print(r, p[i], c[i])

    # 子のパラメータはp[n]+c[n]となる
    child = p[n] + c[n]

    # パラメータが範囲外ならやり直し
    # if CheckIB(child, fig, max_p, min_p, l):
    #     break

    # パラメータが範囲外なら子は生成しない
    # if not CheckIB(child, fig, max_p, min_p, l):
    #     return None
        
    # パラメータをpersonクラスに代入する

    if fig == 0:
        figure = F.circle(child)
    elif fig == 1:
        figure = F.tri(child)
    elif fig == 2:
        figure = F.rect(child)

    return person(fig, figure)

# def CheckIB(child, fig, max_p, min_p, l):
#     # 円
#     if fig==0:
#         x, y, r = child
#         w, h, t_tri, t_rec = l/2, l/2, np.pi/6, np.pi/6
#     # 正三角形
#     elif fig==1:
#         x, y, r, t_tri = child
#         w, h, t_rec = l/2, l/2, np.pi/6
#     # 長方形
#     elif fig==2:
#         x, y, w, h, t_rec = child
#         r, t_tri = l/2, np.pi/6

#     if (min_p[0] < x < max_p[0]) and (min_p[1] < y < max_p[1]) and (0 < r < l) and (0 < w < l) and (0 < h < l):
#         #and (0 < t_tri < np.pi*2/3) and (0 < t_rec < np.pi/2)
#         return True

#     else:
#         return False
            

def DrawFig(points, person, out_points, out_area, path, AABB_size=1.5):
    # print(CalcIoU3(points, out_points, out_area, person.figure, flag=True))

    # Figureの初期化
    #fig = plt.figure(figsize=(12, 8))

    # 目標点群プロット
    X1, Y1= Disassemble2d(points)
    plt.plot(X1, Y1, marker=".",linestyle="None",color="yellow")

    # 推定図形プロット
    max_p, min_p, _, _, _ = buildAABB2d(points)
    
    AABB = [min_p[0], max_p[0], min_p[1], max_p[1]]

    points2 = ContourPoints(person.figure.f_rep, AABB=AABB, AABB_size=AABB_size, grid_step=1000, epsilon=0.01, down_rate=1)
    X2, Y2= Disassemble2d(points2)
    plt.plot(X2, Y2, marker=".",linestyle="None",color="red")

    plt.savefig(path)
    plt.close()

def SaveCSV(person, epoch, path):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, person.score])
