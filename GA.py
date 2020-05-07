import numpy as np
import copy
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from method2d import *
import figure2d as F

class person:
    """ GAの個体クラス """
    def __init__(self, fig_type, figure):
        self.fig_type = fig_type
        self.figure = figure
        self.score = 0
        self.scoreFlag = False
        self.area = figure.CalcArea()

    def Profile(self):
        """ 個体情報をプリントする関数 """
        print("fig_type: {}".format(self.fig_type))
        print("para: {}".format(self.figure.p))
        print("score: {}".format(self.score))

def EntireGA(points2d, out_points, out_area, score_f, out_path, i):
    """ 3種類の図形単体GAを回してスコア最大の図形を選択 """

    # 円、正三角形、長方形でそれぞれ単体のGAを回す
    print("円 検出開始")
    best_circle = SingleGA(points2d, out_points, out_area, score_f, out_path+"/GA/circle" + str(i) + ".png", 0, 
                            n_epoch=100, N=100, half_reset_num=10, all_reset_num=5)
    print("\n正三角形 検出開始")
    best_tri = SingleGA(points2d, out_points, out_area, score_f, out_path+"/GA/tri" + str(i) + ".png", 1, 
                        n_epoch=300, N=100, half_reset_num=15, all_reset_num=9)
    print("\n長方形 検出開始")
    best_rect = SingleGA(points2d, out_points, out_area, score_f, out_path+"/GA/rect" + str(i) + ".png", 2, 
                        n_epoch=600, N=100, half_reset_num=30, all_reset_num=10)
    
    people_list = [best_circle, best_tri, best_rect]
    score_list = [best_circle.score, best_tri.score, best_rect.score]

    # スコア最大の図形を選択
    max_index = score_list.index(max(score_list))
    best = people_list[max_index]
    best_fig_type = max_index

    return best, best_fig_type

def SingleGA(points, out_points, out_area, score_f, imgPath, fig_type,
    n_epoch=300, N=100, save_num=5, tournament_size=10, 
    cross_rate=1, half_reset_num=10000, all_reset_num=10000):

    """ 1種類の図形単体のGA """

    # reset指数
    half_num = 0
    all_num = 0

    # 図形の種類によって"Crossover"で取る個体の数変更
    if fig_type == 0:
        parent_num = 4
    elif fig_type == 1:
        parent_num = 5
    elif fig_type == 2:
        parent_num = 6

    # 前世代のスコア最大値
    prev_score = 0

    # 全個体初期化前の1位の記録
    records = []

    # 最終結果の図形保存
    result = []

    # AABB生成
    max_p, min_p, l, _ = buildAABB2d(points)

    # 図形の種類ごとにN人クリーチャー作成
    people = CreateRandomPopulation(N, max_p, min_p, l, fig_type)

    for epoch in tqdm(range(n_epoch)):

        # スコア順に並び替え
        people, _ = Rank(score_f, people, points, out_points, out_area)
        # 上位n人は保存
        next_people = people[:save_num]

        # 次世代がN人超すまで
        # トーナメント選択->交叉、突然変異->保存
        # を繰り返す
        while len(next_people) < N:
            # トーナメントサイズの人数出場
            entry = np.random.choice(people, tournament_size, replace=False)

            # 上位num+1人選択
            winners, _ = Rank(score_f, entry, points, out_points, out_area)[:parent_num+1]
            # 突然変異させる人を選択
            mutate_index = np.random.choice(parent_num+1)
            # それ以外を交叉
            cross_child = Crossover(np.delete(winners, mutate_index), fig_type, max_p, min_p, l, cross_rate=cross_rate)
            # 突然変異
            #mutate_child = Mutate(winners[mutate_index], max_p, min_p, l, rate=mutate_rate)

            # 次世代に子を追加
            next_people = np.append(next_people, cross_child)
            
        # 次世代を現世代に
        people, score_list = Rank(score_f, next_people, points, out_points, out_area)

        ##### RESET処理 ############################################################################
        current_score = score_list[0]
    
        # スコアが変わらないようならhalf_nを増やす
        if prev_score >= current_score:
            half_num += 1

            # 半初期化する状況、かつ半初期化したらall_nが上限に達するというときに、1位を記録に残しておいて全て初期化
            if all_num == all_reset_num-1 and half_num == half_reset_num:
                records.append(people[0])
                people = CreateRandomPopulation(N, max_p, min_p, l, fig_type)

                half_num = 0
                all_num = 0

            # half_nが上限に達したら(1位以外の)半数をランダムに初期化
            if half_num == half_reset_num:
                reset_index = np.random.choice([i for i in range(1, N)], int(N/2), replace=False)
                people = np.delete(people, reset_index)
                new_people = CreateRandomPopulation(int(N/2), max_p, min_p, l, fig_type)
                people = np.concatenate([people, new_people])

                half_num = 0
                all_num += 1

        # スコアが上がったらhalfもallも0に
        else:
            half_num = 0
            all_num = 0

        # 現在のスコアを前のスコアとして、終わり
        prev_score = current_score

    ###############################################################################    
        
    # 最終世代   
    people, score_list = Rank(score_f, people, points, out_points, out_area)

    # 記録した図形を呼び出す
    if len(records) != 0:
        record_score_list = [records[i].score for i in range(len(records))]
        record_score = max(record_score_list)
        record_fig = records[record_score_list.index(max(record_score_list))]

        # 最終世代の一位と記録図形の一位を比較
        if score_list[0] >= record_score:
            result = people[0]
        else:
            result = record_fig

    else:
        result = people[0]

    # 描画
    DrawFig(points, result, out_points, out_area, imgPath)

    return result

def CreateRandomPerson(fig_type, max_p, min_p, l):
    """ 個体をランダム生成 """

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
    """ ランダムに図形の種類を選択し、個体たちを生成 """

    population = np.array([CreateRandomPerson(fig, max_p, min_p, l) for i in range(num)])

    return population

def Score(score_f, person, points, out_points, out_area):
    """ 適応度(スコア)を計算 """

    # scoreFlagが立ってなかったらIoUを計算
    if person.scoreFlag == False:
        person.score = score_f(points, out_points, out_area, person.figure)
        person.scoreFlag = True

    return person.score

def Rank(score_f, people, points, out_points, out_area):
    """ 集団をランク付け """

    # リストにスコアを記録していく
    score_list = [Score(score_f, people[i], points, out_points, out_area) for i in range(len(people))]
    # Scoreの大きい順からインデックスを読み上げ、リストに記録
    index_list = sorted(range(len(score_list)), reverse=True, key=lambda k: score_list[k])
    # index_listの順にPeopleを並べる
    return np.array(people)[index_list], np.array(score_list)[index_list]

def Mutate(person, max_p, min_p, l, rate=1.0):
    """ 
    突然変異を実装
    図形パラメータの1つを変更
    """

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

def Crossover(parents, fig, max_p, min_p, l, cross_rate):
    """
    交叉を実装
    シンプレックス(SPX)交叉を採用
    """

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

    # 子のパラメータはp[n]+c[n]となる
    child = p[n] + c[n]
        
    # パラメータをpersonクラスに代入する
    if fig == 0:
        figure = F.circle(child)
    elif fig == 1:
        figure = F.tri(child)
    elif fig == 2:
        figure = F.rect(child)

    return person(fig, figure)    

def DrawFig(points, person, out_points, out_area, path, AABB_size=1.5):
    """ GAでの推定図形の結果をプロット """

    # 正解点群プロット
    X1, Y1= Disassemble2d(points)
    plt.plot(X1, Y1, marker=".",linestyle="None",color="yellow")

    # 推定図形プロット
    max_p, min_p, _, _= buildAABB2d(points)
    
    AABB = [min_p[0], max_p[0], min_p[1], max_p[1]]

    points2 = ContourPoints2d(person.figure.f_rep, AABB=AABB, AABB_size=AABB_size, grid_step=1000, epsilon=0.01, down_rate=1)
    X2, Y2= Disassemble2d(points2)
    plt.plot(X2, Y2, marker=".",linestyle="None",color="red")

    plt.savefig(path)
    plt.close()
