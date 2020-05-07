PFD_withPython
==============

平面図形検出(Plane Figure Detect)をPythonで実装した。

概要
-----
平面図形検出とは3次元点群から円や四角形などの平面図形のパラメータを検出することであり、道路周辺の3次元点群から標識の表示板の図形を検出することを目標として実装した

今回は円、正三角形、長方形の3つを検出対象としている。

<img src="./samples/flow.PNG" width="600px">

1. 平面検出：3次元点群からRANSACにより平面を検出し、
その平面にフィットした点を2次元点群に射影

1. 外枠生成：モルフォロジー演算を利用して目標図形の外枠を生成することで、ノイズ点群を除去し図形の探索範囲を絞る

1. 平面図形検出：遺伝的アルゴリズムを利用して最適な図形パラメータを探索

Pythonでの必須ライブラリ (動作確認時のバージョン)
-----
- numpy (1.18.1)
- matplotlib (3.1.3)
- seaborn (0.10.1)
- tqdm (4.46.0)
- opencv-python (3.4.2.17)
- open3d-python (0.7.0.0)

注意
-----
- opencv-pythonは現時点での最新バージョンでエラーが起きたので注意
- opencv-python, open3d-pythonはcondaでインストールするのが難しいためAnaconda非推奨

使用方法
-----
シミュレーションで作成した点群を入力にして平面図形検出をする。

mainモジュールは`Work.py`であり、

```
python Work.py
```

を実行すると平面図形検出を開始する。

ただし`Work.py`の`main`関数を調整する必要がある。
以下に`main`関数のみ抜粋する。

```python:Work.py
def main():
    """
    シミュレーション点群を生成し、それを入力に平面図形を検出

    <引数>
    sign_type: 
    0: 半径0.3mの円
    1: 1辺0.8mの正三角形
    2: 1辺0.9mの正方形
    3. 1辺0.45mのひし形(てか正方形)
    4. 1辺が0.05～1のどれかの長方形

    scale: sign_typeのスケールを標準とした倍率
    noise_rate: 全点群数に対するノイズ点群の割合

    out_path: 出力先のフォルダパス
    
    <出力>
    origin: 元画像(2D点群を画像に変換したもの)
    dil: 膨張演算
    open: オープニング演算
    close: クロージング演算
    add: 膨張演算
    contour: 外枠の生成結果(オレンジが点群、赤が外枠)
    GA: GA図形検出の結果(オレンジが点群、赤が推定図形)
    view_data: 3Dグラフで再表示するための保存フォルダ

    """

    # 出力先のフォルダパス
    out_path = "data/"

    sign_type, scale, density, noise_rate = 1, 1, 2500, 0.2

    # 平面図形検出
    simulation(sign_type, scale, density, noise_rate, out_path)

    # # 再表示(第二引数：何番目の検出結果を表示するか(0始まり))
    # review(out_path, 0)
```

`simulation`関数の引数はコメントアウトに書いてあるので参照してほしいが、
特に`out_path`を自分の環境に合わせて変更してもらう必要がある。  
`out_path`上に`origin`, `dil`, `open`, `close`, `add`, `contour`, `GA`, `view_data`フォルダが作成され、検出結果が保存される(各フォルダの内容はコメントアウト参照)。

**結果:**

- **外枠生成の結果(`contour`) :**

<img src="./samples/contour.png" width="400px" title="外枠生成の結果">  

(オレンジが点群、赤が外枠)

- **GAでの三角形検出の結果(`GA`) :**

<img src="./samples/tri.png" width="400px" title="GAでの三角形検出の結果">  

(黄が点群、赤が検出図形)

また、プログラミングを実行した際に3Dでの結果が表示されるが、これ以降も`review`関数(`main`関数参照)により、`view_data`フォルダに保存されたデータから読み込んで表示できる。

これにより以下のように表示される：

<img src="./samples/result1.png" width="400px">
<img src="./samples/result2.png" width="400px">

(オレンジが図形点群、青がノイズ点群、赤が正解図形、青が検出図形)
