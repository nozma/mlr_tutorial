2018-02-19

予測
====

新しいデータに対する結果を予測する
----------------------------------

新しい観測値に対する目的変数の予測は、Rの他の予測手法と同じように実装されている。一般的には、`predict`を`train`が返すオブジェクトに対して呼び出し、予測したいデータを渡すだけだ。

データを渡す方法は2種類ある。

-   `task`引数を通じて`Task`オブジェクトを渡す。
-   `newdata`引数を通じて`data.frame`を渡す。

最初の方法は、予測したいデータが既に`Task`オブジェクトに含まれている場合に適している。

`train`と同様に、`predict`も`subset`引数を備えている。したがって、`Task`オブジェクトに含まれるデータの異なる部分を訓練と予測に割り当てることができる(より進んだデータ分割の方法はリサンプリングのセクションであらためて解説する)。

以下に、`BostonHousing`データに対し、1つおきの観測値に勾配ブースティングマシンによるフィットを行い、残った観測値に対して予測を行う例を示す。データは`bh.task`に予め入っているものを使用する。

``` r
n = getTaskSize(bh.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)
lrn = makeLearner("regr.gbm", n.trees = 100)
mod = train(lrn, bh.task, subset = train.set)

task.pred = predict(mod, task = bh.task, subset = test.set)
task.pred
```

    $> Prediction: 253 observations
    $> predict.type: response
    $> threshold: 
    $> time: 0.00
    $>    id truth response
    $> 2   2  21.6 22.26995
    $> 4   4  33.4 23.38155
    $> 6   6  28.7 22.40136
    $> 8   8  27.1 22.13382
    $> 10 10  18.9 22.13382
    $> 12 12  18.9 22.14428
    $> ... (253 rows, 3 cols)

2つめの方法は予測したいデータが`Task`オブジェクトに含まれていない場合に使える。

目的変数を除外した`iris`を使ってクラスター分析を行う例を示そう。奇数インデックスの要素を`Task`オブジェクトに含めて訓練を行い、残ったオブジェクトに対して予測を行う。

``` r
n = nrow(iris)
iris.train = iris[seq(1, n, by = 2), -5]
iris.test = iris[seq(2, n, by = 2), -5]
task = makeClusterTask(data = iris.train)
mod = train("cluster.kmeans", task)

newdata.pred = predict(mod, newdata = iris.test)
newdata.pred
```

    $> Prediction: 75 observations
    $> predict.type: response
    $> threshold: 
    $> time: 0.00
    $>    response
    $> 2         1
    $> 4         1
    $> 6         1
    $> 8         1
    $> 10        1
    $> 12        1
    $> ... (75 rows, 1 cols)

なお、教師あり学習の場合はデータセットから目的変数列を削除する必要はない。これは`predict`を呼び出す際に自動的に削除される。

予測へのアクセス
----------------

`predict`関数は`Prediction`クラスの名前付きリストを返す。もっとも重要な要素は`$data`であり、このdata.frameは目的変数の真値と予測値の列を含む(教師あり学習の場合)。`as.data.frame`を使うとこれに直接アクセスできる。

``` r
## task引数を通じてデータを渡した例の結果
head(as.data.frame(task.pred))
```

    $>    id truth response
    $> 2   2  21.6 22.26995
    $> 4   4  33.4 23.38155
    $> 6   6  28.7 22.40136
    $> 8   8  27.1 22.13382
    $> 10 10  18.9 22.13382
    $> 12 12  18.9 22.14428

``` r
## newdata引数を通じてデータを渡した場合の結果
head(as.data.frame(newdata.pred))
```

    $>    response
    $> 2         1
    $> 4         1
    $> 6         1
    $> 8         1
    $> 10        1
    $> 12        1

`Task`オブジェクトを通じてデータを渡した例の結果を見るとわかるように、結果のdata.frameには`id`列が追加されている。これは、予測値が元のデータセットのどの値に対応しているのかを示している。

真値と予測値に直接アクセスするための関数として`getPredictionTruth`関数と`getPredictionResponse`関数が用意されている。

``` r
head(getPredictionTruth(task.pred))
```

    $> [1] 21.6 33.4 28.7 27.1 18.9 18.9

``` r
head(getPredictionResponse(task.pred))
```

    $> [1] 22.26995 23.38155 22.40136 22.13382 22.13382 22.14428

回帰: 標準誤差を取得する
------------------------

学習器のなかには標準誤差の出力に対応しているものがあるが、これも`mlr`からアクセスできる。対応している学習器の一覧は、`listLearners`関数に引数`properties = "se"`を指定して呼び出すことで取得できる。このとき、`check.packages = FALSE`を指定することで、他のパッケージ由来の学習器で当該パッケージをまだインストールしていないものについても一覧に含めることができる。

``` r
listLearners("regr", properties = "se", check.packages = FALSE)[c("class", "name")]
```

    $> Warning in listLearners.character("regr", properties = "se", check.packages = FALSE): The following learners could not be constructed, probably because their packages are not installed:
    $> classif.bartMachine,classif.extraTrees,classif.IBk,classif.J48,classif.JRip,classif.OneR,classif.PART,cluster.Cobweb,cluster.EM,cluster.FarthestFirst,cluster.SimpleKMeans,cluster.XMeans,regr.bartMachine,regr.extraTrees,regr.IBk
    $> Check ?learners to see which packages you need or install mlr with all suggestions.

    $>          class
    $> 1   regr.bcart
    $> 2     regr.bgp
    $> 3  regr.bgpllm
    $> 4     regr.blm
    $> 5    regr.btgp
    $> 6 regr.btgpllm
    $>                                                                      name
    $> 1                                                           Bayesian CART
    $> 2                                               Bayesian Gaussian Process
    $> 3       Bayesian Gaussian Process with jumps to the Limiting Linear Model
    $> 4                                                   Bayesian Linear Model
    $> 5                                         Bayesian Treed Gaussian Process
    $> 6 Bayesian Treed Gaussian Process with jumps to the Limiting Linear Model
    $> ... (15 rows, 2 cols)

標準誤差出力の例として、`BostonHousing`に線形回帰モデルを適用する場合を示そう。標準誤差を計算するためには、`predict.type`に`"se"`を指定する。

``` r
lrn.lm = makeLearner("regr.lm", predict.type = "se")
mod.lm = train(lrn.lm, bh.task, subset = train.set)
task.pred.lm = predict(mod.lm, task = bh.task, subset = test.set)
task.pred.lm
```

    $> Prediction: 253 observations
    $> predict.type: se
    $> threshold: 
    $> time: 0.00
    $>    id truth response        se
    $> 2   2  21.6 24.83734 0.7501615
    $> 4   4  33.4 28.38206 0.8742590
    $> 6   6  28.7 25.16725 0.8652139
    $> 8   8  27.1 19.38145 1.1963265
    $> 10 10  18.9 18.66449 1.1793944
    $> 12 12  18.9 21.25802 1.0727918
    $> ... (253 rows, 4 cols)

標準誤差だけが欲しければ、`getPredictionSE`関数を使用する。

``` r
head(getPredictionSE(task.pred.lm))
```

    $> [1] 0.7501615 0.8742590 0.8652139 1.1963265 1.1793944 1.0727918

分類とクラスタリング: 確率を取得する
------------------------------------

予測値に対する確率は`Prediction`オブジェクトに`getPredictionProbabilities`関数を使うことで取得できる。以下にクラスタ分析の別の例を示そう。ここではファジイc-means法によりmtcarsデータセットをクラスタリングしている。

``` r
lrn = makeLearner("cluster.cmeans", predict.type = "prob")
mod = train(lrn, mtcars.task)

pred = predict(mod, task = mtcars.task)
head(getPredictionProbabilities(pred))
```

    $>                            1           2
    $> Mazda RX4         0.97959860 0.020401403
    $> Mazda RX4 Wag     0.97963881 0.020361186
    $> Datsun 710        0.99265893 0.007341069
    $> Hornet 4 Drive    0.54294487 0.457055126
    $> Hornet Sportabout 0.01870854 0.981291464
    $> Valiant           0.75748343 0.242516570

分類問題においては、注目すべきものがいくつかあるが、デフォルトではクラスラベルが予測される。

``` r
mod = train("classif.lda", task = iris.task)

pred = predict(mod, task = iris.task)
pred
```

    $> Prediction: 150 observations
    $> predict.type: response
    $> threshold: 
    $> time: 0.00
    $>   id  truth response
    $> 1  1 setosa   setosa
    $> 2  2 setosa   setosa
    $> 3  3 setosa   setosa
    $> 4  4 setosa   setosa
    $> 5  5 setosa   setosa
    $> 6  6 setosa   setosa
    $> ... (150 rows, 3 cols)

事後確率を得たければ、学習器を作成する際に`predict.type`引数に適当な値を指定する必要がある。

``` r
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, iris.task)

pred = predict(mod, newdata = iris)
head(as.data.frame(pred))
```

    $>    truth prob.setosa prob.versicolor prob.virginica response
    $> 1 setosa           1               0              0   setosa
    $> 2 setosa           1               0              0   setosa
    $> 3 setosa           1               0              0   setosa
    $> 4 setosa           1               0              0   setosa
    $> 5 setosa           1               0              0   setosa
    $> 6 setosa           1               0              0   setosa

クラスラベルは確率が最大のものが選択され、確率がタイの要素があればアトランダムに選択される。

もし事後確率だけがほしければ`getPredictionProbabilities`関数を使う。

``` r
head(getPredictionProbabilities(pred))
```

    $>   setosa versicolor virginica
    $> 1      1          0         0
    $> 2      1          0         0
    $> 3      1          0         0
    $> 4      1          0         0
    $> 5      1          0         0
    $> 6      1          0         0

分類: 混同行列を取得する
------------------------

混同行列は`calculateConfusionMatrix`関数により得ることが出来る。列は予測したクラス、行は真のクラスのラベルを表す。

``` r
calculateConfusionMatrix(pred)
```

    $>             predicted
    $> true         setosa versicolor virginica -err.-
    $>   setosa         50          0         0      0
    $>   versicolor      0         49         1      1
    $>   virginica       0          5        45      5
    $>   -err.-          0          5         1      6

対角成分には正しく分類された要素の数が、それ以外の部分には誤分類された要素の数が現れる。また、`-err.-`の行および列には誤分類された要素の合計数が表示される。

`relative=TRUE`を指定することで、相対頻度を得ることも出来る。

``` r
conf.matrix = calculateConfusionMatrix(pred, relative = TRUE)
conf.matrix
```

    $> Relative confusion matrix (normalized by row/column):
    $>             predicted
    $> true         setosa    versicolor virginica -err.-   
    $>   setosa     1.00/1.00 0.00/0.00  0.00/0.00 0.00     
    $>   versicolor 0.00/0.00 0.98/0.91  0.02/0.02 0.02     
    $>   virginica  0.00/0.00 0.10/0.09  0.90/0.98 0.10     
    $>   -err.-          0.00      0.09       0.02 0.04     
    $> 
    $> 
    $> Absolute confusion matrix:
    $>             predicted
    $> true         setosa versicolor virginica -err.-
    $>   setosa         50          0         0      0
    $>   versicolor      0         49         1      1
    $>   virginica       0          5        45      5
    $>   -err.-          0          5         1      6

相対頻度を計算する際、行方向と列方向の2通りの正規化の仕方があるため、上記相対混同行列の中には各要素ごとに2つの値が現れている。セットになった2つの値の1つめは行方向、つまり真のラベルについてグループ化した値で、2つめは予測値についてグループ化した値である。

相対値は`$relative.row`および`$relative.col`を通して直接アクセスすることもできる。詳しくは`ConfusionMatrix`のドキュメント([ConfusionMatrix function | R Documentation](https://www.rdocumentation.org/packages/mlr/versions/2.10/topics/ConfusionMatrix))を参照してもらいたい。

``` r
conf.matrix$relative.row
```

    $>            setosa versicolor virginica -err-
    $> setosa          1       0.00      0.00  0.00
    $> versicolor      0       0.98      0.02  0.02
    $> virginica       0       0.10      0.90  0.10

最後に、予測値および真値について、各クラスに振り分けられた要素数を`sums=TRUE`を指定することで結果に追加できる。これは相対混同行列と絶対混同行列の両方に追加される(相対と絶対で行列が入れ替わっているのはなぜだ…？)。

``` r
calculateConfusionMatrix(pred, relative = TRUE, sums = TRUE)
```

    $> Relative confusion matrix (normalized by row/column):
    $>             predicted
    $> true         setosa    versicolor virginica -err.-    -n- 
    $>   setosa     1.00/1.00 0.00/0.00  0.00/0.00 0.00      50  
    $>   versicolor 0.00/0.00 0.98/0.91  0.02/0.02 0.02      54  
    $>   virginica  0.00/0.00 0.10/0.09  0.90/0.98 0.10      46  
    $>   -err.-          0.00      0.09       0.02 0.04      <NA>
    $>   -n-        50        50         50        <NA>      150 
    $> 
    $> 
    $> Absolute confusion matrix:
    $>            setosa versicolor virginica -err.- -n-
    $> setosa         50          0         0      0  50
    $> versicolor      0         49         1      1  50
    $> virginica       0          5        45      5  50
    $> -err.-          0          5         1      6  NA
    $> -n-            50         54        46     NA 150

分類: 決定閾値の調整
--------------------
