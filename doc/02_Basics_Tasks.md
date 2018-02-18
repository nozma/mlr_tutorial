2018-02-18

Learning Tasks
==============

**タスク**はデータ及び機械学習問題に関する情報、例えば教師あり学習におけるターゲットの名前などをカプセル化したものだ。

タスクの種類と作成
------------------

全てのタスクは`Task`クラスを頂点とする階層構造を持っている。以下に示すクラスは全て`Task`のサブクラスである。

-   `RegrTask`: 回帰分析に関するタスク。
-   `ClassifTask`: 2クラス分類または多クラス分類に関するタスク(注: コストがクラス依存であるコスト考慮型分類も扱うことができる)。
-   `SurvTask`: 生存時間分析に関するタスク。
-   `ClusterTask`: クラスター分析に関するタスク。
-   `MultilabelTask`: マルチラベル分類に関するタスク。
-   `CostSensTask`: 一般のコスト考慮型分類に関するタスク(コストが事例に依存するもの)。

タスクを作成するには、`make<タスク名>`というような名前の関数を使う。例えば分類タスクであれば`makeClassifTask`である。全てのタスクはID(引数`id`に指定する)とデータフレーム(引数`data`に指定する)を最低限必要とする。ただし、IDを未指定の場合は、データの変数名に基づいて自動的に割り当てられる。IDはプロットの際の注釈や、ベンチマークテストの際に名前として使われる。また、問題の性質に応じて、追加の引数が必要となる場合もある。

以下にそれぞれのタスクの生成方法を説明する。

### 回帰

教師あり学習である回帰では、`data`の他に目的変数列名である`target`を指定する必要がある。これは後に見る分類と生存時間分析においても同様である。

``` r
library(mlr)
```

    $> Loading required package: ParamHelpers

    $> Warning: replacing previous import 'BBmisc::isFALSE' by
    $> 'backports::isFALSE' when loading 'mlr'

``` r
data(BostonHousing, package = "mlbench")
regr.task = makeRegrTask(id = "bh", data = BostonHousing, target = "medv")
regr.task
```

    $> Supervised task: bh
    $> Type: regr
    $> Target: medv
    $> Observations: 506
    $> Features:
    $> numerics  factors  ordered 
    $>       12        1        0 
    $> Missings: FALSE
    $> Has weights: FALSE
    $> Has blocking: FALSE

`Task`オブジェクトの中には、学習問題のタイプと、データセットに関する基本的な情報(例えば特徴量の型やデータ数、欠測値の有無など)が格納されている。

分類でも生存時間分析でもタスク作成の基本的な枠組みは同じである。ただし、`data`の中の目的変数の種類は異なる。これについては以下で述べる。

### 分類

分類問題では、目的変数は因子型でなければならない。

以下に`BreastCancer`データセットを使って分類タスクを作成する例を示そう。ここでは`Id`列を除外していることに注意してもらいたい(訳注: `Id`列はその意味から考えて特徴量に含めるのが適当でないのは当然のこと、`character`型であるため特徴量に含めようとしてもエラーがでる)。

``` r
data(BreastCancer, package = "mlbench")
df = BreastCancer
df$Id = NULL
classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "Class")
classif.task
```

    $> Supervised task: BreastCancer
    $> Type: classif
    $> Target: Class
    $> Observations: 699
    $> Features:
    $> numerics  factors  ordered 
    $>        0        4        5 
    $> Missings: TRUE
    $> Has weights: FALSE
    $> Has blocking: FALSE
    $> Classes: 2
    $>    benign malignant 
    $>       458       241 
    $> Positive class: benign

2クラス分類においては、しばしばそれぞれのクラスを**positive**と**negative**に対応させる。上記例を見るとわかるように、デフォルトでは因子型における最初のレベルが**positive**に割り当てられるが、引数`positive`によって明示的に指定することもできる。

``` r
makeClassifTask(data = df, target = "Class", positive = "malignant")
```

    $> Supervised task: df
    $> Type: classif
    $> Target: Class
    $> Observations: 699
    $> Features:
    $> numerics  factors  ordered 
    $>        0        4        5 
    $> Missings: TRUE
    $> Has weights: FALSE
    $> Has blocking: FALSE
    $> Classes: 2
    $>    benign malignant 
    $>       458       241 
    $> Positive class: malignant

### 生存時間分析

生存時間分析においては目的変数列が2つ必要になる。左打ち切り及び右打ち切りデータでは、生存時間と打ち切りかどうかを示す二値変数が必要である。区間打ち切りデータでは、`interval2`形式でのデータ指定が必要である(詳しくは[Surv function | R Documentation](https://www.rdocumentation.org/packages/survival/versions/2.41-2/topics/Surv)を参照)。

``` r
data(lung, package = "survival")
# statusは1=censored, 2=deadとして符号化されているので、
# 論理値に変換する必要がある。
lung$status = (lung$status == 2) 
surv.task = makeSurvTask(data = lung, target = c("time", "status"))
surv.task
```

    $> Supervised task: lung
    $> Type: surv
    $> Target: time,status
    $> Events: 165
    $> Observations: 228
    $> Features:
    $> numerics  factors  ordered 
    $>        8        0        0 
    $> Missings: TRUE
    $> Has weights: FALSE
    $> Has blocking: FALSE

打ち切りの種類は`censoring`引数で明示的に指定できる。デフォルトは`rcens`(右打ち切り)である。

### マルチラベル分類

マルチラベル分類とは、対象が複数のカテゴリに同時に属す可能性があるような分類問題である。

`data`にはクラスラベルと同じだけの数の目的変数列が必要である。また、それぞれの目的変数列は論理値によってそのクラスに属するかどうかを示す必要がある。

以下に`yeast`データを用いた例を示そう。

``` r
yeast = getTaskData(yeast.task)

labels = colnames(yeast)[1:14]
yeast.task = makeMultilabelTask(id = "multi", data = yeast, target = labels)
yeast.task
```

    $> Supervised task: multi
    $> Type: multilabel
    $> Target: label1,label2,label3,label4,label5,label6,label7,label8,label9,label10,label11,label12,label13,label14
    $> Observations: 2417
    $> Features:
    $> numerics  factors  ordered 
    $>      103        0        0 
    $> Missings: FALSE
    $> Has weights: FALSE
    $> Has blocking: FALSE
    $> Classes: 14
    $>  label1  label2  label3  label4  label5  label6  label7  label8  label9 
    $>     762    1038     983     862     722     597     428     480     178 
    $> label10 label11 label12 label13 label14 
    $>     253     289    1816    1799      34

### クラスター分析

クラスター分析は教師なし学習の一種である。タスクの作成に必須の引数は`data`だけだ。`mtcars`を使ってクラスター分析のタスクを作成する例を示そう。

``` r
data(mtcars, package = "datasets")
cluster.task = makeClusterTask(data = mtcars)
cluster.task
```

    $> Unsupervised task: mtcars
    $> Type: cluster
    $> Observations: 32
    $> Features:
    $> numerics  factors  ordered 
    $>       11        0        0 
    $> Missings: FALSE
    $> Has weights: FALSE
    $> Has blocking: FALSE

### コスト考慮型分類

一般に分類問題では精度を最大化すること、つまり誤分類の数を最小化することが目的となる。つまり、これは全ての誤分類の価値を平等と考えるということである。しかし、問題によっては間違いの価値は平等とは言えないことがある。例えば健康や金融に関わる問題では、ある間違いは他の間違いよりより深刻であるということがあり得ることは容易に想像できるだろう。

コスト考慮型問題のうち、コストがクラスラベルの種類にのみ依存するような問題は、`ClassifTask`にて扱うことができる。

一方、コストが事例に依存するような例は`CostSensTask`でタスクを作成する必要がある。このケースでは入力*x*と出力*y*からなる事例(*x*, *y*)がそれぞれコストベクトル*K*に結びついていることを想定する。コストベクトル*K*はクラスラベルの数と同じ長さをもち、*k*番目の要素は*x*をクラス*k*に結びつけた場合のコストを表す。当然、*y*はコストを最小化するように選択されることが期待される。

コストベクトルはクラス*y*に関する全ての情報を包含するので、`CostSensTask`を作成するために必要なのは、全ての事例に対するコストベクトルをまとめたコスト行列と、特徴量のみである。

`iris`と人工的に作成したコスト行列を使ってコスト考慮型分類タスクを作成する例を示そう。

``` r
set.seed(123)
df = iris
cost = matrix(runif(150 * 3, 0, 2000), 150) * (1 - diag(3))[df$Species, ]
df$Species = NULL
costsens.task = makeCostSensTask(data = df, cost = cost)
costsens.task
```

    $> Supervised task: df
    $> Type: costsens
    $> Observations: 150
    $> Features:
    $> numerics  factors  ordered 
    $>        4        0        0 
    $> Missings: FALSE
    $> Has blocking: FALSE
    $> Classes: 3
    $> y1, y2, y3

その他の設定
------------

それぞれのタスク作成関数のヘルプページを確認すると、その他の引数についての情報を得ることができるだろう。

例えば、`blocking`引数は、幾つかの観測値が**一緒である**ことを示す。これによって、リサンプリングの際にそれらのデータが分割されなくなる。

他に`weights`という引数がある。これは単に観測頻度やデータ採取方法に由来する重みを表現するための方法であって、重みが本当にタスクに属している場合にのみ使用するようにしてもらいたい。もし、同じタスク内で重みを変化させて訓練をしたいと考えているのであれば、`mlr`はそのための他の方法を用意している。詳しくは`training`のチュートリアルページか`makeWeightedClassesWrapper`関数のヘルプを確認してもらいたい。

タスクへのアクセス
------------------

タスクオブジェクト内の要素を取得する方法は複数ある。これらの中で重要なものは各タスクおよび`getTaskData`のヘルプページにリストアップされている。

まずは`?TaskDesc`でリストアップされている要素を取得する方法を示そう。

``` r
getTaskDesc(classif.task)
```

    $> $id
    $> [1] "BreastCancer"
    $> 
    $> $type
    $> [1] "classif"
    $> 
    $> $target
    $> [1] "Class"
    $> 
    $> $size
    $> [1] 699
    $> 
    $> $n.feat
    $> numerics  factors  ordered 
    $>        0        4        5 
    $> 
    $> $has.missings
    $> [1] TRUE
    $> 
    $> $has.weights
    $> [1] FALSE
    $> 
    $> $has.blocking
    $> [1] FALSE
    $> 
    $> $class.levels
    $> [1] "benign"    "malignant"
    $> 
    $> $positive
    $> [1] "benign"
    $> 
    $> $negative
    $> [1] "malignant"
    $> 
    $> attr(,"class")
    $> [1] "ClassifTaskDesc"    "SupervisedTaskDesc" "TaskDesc"

取得できる要素の種類はタスクの種類によって多少異なる。

よく使う要素については、直接アクセスする手段が用意されている。

``` r
## ID
getTaskId(classif.task)
```

    $> [1] "BreastCancer"

``` r
## タスクの種類
getTaskType(classif.task)
```

    $> [1] "classif"

``` r
## 目的変数の列名
getTaskTargetNames(classif.task)
```

    $> [1] "Class"

``` r
## 観測値の数
getTaskSize(classif.task)
```

    $> [1] 699

``` r
## 特徴量の種類数
getTaskNFeats(classif.task)
```

    $> [1] 9

``` r
## クラスレベル
getTaskClassLevels(classif.task)
```

    $> [1] "benign"    "malignant"

`mlr`はさらに幾つかの関数を提供する。

``` r
## タスク内のデータ
str(getTaskData(classif.task))
```

    $> 'data.frame':    699 obs. of  10 variables:
    $>  $ Cl.thickness   : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 5 5 3 6 4 8 1 2 2 4 ...
    $>  $ Cell.size      : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 1 4 1 8 1 10 1 1 1 2 ...
    $>  $ Cell.shape     : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 1 4 1 8 1 10 1 2 1 1 ...
    $>  $ Marg.adhesion  : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 1 5 1 1 3 8 1 1 1 1 ...
    $>  $ Epith.c.size   : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 2 7 2 3 2 7 2 2 2 2 ...
    $>  $ Bare.nuclei    : Factor w/ 10 levels "1","2","3","4",..: 1 10 2 4 1 10 10 1 1 1 ...
    $>  $ Bl.cromatin    : Factor w/ 10 levels "1","2","3","4",..: 3 3 3 3 3 9 3 3 1 2 ...
    $>  $ Normal.nucleoli: Factor w/ 10 levels "1","2","3","4",..: 1 2 1 7 1 7 1 1 1 1 ...
    $>  $ Mitoses        : Factor w/ 9 levels "1","2","3","4",..: 1 1 1 1 1 1 1 1 5 1 ...
    $>  $ Class          : Factor w/ 2 levels "benign","malignant": 1 1 1 1 1 2 1 1 1 1 ...

``` r
## 特徴量の名前
getTaskFeatureNames(cluster.task)
```

    $>  [1] "mpg"  "cyl"  "disp" "hp"   "drat" "wt"   "qsec" "vs"   "am"   "gear"
    $> [11] "carb"

``` r
## 目的変数の値
head(getTaskTargets(surv.task))
```

    $>   time status
    $> 1  306   TRUE
    $> 2  455   TRUE
    $> 3 1010  FALSE
    $> 4  210   TRUE
    $> 5  883   TRUE
    $> 6 1022  FALSE

``` r
## コスト行列
head(getTaskCosts(costsens.task))
```

    $>      y1        y2         y3
    $> [1,]  0 1694.9063 1569.15053
    $> [2,]  0  995.0545   18.85981
    $> [3,]  0  775.8181 1558.13177
    $> [4,]  0  492.8980 1458.78130
    $> [5,]  0  222.1929 1260.26371
    $> [6,]  0  779.9889  961.82166

タスクの編集
------------

`mlr`には既存のタスクを編集するための関数も用意されている。既存タスクの編集は、新しいタスクをゼロから作成するよりも便利な場合がある。以下に例を示そう。

``` r
## dataの編集
cluster.task2 = subsetTask(cluster.task, subset = 4:17)
getTaskData(cluster.task)
```

    $>                      mpg cyl  disp  hp drat    wt  qsec vs am gear carb
    $> Mazda RX4           21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4
    $> Mazda RX4 Wag       21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
    $> Datsun 710          22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
    $> Hornet 4 Drive      21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
    $> Hornet Sportabout   18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
    $> Valiant             18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
    $> Duster 360          14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4
    $> Merc 240D           24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
    $> Merc 230            22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
    $> Merc 280            19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
    $> Merc 280C           17.8   6 167.6 123 3.92 3.440 18.90  1  0    4    4
    $> Merc 450SE          16.4   8 275.8 180 3.07 4.070 17.40  0  0    3    3
    $> Merc 450SL          17.3   8 275.8 180 3.07 3.730 17.60  0  0    3    3
    $> Merc 450SLC         15.2   8 275.8 180 3.07 3.780 18.00  0  0    3    3
    $> Cadillac Fleetwood  10.4   8 472.0 205 2.93 5.250 17.98  0  0    3    4
    $> Lincoln Continental 10.4   8 460.0 215 3.00 5.424 17.82  0  0    3    4
    $> Chrysler Imperial   14.7   8 440.0 230 3.23 5.345 17.42  0  0    3    4
    $> Fiat 128            32.4   4  78.7  66 4.08 2.200 19.47  1  1    4    1
    $> Honda Civic         30.4   4  75.7  52 4.93 1.615 18.52  1  1    4    2
    $> Toyota Corolla      33.9   4  71.1  65 4.22 1.835 19.90  1  1    4    1
    $> Toyota Corona       21.5   4 120.1  97 3.70 2.465 20.01  1  0    3    1
    $> Dodge Challenger    15.5   8 318.0 150 2.76 3.520 16.87  0  0    3    2
    $> AMC Javelin         15.2   8 304.0 150 3.15 3.435 17.30  0  0    3    2
    $> Camaro Z28          13.3   8 350.0 245 3.73 3.840 15.41  0  0    3    4
    $> Pontiac Firebird    19.2   8 400.0 175 3.08 3.845 17.05  0  0    3    2
    $> Fiat X1-9           27.3   4  79.0  66 4.08 1.935 18.90  1  1    4    1
    $> Porsche 914-2       26.0   4 120.3  91 4.43 2.140 16.70  0  1    5    2
    $> Lotus Europa        30.4   4  95.1 113 3.77 1.513 16.90  1  1    5    2
    $> Ford Pantera L      15.8   8 351.0 264 4.22 3.170 14.50  0  1    5    4
    $> Ferrari Dino        19.7   6 145.0 175 3.62 2.770 15.50  0  1    5    6
    $> Maserati Bora       15.0   8 301.0 335 3.54 3.570 14.60  0  1    5    8
    $> Volvo 142E          21.4   4 121.0 109 4.11 2.780 18.60  1  1    4    2

``` r
getTaskData(cluster.task2)
```

    $>                      mpg cyl  disp  hp drat    wt  qsec vs am gear carb
    $> Hornet 4 Drive      21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
    $> Hornet Sportabout   18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
    $> Valiant             18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
    $> Duster 360          14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4
    $> Merc 240D           24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
    $> Merc 230            22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
    $> Merc 280            19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
    $> Merc 280C           17.8   6 167.6 123 3.92 3.440 18.90  1  0    4    4
    $> Merc 450SE          16.4   8 275.8 180 3.07 4.070 17.40  0  0    3    3
    $> Merc 450SL          17.3   8 275.8 180 3.07 3.730 17.60  0  0    3    3
    $> Merc 450SLC         15.2   8 275.8 180 3.07 3.780 18.00  0  0    3    3
    $> Cadillac Fleetwood  10.4   8 472.0 205 2.93 5.250 17.98  0  0    3    4
    $> Lincoln Continental 10.4   8 460.0 215 3.00 5.424 17.82  0  0    3    4
    $> Chrysler Imperial   14.7   8 440.0 230 3.23 5.345 17.42  0  0    3    4

`data`のサブセットをとると、特徴量によっては値に変化がなくなる場合がある。上記の例では`am`列の値が全て0になる。このような特徴量を除外する関数として`removeConstantFeatures`がある。

``` r
removeConstantFeatures(cluster.task2)
```

    $> Removing 1 columns: am

    $> Unsupervised task: mtcars
    $> Type: cluster
    $> Observations: 14
    $> Features:
    $> numerics  factors  ordered 
    $>       10        0        0 
    $> Missings: FALSE
    $> Has weights: FALSE
    $> Has blocking: FALSE

特定の特徴量を除外したい場合は、`dropFeatures`が使える。

``` r
dropFeatures(surv.task, c("meal.cal", "wt.loss"))
```

    $> Supervised task: lung
    $> Type: surv
    $> Target: time,status
    $> Events: 165
    $> Observations: 228
    $> Features:
    $> numerics  factors  ordered 
    $>        6        0        0 
    $> Missings: TRUE
    $> Has weights: FALSE
    $> Has blocking: FALSE

数値型の特徴量を正規化したければ`normalizeFeatures`を使おう。

``` r
task = normalizeFeatures(cluster.task, method = "range")
summary(getTaskData(task))
```

    $>       mpg              cyl              disp              hp        
    $>  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
    $>  1st Qu.:0.2138   1st Qu.:0.0000   1st Qu.:0.1240   1st Qu.:0.1572  
    $>  Median :0.3745   Median :0.5000   Median :0.3123   Median :0.2509  
    $>  Mean   :0.4124   Mean   :0.5469   Mean   :0.3982   Mean   :0.3346  
    $>  3rd Qu.:0.5277   3rd Qu.:1.0000   3rd Qu.:0.6358   3rd Qu.:0.4523  
    $>  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
    $>       drat              wt              qsec              vs        
    $>  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
    $>  1st Qu.:0.1475   1st Qu.:0.2731   1st Qu.:0.2848   1st Qu.:0.0000  
    $>  Median :0.4309   Median :0.4633   Median :0.3821   Median :0.0000  
    $>  Mean   :0.3855   Mean   :0.4358   Mean   :0.3987   Mean   :0.4375  
    $>  3rd Qu.:0.5346   3rd Qu.:0.5362   3rd Qu.:0.5238   3rd Qu.:1.0000  
    $>  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
    $>        am              gear             carb       
    $>  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
    $>  1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.1429  
    $>  Median :0.0000   Median :0.5000   Median :0.1429  
    $>  Mean   :0.4062   Mean   :0.3438   Mean   :0.2589  
    $>  3rd Qu.:1.0000   3rd Qu.:0.5000   3rd Qu.:0.4286  
    $>  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000

タスクの例と便利な関数
----------------------

利便性のために、`mlr`には予めいくつかのタスクが定義してある。チュートリアルでもコードを短くするためにこれらを使う場合がある。これらの一覧は[Example Tasks - mlr tutorial](http://mlr-org.github.io/mlr-tutorial/release/html/example_tasks/index.html)を参照のこと。

また、`convertMLBenchObjToTask`関数は、`mlbench`パッケージに含まれるデータセットやデータ生成関数に由来するデータからタスクを生成するための関数である。
