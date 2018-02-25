2018-02-25

リサンプリング
==============

一般的に学習機の性能評価はリサンプリングを通じて行われる。リサンプリングの概要は次のようなものである。まず、データセット全体を*D*として、これを訓練セット*D*<sup>\**b*</sup>とテストセット*D* \\ *D*<sup>\**b*</sup>に分割する。この種の分割を*B*回行う(つまり、*b* = 1, ..., *B*とする)。そして、それぞれのテストセット、訓練セットの対を用いて訓練と予測を行い、性能指標*S*(*D*<sup>\**b*</sup>, *D* \\ *D*<sup>\**b*</sup>)を計算する。これにより*B*個の性能指標が得られるが、これを集約する(一般的には平均値が用いられる)。リサンプリングの方法には、クロスバリデーションやブートストラップなど様々な手法が存在する。

もしさらに詳しく知りたいのであれば、Simonによる論文([Resampling Strategies for Model Assessment and Selection | SpringerLink](https://link.springer.com/chapter/10.1007%2F978-0-387-47509-7_8))を読むのは悪い選択ではないだろう。また、Berndらによる論文、[Resampling methods for meta-model validation with recommendations for evolutionary computation](https://www.mitpressjournals.org/doi/pdf/10.1162/EVCO_a_00069)では、リサンプリング手法の統計的な背景に対して多くの説明がなされている。

リサンプリング手法を決める
--------------------------

`mlr`では`makeResampleDesc`関数を使ってリサンプリング手法を設定する。この関数にはリサンプリング手法の名前とともに、手法に応じてその他の情報(例えば繰り返し数など)を指定する。サポートしているサンプリング手法は以下のとおりである。

-   `CV`: クロスバリデーション(Cross-varidation)
-   `LOO`: 一つ抜き法(Leave-one-out cross-varidation)
-   `RepCV`: Repeatedクロスバリデーション(Repeated cross-varidation)
-   `Bootstrap`: out-of-bagブートストラップとそのバリエーション(b632等)
-   `Subsample`: サブサンプリング(モンテカルロクロスバリデーションとも呼ばれる)
-   `Holdout`: ホールドアウト法

3-fold(3分割)クロスバリデーションの場合は

``` r
rdesc = makeResampleDesc("CV", iters = 3)
rdesc
```

    $> Resample description: cross-validation with 3 iterations.
    $> Predict: test
    $> Stratification: FALSE

ホールドアウト法の場合は

``` r
rdesc = makeResampleDesc("Holdout")
rdesc
```

    $> Resample description: holdout with 0.67 split rate.
    $> Predict: test
    $> Stratification: FALSE

という具合だ。

これらのリサンプルdescriptionのうち、よく使うものは予め別名が用意してある。例えばホールドアウト法は`hout`、クロスバリデーションは`cv5`や`cv10`などよく使う分割数に対して定義してある。

``` r
hout
```

    $> Resample description: holdout with 0.67 split rate.
    $> Predict: test
    $> Stratification: FALSE

``` r
cv3
```

    $> Resample description: cross-validation with 3 iterations.
    $> Predict: test
    $> Stratification: FALSE

リサンプリングを実行する
------------------------

`resample`関数は指定されたリサンプリング手法により、学習機をタスク上で評価する。

最初の例として、`BostonHousing`データに対する線形回帰を3分割クロスバリデーションで評価してみよう。

*K*分割クロスバリデーションはデータセット*D*を*K*個の(ほぼ)等しいサイズのサブセットに分割する。*K*回の繰り返しの*b*番目では、*b*番目のサブセットがテストに、残りが訓練に使用される。

`resample`関数に学習器を指定する際には、`Learner`クラスのオブジェクトか学習器の名前(`regr.lm`など)のいずれを渡しても良い。性能指標は指定しなければ学習器に応じたデフォルトが使用される(回帰の場合は平均二乗誤差)。

``` r
rdesc = makeResampleDesc("CV", iters = 3)

r = resample("regr.lm", bh.task, rdesc)
```

    $> [Resample] cross-validation iter 1: mse.test.mean=25.1
    $> [Resample] cross-validation iter 2: mse.test.mean=23.1
    $> [Resample] cross-validation iter 3: mse.test.mean=21.9
    $> [Resample] Aggr. Result: mse.test.mean=23.4

``` r
r
```

    $> Resample Result
    $> Task: BostonHousing-example
    $> Learner: regr.lm
    $> Aggr perf: mse.test.mean=23.4
    $> Runtime: 0.047179

ここで`r`に格納したオブジェクトは`ResampleResult`クラスである。この中には評価結果の他に、実行時間や予測値、リサンプリング毎のフィット済みモデルなどが格納されている。

``` r
## 中身をざっと確認
names(r)
```

    $>  [1] "learner.id"     "task.id"        "task.desc"      "measures.train"
    $>  [5] "measures.test"  "aggr"           "pred"           "models"        
    $>  [9] "err.msgs"       "err.dumps"      "extract"        "runtime"

`r$measures.test`には各テストセットの性能指標が入っている。

``` r
## 各テストセットの性能指標
r$measures.test
```

    $>   iter      mse
    $> 1    1 25.13717
    $> 2    2 23.12795
    $> 3    3 21.91527

`r$aggr`には集約(aggrigate)後の性能指標が入っている。

``` r
## 集約後の性能指標
r$aggr
```

    $> mse.test.mean 
    $>      23.39346

名前`mse.test.mean`は、性能指標が`mse`であり、`test.mean`によりデータが集約されていることを表している。`test.mean`は多くの性能指標においてデフォルトの集約方法であり、その名前が示すようにテストデータの性能指標の平均値である。

`mlr`ではどのような種類の学習器も同じようにリサンプリングを行える。以下では、分類問題の例として`Sonar`データセットに対する分類木を5反復のサブサンプリングで評価してみよう。

サブサンプリングの各繰り返しでは、データセット*D*はランダムに訓練データとテストデータに分割される。このとき、テストデータには指定の割合のデータ数が割り当てられる。この反復が1の場合はホールドアウト法と同じである。

評価したい性能指標はリストとしてまとめて指定することもできる。以下の例では平均誤分類、偽陽性・偽陰性率、訓練時間を指定している。

``` r
rdesc = makeResampleDesc("Subsample", iter = 5, split = 4/5)
lrn = makeLearner("classif.rpart", parms = list(split = "information"))
r = resample(lrn, sonar.task, rdesc, measures = list(mmce, fpr, fnr, timetrain))
```

    $> [Resample] subsampling iter 1: mmce.test.mean=0.405,fpr.test.mean= 0.5,fnr.test.mean=0.318,timetrain.test.mean=0.019
    $> [Resample] subsampling iter 2: mmce.test.mean=0.262,fpr.test.mean=0.381,fnr.test.mean=0.143,timetrain.test.mean=0.016
    $> [Resample] subsampling iter 3: mmce.test.mean=0.19,fpr.test.mean=0.304,fnr.test.mean=0.0526,timetrain.test.mean=0.015
    $> [Resample] subsampling iter 4: mmce.test.mean=0.429,fpr.test.mean=0.35,fnr.test.mean= 0.5,timetrain.test.mean=0.013
    $> [Resample] subsampling iter 5: mmce.test.mean=0.333,fpr.test.mean=0.235,fnr.test.mean= 0.4,timetrain.test.mean=0.036
    $> [Resample] Aggr. Result: mmce.test.mean=0.324,fpr.test.mean=0.354,fnr.test.mean=0.283,timetrain.test.mean=0.0198

``` r
r
```

    $> Resample Result
    $> Task: Sonar-example
    $> Learner: classif.rpart
    $> Aggr perf: mmce.test.mean=0.324,fpr.test.mean=0.354,fnr.test.mean=0.283,timetrain.test.mean=0.0198
    $> Runtime: 0.160807

もし指標を後から追加したくなったら、`addRRMeasure`関数を使うと良い。

``` r
addRRMeasure(r, list(ber, timepredict))
```

    $> Resample Result
    $> Task: Sonar-example
    $> Learner: classif.rpart
    $> Aggr perf: mmce.test.mean=0.324,fpr.test.mean=0.354,fnr.test.mean=0.283,timetrain.test.mean=0.0198,ber.test.mean=0.318,timepredict.test.mean=0.005
    $> Runtime: 0.160807

デフォルトでは`resample`関数は進捗と途中結果を表示するが、`show.info=FALSE`で非表示にもできる。このようなメッセージを完全に制御したかったら、[Configuration - mlr tutorial](https://mlr-org.github.io/mlr-tutorial/devel/html/configureMlr/index.html)を確認してもらいたい。

上記例では学習器を明示的に作成してから`resample`に渡したが、代わりに学習器の名前を指定しても良い。その場合、学習器のパラメータは`...`引数を通じて渡すことができる。

``` r
resample("classif.rpart", parms = list(split = "information"), sonar.task, rdesc,
         measures = list(mmce, fpr, fnr, timetrain), show.info = FALSE)
```

    $> Resample Result
    $> Task: Sonar-example
    $> Learner: classif.rpart
    $> Aggr perf: mmce.test.mean=0.267,fpr.test.mean=0.246,fnr.test.mean=0.282,timetrain.test.mean=0.0146
    $> Runtime: 0.258963

リサンプル結果へのアクセス
--------------------------

学習器の性能以外にも、リサンプル結果から様々な情報を得ることが出来る。例えばリサンプリングの各繰り返しに対応する予測値やフィット済みモデル等だ。以下で情報の取得の仕方をみていこう。

### 予測値

デフォルトでは、`ResampleResult`はリサンプリングで得た予測値を含んでいる。メモリ節約などの目的でこれを止めさせたければ、`resample`関数に`keep.pred = FALSE`を指定する。

予測値は`$pred`スロットに格納されている。また、`getRRPredictions`関数を使ってアクセスすることもできる。

``` r
r$pred
```

    $> Resampled Prediction for:
    $> Resample description: subsampling with 5 iterations and 0.80 split rate.
    $> Predict: test
    $> Stratification: FALSE
    $> predict.type: response
    $> threshold: 
    $> time (mean): 0.00
    $>    id truth response iter  set
    $> 1 161     M        M    1 test
    $> 2 151     M        R    1 test
    $> 3 182     M        M    1 test
    $> 4 114     M        R    1 test
    $> 5  76     R        R    1 test
    $> 6 122     M        M    1 test
    $> ... (210 rows, 5 cols)

``` r
pred = getRRPredictions(r)
pred
```

    $> Resampled Prediction for:
    $> Resample description: subsampling with 5 iterations and 0.80 split rate.
    $> Predict: test
    $> Stratification: FALSE
    $> predict.type: response
    $> threshold: 
    $> time (mean): 0.00
    $>    id truth response iter  set
    $> 1 161     M        M    1 test
    $> 2 151     M        R    1 test
    $> 3 182     M        M    1 test
    $> 4 114     M        R    1 test
    $> 5  76     R        R    1 test
    $> 6 122     M        M    1 test
    $> ... (210 rows, 5 cols)

ここで作成した`pred`は`ResamplePrediction`クラスのオブジェクトである。これは`Prediction`オブジェクトのように`$data`にデータフレームとして予測値と真値(教師あり学習の場合)が格納されている。`as.data.frame`を使って直接`$data`スロットの中身を取得できる。さらに、`Prediction`オブジェクトに対するゲッター関数は全て利用可能である。

``` r
head(as.data.frame(pred))
```

    $>    id truth response iter  set
    $> 1 161     M        M    1 test
    $> 2 151     M        R    1 test
    $> 3 182     M        M    1 test
    $> 4 114     M        R    1 test
    $> 5  76     R        R    1 test
    $> 6 122     M        M    1 test

``` r
head(getPredictionTruth(pred))
```

    $> [1] M M M M R M
    $> Levels: M R

データフレームの`iter`と`set`は繰り返し回数とデータセットの種類(訓練なのかテストなのか)を示している。

デフォルトでは予測はテストセットだけに行われるが、`makeResampleDesc`に対し、`predict = "train"`を指定で訓練セットだけに、`predict = "both"`を指定で訓練セットとテストセットの両方に予測を行うことが出来る。後で例を見るが、*b632*や*b632+*のようなブートストラップ手法ではこれらの設定が必要となる。

以下では単純なホールドアウト法の例を見よう。つまり、テストセットと訓練セットへの分割は一度だけ行い、予測は両方のデータセットを用いて行う。

``` r
rdesc = makeResampleDesc("Holdout", predict = "both")

r = resample("classif.lda", iris.task, rdesc, show.info = FALSE)
r
```

    $> Resample Result
    $> Task: iris-example
    $> Learner: classif.lda
    $> Aggr perf: mmce.test.mean=0.02
    $> Runtime: 0.0246351

``` r
r$aggr
```

    $> mmce.test.mean 
    $>           0.02

(`predict="both"`の指定にかかわらず、`r$aggr`ではテストデータに対するmmceしか計算しないことに注意してもらいたい。訓練セットに対して計算する方法はこの後で説明する。)

リサンプリング結果から予測を取り出す方法として、`getRRPredictionList`を使う方法もある。これは、分割されたデータセット(訓練/テスト)それぞれと、リサンプリングの繰り返し毎に分割した単位でまとめた予測結果のリストを返す。

``` r
getRRPredictionList(r)
```

    $> $train
    $> $train$`1`
    $> Prediction: 100 observations
    $> predict.type: response
    $> threshold: 
    $> time: 0.00
    $>      id      truth   response
    $> 85   85 versicolor versicolor
    $> 13   13     setosa     setosa
    $> 140 140  virginica  virginica
    $> 109 109  virginica  virginica
    $> 70   70 versicolor versicolor
    $> 27   27     setosa     setosa
    $> ... (100 rows, 3 cols)
    $> 
    $> 
    $> 
    $> $test
    $> $test$`1`
    $> Prediction: 50 observations
    $> predict.type: response
    $> threshold: 
    $> time: 0.00
    $>      id      truth   response
    $> 82   82 versicolor versicolor
    $> 55   55 versicolor versicolor
    $> 29   29     setosa     setosa
    $> 147 147  virginica  virginica
    $> 44   44     setosa     setosa
    $> 83   83 versicolor versicolor
    $> ... (50 rows, 3 cols)

### 訓練済みモデルの抽出

リサンプリング毎に学習器は訓練セットにフィットさせられる。標準では、`WrappedModel`は`ResampleResult`オブジェクトには含まれておらず、`$models`スロットは空だ。これを保持するためには、`resample`関数を呼び出す際に引数`models = TRUE`を指定する必要がある。以下に生存時間分析の例を見よう。

``` r
## 3分割クロスバリデーション
rdesc = makeResampleDesc("CV", iters = 3)

r = resample("surv.coxph", lung.task, rdesc, show.info = FALSE, models = TRUE)
r$models
```

    $> [[1]]
    $> Model for learner.id=surv.coxph; learner.class=surv.coxph
    $> Trained on: task.id = lung-example; obs = 111; features = 8
    $> Hyperparameters: 
    $> 
    $> [[2]]
    $> Model for learner.id=surv.coxph; learner.class=surv.coxph
    $> Trained on: task.id = lung-example; obs = 111; features = 8
    $> Hyperparameters: 
    $> 
    $> [[3]]
    $> Model for learner.id=surv.coxph; learner.class=surv.coxph
    $> Trained on: task.id = lung-example; obs = 112; features = 8
    $> Hyperparameters:

### 他の抽出方法

完全なフィット済みモデルを保持しようとすると、リサンプリングの繰り返し数が多かったりオブジェクトが大きかったりする場合にメモリの消費量が大きくなってしまう。モデルの全ての情報を保持する代わりに、`resample`関数の`extract`引数に指定することで必要な情報だけを保持することができる。引数`extract`に対しては、リサンプリング毎の各`WrapedModel`オブジェクトに適用するための関数を渡す必要がある。

以下では、`mtcars`データセットをk=3のk-meansでクラスタリングし、クラスター中心だけを保持する例を紹介する。

``` r
rdesc = makeResampleDesc("CV", iter = 3)

r = resample("cluster.kmeans", mtcars.task, rdesc, show.info = FALSE,
             centers = 3, extract = function(x){getLearnerModel(x)$centers})
```

    $> 
    $> This is package 'modeest' written by P. PONCET.
    $> For a complete list of functions, use 'library(help = "modeest")' or 'help.start()'.

``` r
r$extract
```

    $> [[1]]
    $>        mpg      cyl      disp    hp     drat       wt     qsec        vs
    $> 1 26.96667 4.000000  99.08333  89.5 4.076667 2.087167 18.26833 0.8333333
    $> 2 20.61429 5.428571 166.81429 104.0 3.715714 3.167857 19.11429 0.7142857
    $> 3 15.26667 8.000000 356.64444 216.0 3.251111 3.956556 16.55556 0.0000000
    $>          am     gear     carb
    $> 1 1.0000000 4.333333 1.500000
    $> 2 0.2857143 3.857143 3.000000
    $> 3 0.2222222 3.444444 3.666667
    $> 
    $> [[2]]
    $>        mpg      cyl     disp       hp   drat       wt     qsec        vs
    $> 1 15.03750 8.000000 351.1750 205.0000 3.1825 4.128125 17.08375 0.0000000
    $> 2 21.22222 5.111111 157.6889 110.1111 3.7600 2.945000 18.75889 0.6666667
    $> 3 31.00000 4.000000  76.1250  62.2500 4.3275 1.896250 19.19750 1.0000000
    $>          am     gear     carb
    $> 1 0.0000000 3.000000 3.375000
    $> 2 0.4444444 3.888889 2.888889
    $> 3 1.0000000 4.000000 1.250000
    $> 
    $> [[3]]
    $>        mpg      cyl     disp       hp     drat       wt     qsec        vs
    $> 1 14.68571 8.000000 384.8571 230.5714 3.378571 4.077000 16.34000 0.0000000
    $> 2 25.30000 4.500000 113.8125 101.2500 4.037500 2.307875 18.00125 0.7500000
    $> 3 16.96667 7.333333 276.1000 145.8333 2.981667 3.580000 18.20500 0.3333333
    $>          am     gear  carb
    $> 1 0.2857143 3.571429 4.000
    $> 2 0.7500000 4.250000 2.375
    $> 3 0.0000000 3.000000 2.000

他の例として、フィット済みの回帰木から変数の重要度を`getFeatureImportance`を使って抽出してみよう(より詳しい内容は[Feature Selection - mlr tutorial](https://mlr-org.github.io/mlr-tutorial/devel/html/feature_selection/index.html)を確認してもらいたい)。

``` r
r = resample("regr.rpart", bh.task, rdesc, show.info = FALSE, extract = getFeatureImportance)
r$extract
```

    $> [[1]]
    $> FeatureImportance:
    $> Task: BostonHousing-example
    $> 
    $> Learner: regr.rpart
    $> Measure: NA
    $> Contrast: NA
    $> Aggregation: function (x)  x
    $> Replace: NA
    $> Number of Monte-Carlo iterations: NA
    $> Local: FALSE
    $>      crim       zn    indus chas      nox       rm      age      dis rad
    $> 1 2689.63 867.4877 3719.711    0 2103.622 16096.32 2574.183 3647.211   0
    $>        tax  ptratio       b    lstat
    $> 1 1972.207 3712.621 395.486 8608.757
    $> 
    $> [[2]]
    $> FeatureImportance:
    $> Task: BostonHousing-example
    $> 
    $> Learner: regr.rpart
    $> Measure: NA
    $> Contrast: NA
    $> Aggregation: function (x)  x
    $> Replace: NA
    $> Number of Monte-Carlo iterations: NA
    $> Local: FALSE
    $>       crim       zn  indus chas      nox       rm      age     dis
    $> 1 7491.707 5423.593 7295.2    0 7348.742 14014.78 1391.373 2309.92
    $>        rad      tax  ptratio b    lstat
    $> 1 340.3975 1871.451 938.0743 0 17618.49
    $> 
    $> [[3]]
    $> FeatureImportance:
    $> Task: BostonHousing-example
    $> 
    $> Learner: regr.rpart
    $> Measure: NA
    $> Contrast: NA
    $> Aggregation: function (x)  x
    $> Replace: NA
    $> Number of Monte-Carlo iterations: NA
    $> Local: FALSE
    $>       crim       zn    indus chas     nox       rm     age      dis
    $> 1 2532.084 4637.312 6150.854    0 6015.01 11330.75 6843.29 2049.772
    $>        rad      tax  ptratio        b    lstat
    $> 1 525.3815 747.8954 1925.899 62.58285 17336.77

階層化とブロック化
------------------

-   カテゴリー変数に対する階層化とは、訓練セットとテストセット内で各値の比率が変わらないようにすることを指す。階層化が可能なのは目的変数がカテゴリーである場合(教師あり学習における分類や生存時間分析)や、説明変数がカテゴリーである場合に限られる。
-   ブロック化とは、観測値の一部分をブロックとして扱い、リサンプリングの間にブロックが分割されないように扱うことを指す。つまり、ブロック全体は訓練セットかテストセットのいずれかにまとまって属すことになる。

### 目的変数の階層化

分類においては、元のデータと同じ比率で各クラスの値が含まれていることが望ましい。これはクラス間の観測数が不均衡であったり、データセットの大きさが小さい場合に有効である。さもなければ、観測数が少ないクラスのデータが訓練セットに含まれないということが起こりうる。これは分類性能の低下やモデルのクラッシュにつながる。階層化リサンプリングを行うためには、`makeResampleDesc`実行時に`stratify = TRUE`を指定する。

``` r
rdesc = makeResampleDesc("CV", iters = 3, stratify = TRUE)

r = resample("classif.lda", iris.task, rdesc, show.info = FALSE)
r
```

    $> Resample Result
    $> Task: iris-example
    $> Learner: classif.lda
    $> Aggr perf: mmce.test.mean=0.02
    $> Runtime: 0.027998

階層化を生存時間分析に対して行う場合は、打ち切りの割合が制御される。

### 説明変数の階層化

説明変数の階層化が必要な場合もある。この場合は、`stratify.cols`引数に対して階層化したい因子型変数を指定する。

``` r
rdesc = makeResampleDesc("CV", iter = 3, stratify.cols = "chas")

r = resample("regr.rpart", bh.task, rdesc, show.info = FALSE)
r
```

    $> Resample Result
    $> Task: BostonHousing-example
    $> Learner: regr.rpart
    $> Aggr perf: mse.test.mean=23.7
    $> Runtime: 0.0576711

### ブロック化

いくつかの観測値が互いに関連しており、これらが訓練データとテストデータに分割されるのが望ましくない場合には、タスク作成時にその情報を`blocking`引数に因子型ベクトルを与えることで指定する。

``` r
## それぞれ30の観測値からなる5つのブロックを指定する例
task = makeClassifTask(data = iris, target = "Species", blocking = factor(rep(1:5, each = 30)))
task
```

    $> Supervised task: iris
    $> Type: classif
    $> Target: Species
    $> Observations: 150
    $> Features:
    $> numerics  factors  ordered 
    $>        4        0        0 
    $> Missings: FALSE
    $> Has weights: FALSE
    $> Has blocking: TRUE
    $> Classes: 3
    $>     setosa versicolor  virginica 
    $>         50         50         50 
    $> Positive class: NA

リサンプリングの詳細とリサンプルのインスタンス
----------------------------------------------

既に説明したように、リサンプリング手法は`makeResampleDesc`関数を使って指定する。

``` r
rdesc = makeResampleDesc("CV", iter = 3)
rdesc
```

    $> Resample description: cross-validation with 3 iterations.
    $> Predict: test
    $> Stratification: FALSE

``` r
str(rdesc)
```

    $> List of 4
    $>  $ id      : chr "cross-validation"
    $>  $ iters   : int 3
    $>  $ predict : chr "test"
    $>  $ stratify: logi FALSE
    $>  - attr(*, "class")= chr [1:2] "CVDesc" "ResampleDesc"

上記`rdesc`は`ResampleDesc`クラス(resample descriptionの略)を継承しており、原則として、リサンプリング手法に関する必要な情報(繰り返し数、訓練セットとテストセットの比率、階層化したい変数など)を全て含んでいる。

`makeResampleInstance`関数は、データセットに含まれるデータ数を直接指定するか、タスクを指定することで、`ResampleDesc`に従って訓練セットとテストセットの概要を生成する。

``` r
## taskに基づくリサンプルインスタンスの生成
rin = makeResampleInstance(rdesc, iris.task)
rin
```

    $> Resample instance for 150 cases.
    $> Resample description: cross-validation with 3 iterations.
    $> Predict: test
    $> Stratification: FALSE

``` r
str(rin)
```

    $> List of 5
    $>  $ desc      :List of 4
    $>   ..$ id      : chr "cross-validation"
    $>   ..$ iters   : int 3
    $>   ..$ predict : chr "test"
    $>   ..$ stratify: logi FALSE
    $>   ..- attr(*, "class")= chr [1:2] "CVDesc" "ResampleDesc"
    $>  $ size      : int 150
    $>  $ train.inds:List of 3
    $>   ..$ : int [1:100] 11 111 2 125 49 71 82 16 12 121 ...
    $>   ..$ : int [1:100] 18 11 111 20 2 125 16 121 70 68 ...
    $>   ..$ : int [1:100] 18 20 49 71 82 12 68 102 5 25 ...
    $>  $ test.inds :List of 3
    $>   ..$ : int [1:50] 1 5 8 10 14 15 18 20 23 32 ...
    $>   ..$ : int [1:50] 3 4 7 12 17 22 25 27 29 31 ...
    $>   ..$ : int [1:50] 2 6 9 11 13 16 19 21 24 26 ...
    $>  $ group     : Factor w/ 0 levels: 
    $>  - attr(*, "class")= chr "ResampleInstance"

``` r
## データセットのサイズを指定してリサンプルインスタンスを生成する例
rin = makeResampleInstance(rdesc, size = nrow(iris))
str(rin)
```

    $> List of 5
    $>  $ desc      :List of 4
    $>   ..$ id      : chr "cross-validation"
    $>   ..$ iters   : int 3
    $>   ..$ predict : chr "test"
    $>   ..$ stratify: logi FALSE
    $>   ..- attr(*, "class")= chr [1:2] "CVDesc" "ResampleDesc"
    $>  $ size      : int 150
    $>  $ train.inds:List of 3
    $>   ..$ : int [1:100] 23 15 5 81 143 38 102 145 85 132 ...
    $>   ..$ : int [1:100] 99 23 78 41 15 81 108 128 102 145 ...
    $>   ..$ : int [1:100] 99 78 41 5 108 128 143 38 132 84 ...
    $>  $ test.inds :List of 3
    $>   ..$ : int [1:50] 1 3 7 8 9 10 11 13 16 19 ...
    $>   ..$ : int [1:50] 2 5 17 21 24 30 34 36 38 39 ...
    $>   ..$ : int [1:50] 4 6 12 14 15 18 20 22 23 28 ...
    $>  $ group     : Factor w/ 0 levels: 
    $>  - attr(*, "class")= chr "ResampleInstance"

ここで`rin`は`ResampleInstance`クラスを継承しており、訓練セットとテストセットのインデックスをリストとして含んでいる。

`ResampleDesc`が`resample`に渡されると、インスタンスの生成は内部的に行われる。もちろん、`ResampleInstance`を直接渡すこともできる。

リサンプルの詳細(resample description)とリサンプルのインスタンス、そしてリサンプル関数と分割するのは、複雑にしすぎているのではと感じるかもしれないが、幾つかの利点がある。

-   リサンプルインスタンスを用いると、同じ訓練セットとテストセットを用いて学習器の性能比較を行うことが容易になる。これは、既に実施した性能比較試験に対し、他の手法を追加したい場合などに特に便利である。また、後で結果を再現するためにデータとリサンプルインスタンスをセットで保管しておくこともできる。

``` r
rdesc = makeResampleDesc("CV", iter = 3)
rin = makeResampleInstance(rdesc, task = iris.task)

## 同じインスタンスを使い、2種類の学習器で性能指標を計算する
r.lda = resample("classif.lda", iris.task, rin, show.info = FALSE)
r.rpart = resample("classif.rpart", iris.task, rin, show.info = FALSE)
c("lda" = r.lda$aggr, "rpart" = r.rpart$aggr)
```

    $>   lda.mmce.test.mean rpart.mmce.test.mean 
    $>                 0.02                 0.06

-   新しいリサンプリング手法を追加したければ、`ResampleDesc`および`ResampleInstance`クラスのインスタンスを作成すればよく、`resample`関数やそれ以上のメソッドに触る必要はない。

通常、`makeResampleInstance`を呼び出したときの訓練セットとテストセットのインデックスはランダムに割り当てられる。主にホールドアウト法においては、これを完全にマニュアルで行わなければならない場面がある。これは`makeFixedHoldoutInstance`関数を使うと実現できる。

``` r
rin = makeFixedHoldoutInstance(train.inds = 1:100, test.inds = 101:150, size = 150)
rin
```

    $> Resample instance for 150 cases.
    $> Resample description: holdout with 0.67 split rate.
    $> Predict: test
    $> Stratification: FALSE

性能指標の集約
--------------

リサンプリングそれぞれに対して性能指標を計算したら、それを集計する必要がある。

大半のリサンプリング手法(ホールドアウト法、クロスバリデーション、サブサンプリングなど)では、性能指標はテストデータのみで計算され、平均によって集約される。

`mlr`における性能指標を表現する`Measure`クラスのオブジェクトは、`$aggr`スロットに対応するデフォルトの集約手法を格納している。大半は`test.mean`である。例外の一つは平均二乗誤差平方根(rmse)である。

``` r
## 一般的な集約手法
mmce$aggr
```

    $> Aggregation function: test.mean

``` r
## 具体的な計算方法
mmce$aggr$fun
```

    $> function (task, perf.test, perf.train, measure, group, pred) 
    $> mean(perf.test)
    $> <bytecode: 0x7fd6dfea4428>
    $> <environment: namespace:mlr>

``` r
## rmseの場合
rmse$aggr
```

    $> Aggregation function: test.rmse

``` r
## test.rmseの具体的な計算方法
rmse$aggr$fun
```

    $> function (task, perf.test, perf.train, measure, group, pred) 
    $> sqrt(mean(perf.test^2))
    $> <bytecode: 0x7fd6e28f8978>
    $> <environment: namespace:mlr>

`setAggrigation`関数を使うと、集約方法を変更することも出来る。利用可能な集約手法の一覧は[aggregations function | R Documentation](https://www.rdocumentation.org/packages/mlr/versions/2.10/topics/aggregations)を確認してほしい。

### 例: 一つの指標に複数の集約方法

`test.median`、`test.min`、`test.max`はそれぞれテストセットから求めた性能指標を中央値、最小値、最大値で集約する。

``` r
mseTestMedian = setAggregation(mse, test.median)
mseTestMin = setAggregation(mse, test.min)
mseTestMax = setAggregation(mse, test.max)
rdesc = makeResampleDesc("CV", iter = 3)
r = resample("regr.lm", bh.task, rdesc, show.info = FALSE, 
             measures = list(mse, mseTestMedian, mseTestMin, mseTestMax))
r
```

    $> Resample Result
    $> Task: BostonHousing-example
    $> Learner: regr.lm
    $> Aggr perf: mse.test.mean=24.2,mse.test.median=23.9,mse.test.min=20.8,mse.test.max=27.8
    $> Runtime: 0.0312719

``` r
r$aggr
```

    $>   mse.test.mean mse.test.median    mse.test.min    mse.test.max 
    $>        24.17104        23.92198        20.82022        27.77090

### 例: 訓練セットの誤差を計算する

平均誤分類率を訓練セットとテストセットに対して計算する例を示す。`makeResampleDesc`実行時に`predict = "both"`を指定しておく必要があることに注意してもらいたい。

``` r
mmceTrainMean = setAggregation(mmce, train.mean)
rdesc = makeResampleDesc("CV", iters = 3, predict = "both")
r = resample("classif.rpart", iris.task, rdesc, measures = list(mmce, mmceTrainMean))
```

    $> [Resample] cross-validation iter 1: mmce.train.mean=0.03,mmce.test.mean=0.08
    $> [Resample] cross-validation iter 2: mmce.train.mean=0.05,mmce.test.mean=0.04
    $> [Resample] cross-validation iter 3: mmce.train.mean=0.01,mmce.test.mean= 0.1
    $> [Resample] Aggr. Result: mmce.test.mean=0.0733,mmce.train.mean=0.03

### 例: ブートストラップ

out-of-bagブートストラップ推定では、まず元のデータセット*D*から重複ありの抽出によって*D*<sup>\*1</sup>, ..., *D*<sup>\**B*</sup>と*B*個の新しいデータセット(要素数は元のデータセットと同じ)を作成する。そして、*b*回目の繰り返しでは、*D*<sup>\**b*</sup>を訓練セットに使い、使われなかった要素*D* \\ *D*<sup>\**b*</sup>をテストセットに用いて各繰り返しに対する推定値を計算し、最終的に*B*個の推定値を得る。

out-of-bagブートストラップの変種である*b632*と*b632+*では、訓練セットのパフォーマンスとOOBサンプルのパフォーマンスの凸結合を計算するため、訓練セットに対する予測と適切な集計方法を必要とする。

``` r
## ブートストラップをリサンプリング手法に選び、予測は訓練セットとテストセットの両方に行う
rdesc = makeResampleDesc("Bootstrap", predict = "both", iters = 10)

## b632およびb632+専用の集計手法を設定する
mmceB632 = setAggregation(mmce, b632)
mmceB632plus = setAggregation(mmce, b632plus)

r = resample("classif.rpart", iris.task, rdesc, measures = list(mmce, mmceB632, mmceB632plus),
             show.info = FALSE)
r$measures.train
```

    $>    iter        mmce        mmce        mmce
    $> 1     1 0.006666667 0.006666667 0.006666667
    $> 2     2 0.033333333 0.033333333 0.033333333
    $> 3     3 0.026666667 0.026666667 0.026666667
    $> 4     4 0.026666667 0.026666667 0.026666667
    $> 5     5 0.020000000 0.020000000 0.020000000
    $> 6     6 0.026666667 0.026666667 0.026666667
    $> 7     7 0.033333333 0.033333333 0.033333333
    $> 8     8 0.026666667 0.026666667 0.026666667
    $> 9     9 0.026666667 0.026666667 0.026666667
    $> 10   10 0.006666667 0.006666667 0.006666667

``` r
r$aggr
```

    $> mmce.test.mean      mmce.b632  mmce.b632plus 
    $>     0.04612359     0.03773677     0.03841055

便利な関数
----------

これまでに説明した方法は柔軟ではあるが、学習器を少し試してみたい場合にはタイプ数が多くて面倒である。`mlr`には様々な略記法が用意してあるが、リサンプリング手法についても同様である。ホールドアウトやクロスバリデーション、ブートストラップ(b632)等のよく使うリサンプリング手法にはそれぞれ特有の関数が用意してある。

``` r
crossval("classif.lda", iris.task, iters = 3, measures = list(mmce, ber))
```

    $> [Resample] cross-validation iter 1: mmce.test.mean=0.04,ber.test.mean=0.0303
    $> [Resample] cross-validation iter 2: mmce.test.mean=0.02,ber.test.mean=0.0167
    $> [Resample] cross-validation iter 3: mmce.test.mean=   0,ber.test.mean=   0
    $> [Resample] Aggr. Result: mmce.test.mean=0.02,ber.test.mean=0.0157

    $> Resample Result
    $> Task: iris-example
    $> Learner: classif.lda
    $> Aggr perf: mmce.test.mean=0.02,ber.test.mean=0.0157
    $> Runtime: 0.0347431

``` r
bootstrapB632plus("regr.lm", bh.task, iters = 3, measures = list(mse, mae))
```

    $> [Resample] OOB bootstrapping iter 1: mse.b632plus=21.1,mae.b632plus=3.15,mse.b632plus=23.7,mae.b632plus=3.59
    $> [Resample] OOB bootstrapping iter 2: mse.b632plus=18.4,mae.b632plus=3.04,mse.b632plus=28.8,mae.b632plus=3.98
    $> [Resample] OOB bootstrapping iter 3: mse.b632plus=23.1,mae.b632plus=3.35,mse.b632plus=16.1,mae.b632plus=2.99
    $> [Resample] Aggr. Result: mse.b632plus=22.2,mae.b632plus=3.41

    $> Resample Result
    $> Task: BostonHousing-example
    $> Learner: regr.lm
    $> Aggr perf: mse.b632plus=22.2,mae.b632plus=3.41
    $> Runtime: 0.0514081
