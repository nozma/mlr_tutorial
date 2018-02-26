2018-02-26

ベンチマーク試験
================

ベンチマーク試験では、1つ、あるいは複数の性能指標に基いてアルゴリズムを比較するために、異なる学習手法を1つあるいは複数のデータセットに適用する。

`mlr`では`benchmark`関数に学習器とタスクをリストで渡すことでベンチマーク試験を実施できる。`benchmark`は通常、学習器とタスクの対に対してリサンプリングを実行する。タスクと性能指標の組み合わせに対してどのようなリサンプリング手法を適用するかは選択することができる。

ベンチマーク試験の実施
----------------------

小さな例から始めよう。線形判別分析(lda)と分類木(rpart)を`sonar.task`に適用する。リサンプリング手法はホールドアウト法を用いる。

以下の例では`ResampleDesc`オブジェクトを作成する。各リサンプリングのインスタンスは`benchmark`関数によって自動的に作成される。インスタンス化はタスクに対して1度だけ実行される。つまり、全ての学習器は全くおなじ訓練セット、テストセットを用いることになる。なお、明示的に`ResampleInstance`を渡しても良い。

もし、データセットの作成を無作為ではなく任意に行いたければ、`makeFixedHoldoutinstance`を使うと良いだろう。

``` r
lrns = list(makeLearner("classif.lda"), makeLearner("classif.rpart"))

rdesc = makeResampleDesc("Holdout")

bmr = benchmark(lrns, sonar.task, rdesc)
```

    $> Task: Sonar-example, Learner: classif.lda

    $> [Resample] holdout iter 1: mmce.test.mean= 0.3
    $> [Resample] Aggr. Result: mmce.test.mean= 0.3
    $> Task: Sonar-example, Learner: classif.rpart
    $> [Resample] holdout iter 1: mmce.test.mean=0.343
    $> [Resample] Aggr. Result: mmce.test.mean=0.343

``` r
bmr
```

    $>         task.id    learner.id mmce.test.mean
    $> 1 Sonar-example   classif.lda      0.3000000
    $> 2 Sonar-example classif.rpart      0.3428571

もし`makeLearner`に学習器の種類以外の引数を指定するつもりがなければ、明示的に`makeLearner`を呼び出さずに単に学習器の名前を指定しても良い。上記の例は次のように書き換えることができる。

``` r
## 学習器の名前だけをベクトルで指定しても良い
lrns = c("classif.lda", "classif.rpart")

## 学習器の名前とLearnerオブジェクトを混ぜたリストでも良い
lrns = list(makeLearner("classif.lda", predict.type = "prob"), "classif.rpart")

bmr = benchmark(lrns, sonar.task, rdesc)
```

    $> Task: Sonar-example, Learner: classif.lda

    $> [Resample] holdout iter 1: mmce.test.mean= 0.3
    $> [Resample] Aggr. Result: mmce.test.mean= 0.3
    $> Task: Sonar-example, Learner: classif.rpart
    $> [Resample] holdout iter 1: mmce.test.mean= 0.3
    $> [Resample] Aggr. Result: mmce.test.mean= 0.3

``` r
bmr
```

    $>         task.id    learner.id mmce.test.mean
    $> 1 Sonar-example   classif.lda            0.3
    $> 2 Sonar-example classif.rpart            0.3

`print`の結果は各行がタスクと学習器の1つの組合せに対応している。ここでは分類のデフォルトの指標である平均誤分類率が示されている。

`bmr`は`BenchmarcResult`クラスのオブジェクトである。基本的には、これは`ResampleResult`クラスのオブジェクトのリストのリストを含んでおり、最初のリストはタスク、その中のリストは学習器に対応した並びになっている。

### 実験を再現可能にする

一般的にいって、実験は再現可能であることが望ましい。`mlr`は`set.seed`関数の設定に従うので、スクリプト実行前に`set.seed`によって乱数種を固定しておけば再現性が確保できる。

もし並列計算を使用する場合は、ユースケースに合わせて`set.seed`の呼び出し方を調整する必要がある。例えば、`set.seed(123, "L'Ecuyer")`と指定すると子プロセス単位で再現性が確保できる。mcapplyの例([mclapply function | R Documentation](https://www.rdocumentation.org/packages/parallel/versions/3.4.1/topics/mclapply))を見ると並列計算と再現性に関するもう少し詳しい情報が得られるだろう(訳注:こちらのほうが良いかも？[R: Parallel version of lapply](https://rforge.net/doc/packages/multicore/mclapply.html))。

ベンチマーク結果へのアクセス
----------------------------

`mlr`は`getBMR<抽出対象>`という名前のアクセサ関数を幾つか用意している。これにより、さらなる分析のために情報を探索することができる。これには検討中の学習アルゴリズムに関するパフォーマンスや予測などが含まれる。

### 学習器の性能

先程のベンチマーク試験の結果を見てみよう。`getBMRPerformances`は個々のリサンプリング毎の性能指標を返し、`getMBRAggrPerformances`は性能指標の集約値を返す。

``` r
getBMRPerformances(bmr)
```

    $> $`Sonar-example`
    $> $`Sonar-example`$classif.lda
    $>   iter mmce
    $> 1    1  0.3
    $> 
    $> $`Sonar-example`$classif.rpart
    $>   iter mmce
    $> 1    1  0.3

``` r
getBMRAggrPerformances(bmr)
```

    $> $`Sonar-example`
    $> $`Sonar-example`$classif.lda
    $> mmce.test.mean 
    $>            0.3 
    $> 
    $> $`Sonar-example`$classif.rpart
    $> mmce.test.mean 
    $>            0.3

今回の例ではリサンプリング手法にホールドアウト法を選んだので、リサンプリングは1回しか行っていない。そのため、個々のリサンプリング結果に基づく性能指標と集約値はどちらも同じ表示結果になっている。

デフォルトでは、ほぼすべてのゲッター関数ネストされたリストを返す。リストの最初のレベルはタスクで分類されており、二番目のレベルは学習器での分類になる。学習器またはタスクが1つしかない場合は、`drop = TRUE`を指定するとフラットなリストを得ることもできる。

``` r
getBMRPerformances(bmr, drop = TRUE)
```

    $> $classif.lda
    $>   iter mmce
    $> 1    1  0.3
    $> 
    $> $classif.rpart
    $>   iter mmce
    $> 1    1  0.3

大抵の場合はデータフレームの方が便利だろう。`as.df = TRUE`を指定すると結果をデータフレームに変換できる。

``` r
getBMRPerformances(bmr, as.df = TRUE)
```

    $>         task.id    learner.id iter mmce
    $> 1 Sonar-example   classif.lda    1  0.3
    $> 2 Sonar-example classif.rpart    1  0.3

### 予測

デフォルトでは、`BenchmarkResult`は学習器の予測結果も含んでいる。もし、メモリ節約などの目的でこれを止めさせたければ`keep.pred = FALSE`を`benchmark`関数に指定すれば良い。

予測へのアクセスは`getBMRPredictions`関数を使う。デフォルトでは、`ResamplePrediction`オブジェクトのネストされたリストが返ってくる。性能指標の場合と同様に、ここでも`drop`及び`as.df`引数を使うことができる。

``` r
getBMRPredictions(bmr)
```

    $> $`Sonar-example`
    $> $`Sonar-example`$classif.lda
    $> Resampled Prediction for:
    $> Resample description: holdout with 0.67 split rate.
    $> Predict: test
    $> Stratification: FALSE
    $> predict.type: prob
    $> threshold: M=0.50,R=0.50
    $> time (mean): 0.01
    $>    id truth      prob.M       prob.R response iter  set
    $> 1 194     M 0.001914379 9.980856e-01        R    1 test
    $> 2  18     R 0.474564117 5.254359e-01        R    1 test
    $> 3 196     M 0.996712551 3.287449e-03        M    1 test
    $> 4  10     R 0.001307244 9.986928e-01        R    1 test
    $> 5 134     M 0.999999755 2.445735e-07        M    1 test
    $> 6  34     R 0.999761364 2.386361e-04        M    1 test
    $> ... (70 rows, 7 cols)
    $> 
    $> 
    $> $`Sonar-example`$classif.rpart
    $> Resampled Prediction for:
    $> Resample description: holdout with 0.67 split rate.
    $> Predict: test
    $> Stratification: FALSE
    $> predict.type: response
    $> threshold: 
    $> time (mean): 0.01
    $>    id truth response iter  set
    $> 1 194     M        M    1 test
    $> 2  18     R        R    1 test
    $> 3 196     M        M    1 test
    $> 4  10     R        R    1 test
    $> 5 134     M        M    1 test
    $> 6  34     R        M    1 test
    $> ... (70 rows, 5 cols)

``` r
head(getBMRPredictions(bmr, as.df = TRUE))
```

    $>         task.id  learner.id  id truth      prob.M       prob.R response
    $> 1 Sonar-example classif.lda 194     M 0.001914379 9.980856e-01        R
    $> 2 Sonar-example classif.lda  18     R 0.474564117 5.254359e-01        R
    $> 3 Sonar-example classif.lda 196     M 0.996712551 3.287449e-03        M
    $> 4 Sonar-example classif.lda  10     R 0.001307244 9.986928e-01        R
    $> 5 Sonar-example classif.lda 134     M 0.999999755 2.445735e-07        M
    $> 6 Sonar-example classif.lda  34     R 0.999761364 2.386361e-04        M
    $>   iter  set
    $> 1    1 test
    $> 2    1 test
    $> 3    1 test
    $> 4    1 test
    $> 5    1 test
    $> 6    1 test

IDを通じて特定の学習器やタスクの結果にアクセスすることもできる。多くのゲッター関数はIDを指定するための`learner.ids`引数と`task.ids`引数が用意されている。

``` r
head(getBMRPredictions(bmr, learner.ids = "classif.rpart", as.df = TRUE))
```

    $>         task.id    learner.id  id truth response iter  set
    $> 1 Sonar-example classif.rpart 194     M        M    1 test
    $> 2 Sonar-example classif.rpart  18     R        R    1 test
    $> 3 Sonar-example classif.rpart 196     M        M    1 test
    $> 4 Sonar-example classif.rpart  10     R        R    1 test
    $> 5 Sonar-example classif.rpart 134     M        M    1 test
    $> 6 Sonar-example classif.rpart  34     R        M    1 test

デフォルトのIDが嫌なら、`makeLearner`や`make*Task`関数の`id`引数を通じて設定できる。さらに、学習器のIDを簡単に変更するための関数として`setLearnerId`関数も用意されている。

### ID

ベンチマーク試験における学習器、タスク、性能指標のIDは、以下のように取得できる。

``` r
getBMRTaskIds(bmr)
```

    $> [1] "Sonar-example"

``` r
getBMRLearnerIds(bmr)
```

    $> [1] "classif.lda"   "classif.rpart"

``` r
getBMRMeasureIds(bmr)
```

    $> [1] "mmce"

### フィット済みモデル

デフォルトでは`BenchmarkResult`オブジェクトはフィット済みモデルも含んでいる。これは、`benchmark`関数を呼び出す際に`models = FALSE`を指定することで抑制できる。フィット済みモデルは`getBMRModels`関数を使うことで確認できる。この関数が返すのは(おそらくネストされた)`WrappedModel`オブジェクトのリストである。

``` r
getBMRModels(bmr)
```

    $> $`Sonar-example`
    $> $`Sonar-example`$classif.lda
    $> $`Sonar-example`$classif.lda[[1]]
    $> Model for learner.id=classif.lda; learner.class=classif.lda
    $> Trained on: task.id = Sonar-example; obs = 138; features = 60
    $> Hyperparameters: 
    $> 
    $> 
    $> $`Sonar-example`$classif.rpart
    $> $`Sonar-example`$classif.rpart[[1]]
    $> Model for learner.id=classif.rpart; learner.class=classif.rpart
    $> Trained on: task.id = Sonar-example; obs = 138; features = 60
    $> Hyperparameters: xval=0

### 学習器
