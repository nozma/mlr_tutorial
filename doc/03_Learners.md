2018-02-19

学習器
======

以下に示すクラスは、(コスト考慮型)分類、回帰、生存時間分析、クラスタリングのための統一的なインターフェースを提供する。多くの手法はすでに`mlr`に統合されているが、そうでないものもある。しかし、パッケージは容易に拡張できるよう設計されている。

[Integrated Learners - mlr tutorial](http://mlr-org.github.io/mlr-tutorial/release/html/integrated_learners/index.html)には実装済みの手法とその特徴について一覧が示してある。もし使いたい手法が見つからなければ、issueに書き込むか、[Create Custom Learners - mlr tutorial](http://mlr-org.github.io/mlr-tutorial/release/html/create_learner/index.html)を確認してもらいたい。

まずは実装済みの手法についていかに使用するかを説明しよう。

学習器を構築する
----------------

学習器は`makeLearner`関数で作成する。このとき、どのような学習手法を使うのかを指定する。加えて以下の要素を指定できる。

-   ハイパーパラメータの指定。
-   予測後の出力方法(例えば、分類問題において予測されたクラスラベルなのか、確率なのか)
-   ID(いくつかの手法ではこのIDを出力やプロット時の名前として利用できる)

``` r
## ランダムフォレストによる分類で確率も出力する
classif.lrn = makeLearner(
  "classif.randomForest", predict.type = "prob", fix.factors.prediction = TRUE)
## 勾配ブースティング回帰でハイパーパラメータも指定する
regr.lrn = makeLearner(
  "regr.gbm", par.vals = list(n.trees = 500, interaction.depth = 3))
## コックス比例ハザードモデルでidも指定する
surv.lrn = makeLearner("surv.coxph", id = "cph")
## K平均法でクラスタ数を指定する
cluster.lrn = makeLearner("cluster.kmeans", centers = 5)
## マルチラベルRandom-Ferns
multilabel.lrn = makeLearner("multilabel.rFerns")
```

最初の引数はどのアルゴリズムを使うのかを指定する。アルゴリズム名は以下の命名規則に従っている。

-   `classif.<Rのメソッド名>`: 分類
-   `regr.<Rのメソッド名>`: 回帰
-   `surv.<Rのメソッド名>`: 生存時間分析
-   `cluster.<Rのメソッド名>`: クラスター分析
-   `multilabel.<Rのメソッド名>`: マルチラベル分類

ハイパーパラメータは`...`引数として渡すか、`par.vals`引数にリストとして渡せる。

因子型の特徴量は、訓練データよりテストデータの方が水準が少なくなってしまうという問題が起こるときがある。`fix.factors.prediction = TRUE`を指定しておけば、不足する水準をテストデータに加えるという方法によってこの問題を回避できる。

では、先ほど作成した学習器の中身を見てみよう。

``` r
classif.lrn
```

    $> Learner classif.randomForest from package randomForest
    $> Type: classif
    $> Name: Random Forest; Short name: rf
    $> Class: classif.randomForest
    $> Properties: twoclass,multiclass,numerics,factors,ordered,prob,class.weights,oobpreds,featimp
    $> Predict-Type: prob
    $> Hyperparameters:

``` r
surv.lrn
```

    $> Learner cph from package survival
    $> Type: surv
    $> Name: Cox Proportional Hazard Model; Short name: coxph
    $> Class: surv.coxph
    $> Properties: numerics,factors,weights,rcens
    $> Predict-Type: response
    $> Hyperparameters:

全ての学習器は`Learner`クラスのオブジェクトである。クラスには、どのような種類の特徴量を扱えるのか、予測の際にはどのような種類の出力が可能か、マルチクラス分類の問題なのか、観測値は重み付けられているのか、欠測値はサポートされているのか、といった、手法に関する情報が含まれている。

気づいたかも知れないが、今のところコスト考慮型分類に関する専用のクラスはない。一般的な誤分類コストについては、標準的な分類手法で扱うことができる。事例依存的なコストについては、コスト考慮型の学習器を一般的な回帰と分類の学習機から生成するための方法がいくつかある。この点についてはこのセクションで扱うには大きすぎるので、別途セクションを設けて解説してある。

学習器へアクセスする
--------------------

`Learner`オブジェクトはリストであり、ハイパーパラメータと予測の種類に関する情報を含んでいる。

``` r
## デフォルト値以外を指定したハイパーパラメータ
cluster.lrn$par.vals
```

    $> $centers
    $> [1] 5

``` r
## ハイパーパラメータ一覧
cluster.lrn$par.set
```

    $>               Type len           Def                             Constr
    $> centers    untyped   -             -                                  -
    $> iter.max   integer   -            10                           1 to Inf
    $> nstart     integer   -             1                           1 to Inf
    $> algorithm discrete   - Hartigan-Wong Hartigan-Wong,Lloyd,Forgy,MacQueen
    $> trace      logical   -             -                                  -
    $>           Req Tunable Trafo
    $> centers     -    TRUE     -
    $> iter.max    -    TRUE     -
    $> nstart      -    TRUE     -
    $> algorithm   -    TRUE     -
    $> trace       -   FALSE     -

``` r
##予測のタイプ
regr.lrn$predict.type
```

    $> [1] "response"

`$par.set`スロットには`ParamSet`クラスのオブジェクトが入っている。これには、ハイパーパラメータの型(数値なのか論理型なのか)、デフォルト値、そして可能な値の範囲が格納されている。

また、`mlr`は`Lerner`の現在のハイパーパラメータの設定にアクセスする`getHyperPars`や`getLernerParVals`、可能な設定項目の詳細を取得する`getParamSet`関数を用意している。これらは、ラップされた学習器において特に有用である場合がある。例えば、学習器が特徴量選択の手法と融合しており、特徴量選択手法と学習器の両方がハイパーパラメータを持つような場合である。この点については別途セクションを設けて解説している。

``` r
## ハイパーパラメータのセッティングの取得
getHyperPars(cluster.lrn)
```

    $> $centers
    $> [1] 5

``` r
## 設定可能なハイパーパラメータの詳細一覧
getParamSet(cluster.lrn)
```

    $>               Type len           Def                             Constr
    $> centers    untyped   -             -                                  -
    $> iter.max   integer   -            10                           1 to Inf
    $> nstart     integer   -             1                           1 to Inf
    $> algorithm discrete   - Hartigan-Wong Hartigan-Wong,Lloyd,Forgy,MacQueen
    $> trace      logical   -             -                                  -
    $>           Req Tunable Trafo
    $> centers     -    TRUE     -
    $> iter.max    -    TRUE     -
    $> nstart      -    TRUE     -
    $> algorithm   -    TRUE     -
    $> trace       -   FALSE     -

また、`getParamSet`(またはそのエイリアスである`getLearnerParamSet`)を使い、`Lerner`オブジェクトを作成すること無くそのデフォルト値を取得することもできる。

``` r
getParamSet("classif.randomForest")
```

    $>                      Type  len   Def   Constr Req Tunable Trafo
    $> ntree             integer    -   500 1 to Inf   -    TRUE     -
    $> mtry              integer    -     - 1 to Inf   -    TRUE     -
    $> replace           logical    -  TRUE        -   -    TRUE     -
    $> classwt     numericvector <NA>     - 0 to Inf   -    TRUE     -
    $> cutoff      numericvector <NA>     -   0 to 1   -    TRUE     -
    $> strata            untyped    -     -        -   -   FALSE     -
    $> sampsize    integervector <NA>     - 1 to Inf   -    TRUE     -
    $> nodesize          integer    -     1 1 to Inf   -    TRUE     -
    $> maxnodes          integer    -     - 1 to Inf   -    TRUE     -
    $> importance        logical    - FALSE        -   -    TRUE     -
    $> localImp          logical    - FALSE        -   -    TRUE     -
    $> proximity         logical    - FALSE        -   -   FALSE     -
    $> oob.prox          logical    -     -        -   Y   FALSE     -
    $> norm.votes        logical    -  TRUE        -   -   FALSE     -
    $> do.trace          logical    - FALSE        -   -   FALSE     -
    $> keep.forest       logical    -  TRUE        -   -   FALSE     -
    $> keep.inbag        logical    - FALSE        -   -   FALSE     -

学習器に関するメタデータにアクセスするための関数も用意してある。

``` r
## 学習器のID
getLearnerId(surv.lrn)
```

    $> [1] "cph"

``` r
## 学習器の略称
getLearnerShortName(classif.lrn)
```

    $> [1] "rf"

``` r
## 学習機のタイプ
getLearnerType(multilabel.lrn)
```

    $> [1] "multilabel"

``` r
## 学習器に必要なパッケージ
getLearnerPackages(cluster.lrn)
```

    $> [1] "stats" "clue"

学習器の編集
------------

`Learner`オブジェクトを作り直すことなしに編集する関数が用意されている。以下に例を示そう。

``` r
## IDの変更
surv.lrn = setLearnerId(surv.lrn, "CoxModel")
surv.lrn
```

    $> Learner CoxModel from package survival
    $> Type: surv
    $> Name: Cox Proportional Hazard Model; Short name: coxph
    $> Class: surv.coxph
    $> Properties: numerics,factors,weights,rcens
    $> Predict-Type: response
    $> Hyperparameters:

``` r
## 予測タイプの変更
classif.lrn = setPredictType(classif.lrn, "response")

## ハイパーパラメータ
cluster.lrn = setHyperPars(cluster.lrn, centers = 4)

## 設定値を除去してデフォルト値に戻す
regr.lrn = removeHyperPars(regr.lrn, c("n.trees", "interaction.depth"))
```

学習器一覧
----------

`mlr`に統合されている学習器とその特性は[Integrated Learners - mlr tutorial](http://mlr-org.github.io/mlr-tutorial/release/html/integrated_learners/index.html)に示してある。

もし、特定のプロパティや特定のタスクに対して利用可能な学習器の一覧がほしければ、`listLearners`関数を使うと良いだろう。

``` r
## 全ての学習器一覧
head(listLearners()[c("class", "package")])
```

    $> Warning in listLearners.character(obj = NA_character_, properties, quiet, : The following learners could not be constructed, probably because their packages are not installed:
    $> classif.bartMachine,classif.extraTrees,classif.IBk,classif.J48,classif.JRip,classif.OneR,classif.PART,cluster.Cobweb,cluster.EM,cluster.FarthestFirst,cluster.SimpleKMeans,cluster.XMeans,regr.bartMachine,regr.extraTrees,regr.IBk
    $> Check ?learners to see which packages you need or install mlr with all suggestions.

    $>                class      package
    $> 1        classif.ada          ada
    $> 2        classif.bdk      kohonen
    $> 3   classif.binomial        stats
    $> 4 classif.blackboost mboost,party
    $> 5   classif.boosting adabag,rpart
    $> 6        classif.bst          bst

``` r
## 確率を出力可能な分類器
head(listLearners("classif", properties = "prob")[c("class", "package")])
```

    $> Warning in listLearners.character("classif", properties = "prob"): The following learners could not be constructed, probably because their packages are not installed:
    $> classif.bartMachine,classif.extraTrees,classif.IBk,classif.J48,classif.JRip,classif.OneR,classif.PART,cluster.Cobweb,cluster.EM,cluster.FarthestFirst,cluster.SimpleKMeans,cluster.XMeans,regr.bartMachine,regr.extraTrees,regr.IBk
    $> Check ?learners to see which packages you need or install mlr with all suggestions.

    $>                class      package
    $> 1        classif.ada          ada
    $> 2        classif.bdk      kohonen
    $> 3   classif.binomial        stats
    $> 4 classif.blackboost mboost,party
    $> 5   classif.boosting adabag,rpart
    $> 6        classif.C50          C50

``` r
## iris(つまり多クラス)に使えて、確率を出力できる
head(listLearners(iris.task, properties = "prob")[c("class", "package")])
```

    $> Warning in listLearners.character(td$type, union(props, properties), quiet, : The following learners could not be constructed, probably because their packages are not installed:
    $> classif.bartMachine,classif.extraTrees,classif.IBk,classif.J48,classif.JRip,classif.OneR,classif.PART,cluster.Cobweb,cluster.EM,cluster.FarthestFirst,cluster.SimpleKMeans,cluster.XMeans,regr.bartMachine,regr.extraTrees,regr.IBk
    $> Check ?learners to see which packages you need or install mlr with all suggestions.

    $>              class      package
    $> 1      classif.bdk      kohonen
    $> 2 classif.boosting adabag,rpart
    $> 3      classif.C50          C50
    $> 4  classif.cforest        party
    $> 5    classif.ctree        party
    $> 6 classif.cvglmnet       glmnet

``` r
## Learnerオブジェクトを作成することもできる
head(listLearners("cluster", create = TRUE), 2)
```

    $> Warning in listLearners.character("cluster", create = TRUE): The following learners could not be constructed, probably because their packages are not installed:
    $> classif.bartMachine,classif.extraTrees,classif.IBk,classif.J48,classif.JRip,classif.OneR,classif.PART,cluster.Cobweb,cluster.EM,cluster.FarthestFirst,cluster.SimpleKMeans,cluster.XMeans,regr.bartMachine,regr.extraTrees,regr.IBk
    $> Check ?learners to see which packages you need or install mlr with all suggestions.

    $> [[1]]
    $> Learner cluster.cmeans from package e1071,clue
    $> Type: cluster
    $> Name: Fuzzy C-Means Clustering; Short name: cmeans
    $> Class: cluster.cmeans
    $> Properties: numerics,prob
    $> Predict-Type: response
    $> Hyperparameters: centers=2
    $> 
    $> 
    $> [[2]]
    $> Learner cluster.dbscan from package fpc
    $> Type: cluster
    $> Name: DBScan Clustering; Short name: dbscan
    $> Class: cluster.dbscan
    $> Properties: numerics
    $> Predict-Type: response
    $> Hyperparameters: eps=1
