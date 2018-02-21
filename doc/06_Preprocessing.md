2018-02-21

データの前処理
==============

データの前処理というのは、学習アルゴリズムを適用する前にデータに施すあらゆる種類の変換のことだ。例えば、データの矛盾の発見と解決、欠損値への代入、外れ値の特定・除去・置換、数値データの離散化、カテゴリカルデータからのダミー変数の生成、標準化やBox-Cox変換などのあらゆる種類の変換、次元削減、特徴量の抽出・選択などが含まれる。

`mlr`は前処理に関して幾つかの選択肢を用意している。以下に示すようなタスク(あるいはデータフレーム)を変更する単純な手法の中には、タスクについての説明で既に触れたものもある。

-   `capLargeValues`: 大きな値や無限大の値の変換。
-   `createDummyFeature`: 因子型特徴量からのダミー変数の生成。
-   `dropFeatures`: 特徴量の削除。
-   `joinClassLevels`: (分類のみ)複数のクラスを併合して、大きな1つのクラスにする。
-   `mergeSmallFactorLevels`: 因子型特徴量において、例数の少ない水準を併合する。
-   `normalizeFeatures`: 正規化には複数の異なったやり方がある。標準化や特定の範囲への再スケールなど。
-   `removeConstantFeatures`: 1つの値しか持っていない特徴量(=定数)を除去する。
-   `subsetTask`: 観測値や特徴量をタスクから除去する。

また、以下のものについてはチュートリアルを用意してある。

-   特徴量選択
-   欠損値への代入

前処理と学習器を融合する
------------------------

`mlr`のラッパー機能により、学習器と前処理を組み合わせることができる。これは、前処理が学習器に属し、訓練や予測の度に実行されるということを意味する。

このようにすることで非常に便利な点がある。データやタスクの変更なしに、簡単に学習器と前処理の組合せを変えることができるのだ。

また、これは前処理を行ってから学習器のパフォーマンスを測定する際にありがちな一般的な間違いを避けることにもつながる。前処理は学習アルゴリズムとは完全に独立したものだと考えられがちだ。学習器のパフォーマンスを測定する場合を考えてみよう。例えば、クロスバリデーションで雨処理を事前にデータセット全体に対して行い、学習と予測は学習器だけで行うような場合だ。前処理として何が行われたかによっては、評価が楽観的になる危険性がある。例えば、(欠損値への)平均値の代入という前処理が学習器の性能評価前に、データ全体を対象に行われたとすると、これは楽観的なパフォーマンス評価につながる。

前処理にはデータ依存的なものとデータ非依存的なものがあることをはっきりさせておこう。データ依存的な前処理とは、前処理のやり方がデータに依存しており、データセットが異なれば結果も異なるというようなもののことだ。一方でデータ非依存的な前処理は常に結果が同じになる。

データの間違いを修正したり、ID列のような学習に使うべきではないデータ列の除去のような前処理は、明らかにデータ非依存的である。一方、先程例に挙げた欠損値への平均値の代入はデータ依存的である。代入を定数で行うのであれば違うが。

前処理と組み合わせた学習器の性能評価を正しく行うためには、全てのデータ依存的な前処理をリサンプリングに含める必要がある。学習器と前処理を融合させれば、これは自動的に可能になる。

この目的のために、`mlr`パッケージは2つのラッパーを用意している。

-   `makePreprocWrapperCaret`は`caret`パッケージの`preProcess`関数に対するインターフェースを提供するラッパー。
-   `makePreprocWrapper`を使えば、訓練と予測の前の動作を定義することで独自の前処理を作成できる。

これらを使用する前処理は、`normalizeFeatures`などを使う前処理とは異なり、ラップされた学習器に組み込まれる。

-   タスクそのものは変更されない。
-   前処理はデータ全体に対して予め行われるのではなく、リサンプリングなど、訓練とテストの対が発生する毎に実行される。
-   前処理に関わる制御可能なパラメータは、学習器のパラメータと一緒に調整できる。

まずは`makePreprocWrapperCaret`の例から見ていこう。

`makePreprocWrapperCaret`を使用した前処理
-----------------------------------------

`makePreprocWrapperCaret`は`caret`パッケージの`preProcess`関数へのインターフェースだ。`PreProcess`関数は、欠損値への代入やスケール変換やBox-Cox変換、独立主成分分析による次元削減など、様々な手法を提供する関数だ。具体的に何が可能かは`preProcess`関数のヘルプページ([preProcess function | R Documentation](https://www.rdocumentation.org/packages/caret/versions/6.0-78/topics/preProcess))を確認してもらいたい。

まず、`makePreprocWrapperCaret`と`preProcess`の違いを確認しておこう。

-   `makePreprocWrapperCaret`は`preProcess`とほぼ同じ仮引数を持つが、仮引数名に`ppc.`というプレフィックスが付く。
-   上記の例外は`method`引数だ。この引数は`makePreprocWrapperCaret`には無い。その代わりに、本来`method`に渡す前処理に関するオプションは、対応する仮引数に論理値を指定することで制御する。

例を見よう。`preProcess`では行列またはデータフレーム`x`に対して、次のように前処理を行う。

``` r
preProcess(x, method= c("knnInpute", "pca"), pcaComp = 10)
```

一方、`makePreporcWrapperCaret`では、`Learner`クラスのオブジェクトまたはクラスの名前(`"classif.lda"`など)を引数にとって、次のように前処理を指定する。

``` r
makePreprocWrapperCaret(learner, ppc.knnImpute = TRUE, ppc.pca = TRUE, ppc.pcaComp = 10)
```

この例のように複数の前処理(注: kNNを使った代入と主成分分析)を有効にした場合、それらは特定の順序で実行される。詳細は`preProcess`関数のヘルプを確認してほしい(訳注: Details後半の"The operations are applied in this order:..."以下。主成分分析は代入後に実施。)。

以下に主成分分析による次元削減の例を示そう。これは無闇に使用して良い手法ではないが、高次元のデータで問題が起こるような学習器や、データの回転が有用な学習器に対しては有効である。

例では`soner.task`を用いる。これは208の観測値と60の特徴量を持つ。

``` r
sonar.task
```

    $> Supervised task: Sonar-example
    $> Type: classif
    $> Target: Class
    $> Observations: 208
    $> Features:
    $> numerics  factors  ordered 
    $>       60        0        0 
    $> Missings: FALSE
    $> Has weights: FALSE
    $> Has blocking: FALSE
    $> Classes: 2
    $>   M   R 
    $> 111  97 
    $> Positive class: M

今回は、`MASS`パッケージによる二次判別分析と、主成分分析による前処理を組み合わせる。また、閾値として0.9を設定する。これはつまり、主成分が累積寄与率90%を保持しなければならないという指示になる。データは主成分分析の前に自動的に標準化される。

``` r
lrn = makePreprocWrapperCaret("classif.qda", ppc.pca = TRUE, ppc.thresh = 0.9)
lrn
```

    $> Learner classif.qda.preproc from package MASS
    $> Type: classif
    $> Name: ; Short name: 
    $> Class: PreprocWrapperCaret
    $> Properties: twoclass,multiclass,numerics,factors,prob
    $> Predict-Type: response
    $> Hyperparameters: ppc.BoxCox=FALSE,ppc.YeoJohnson=FALSE,ppc.expoTrans=FALSE,ppc.center=TRUE,ppc.scale=TRUE,ppc.range=FALSE,ppc.knnImpute=FALSE,ppc.bagImpute=FALSE,ppc.medianImpute=FALSE,ppc.pca=TRUE,ppc.ica=FALSE,ppc.spatialSign=FALSE,ppc.thresh=0.9,ppc.na.remove=TRUE,ppc.k=5,ppc.fudge=0.2,ppc.numUnique=3

ラップされた学習器を`soner.task`によって訓練する。訓練したモデルを確認することで、22の主成分が訓練に使われたことがわかるだろう。

``` r
mod = train(lrn, sonar.task)
mod
```

    $> Model for learner.id=classif.qda.preproc; learner.class=PreprocWrapperCaret
    $> Trained on: task.id = Sonar-example; obs = 208; features = 60
    $> Hyperparameters: ppc.BoxCox=FALSE,ppc.YeoJohnson=FALSE,ppc.expoTrans=FALSE,ppc.center=TRUE,ppc.scale=TRUE,ppc.range=FALSE,ppc.knnImpute=FALSE,ppc.bagImpute=FALSE,ppc.medianImpute=FALSE,ppc.pca=TRUE,ppc.ica=FALSE,ppc.spatialSign=FALSE,ppc.thresh=0.9,ppc.na.remove=TRUE,ppc.k=5,ppc.fudge=0.2,ppc.numUnique=3

``` r
getLearnerModel(mod)
```

    $> Model for learner.id=classif.qda; learner.class=classif.qda
    $> Trained on: task.id = Sonar-example; obs = 208; features = 22
    $> Hyperparameters:

``` r
getLearnerModel(mod, more.unwrap = TRUE)
```

    $> Call:
    $> qda(f, data = getTaskData(.task, .subset, recode.target = "drop.levels"))
    $> 
    $> Prior probabilities of groups:
    $>         M         R 
    $> 0.5336538 0.4663462 
    $> 
    $> Group means:
    $>          PC1        PC2        PC3         PC4         PC5         PC6
    $> M  0.5976122 -0.8058235  0.9773518  0.03794232 -0.04568166 -0.06721702
    $> R -0.6838655  0.9221279 -1.1184128 -0.04341853  0.05227489  0.07691845
    $>          PC7         PC8        PC9       PC10        PC11          PC12
    $> M  0.2278162 -0.01034406 -0.2530606 -0.1793157 -0.04084466 -0.0004789888
    $> R -0.2606969  0.01183702  0.2895848  0.2051963  0.04673977  0.0005481212
    $>          PC13       PC14        PC15        PC16        PC17        PC18
    $> M -0.06138758 -0.1057137  0.02808048  0.05215865 -0.07453265  0.03869042
    $> R  0.07024765  0.1209713 -0.03213333 -0.05968671  0.08528994 -0.04427460
    $>          PC19         PC20        PC21         PC22
    $> M -0.01192247  0.006098658  0.01263492 -0.001224809
    $> R  0.01364323 -0.006978877 -0.01445851  0.001401586

二次判別分析について、主成分分析を使う場合と使わない場合をベンチマーク試験により比較してみよう。今回の例では各クラスの例数が少ないので、二次判別分析の際のエラーを防ぐためにリサンプリングにおいて層別サンプリングを行っている。

``` r
rin = makeResampleInstance("CV", iters = 3, stratify = TRUE, task = sonar.task)
res = benchmark(list("classif.qda", lrn), sonar.task, rin, show.info = FALSE)
res
```

    $>         task.id          learner.id mmce.test.mean
    $> 1 Sonar-example         classif.qda      0.3505176
    $> 2 Sonar-example classif.qda.preproc      0.2213251

今回の場合では、二次判別分析に対して主成分分析による前処理が効果的だったことがわかる。

前処理オプションと学習器パラメータの連結チューニング
----------------------------------------------------

今の例をもう少し最適化できないか考えてみよう。今回、任意に設定した0.9という閾値によって、主成分は22になった。しかし、主成分の数はもっと多いほうが良いかもしれないし、少ないほうが良いかもしれない。また、`qda`関数にはクラス共分散行列やクラス確率の推定方法を制御するためのいくつかのオプションがある。

学習機と前処理のパラメータは、連結してチューニングすることができる。まずは、ラップされた学習器の全てのパラメータを`getParamSet`関数で確認してみよう。

``` r
getParamSet(lrn)
```

    $>                      Type len     Def                      Constr Req
    $> ppc.BoxCox        logical   -   FALSE                           -   -
    $> ppc.YeoJohnson    logical   -   FALSE                           -   -
    $> ppc.expoTrans     logical   -   FALSE                           -   -
    $> ppc.center        logical   -    TRUE                           -   -
    $> ppc.scale         logical   -    TRUE                           -   -
    $> ppc.range         logical   -   FALSE                           -   -
    $> ppc.knnImpute     logical   -   FALSE                           -   -
    $> ppc.bagImpute     logical   -   FALSE                           -   -
    $> ppc.medianImpute  logical   -   FALSE                           -   -
    $> ppc.pca           logical   -   FALSE                           -   -
    $> ppc.ica           logical   -   FALSE                           -   -
    $> ppc.spatialSign   logical   -   FALSE                           -   -
    $> ppc.thresh        numeric   -    0.95                    0 to Inf   -
    $> ppc.pcaComp       integer   -       -                    1 to Inf   -
    $> ppc.na.remove     logical   -    TRUE                           -   -
    $> ppc.k             integer   -       5                    1 to Inf   -
    $> ppc.fudge         numeric   -     0.2                    0 to Inf   -
    $> ppc.numUnique     integer   -       3                    1 to Inf   -
    $> ppc.n.comp        integer   -       -                    1 to Inf   -
    $> method           discrete   -  moment            moment,mle,mve,t   -
    $> nu                numeric   -       5                    2 to Inf   Y
    $> predict.method   discrete   - plug-in plug-in,predictive,debiased   -
    $>                  Tunable Trafo
    $> ppc.BoxCox          TRUE     -
    $> ppc.YeoJohnson      TRUE     -
    $> ppc.expoTrans       TRUE     -
    $> ppc.center          TRUE     -
    $> ppc.scale           TRUE     -
    $> ppc.range           TRUE     -
    $> ppc.knnImpute       TRUE     -
    $> ppc.bagImpute       TRUE     -
    $> ppc.medianImpute    TRUE     -
    $> ppc.pca             TRUE     -
    $> ppc.ica             TRUE     -
    $> ppc.spatialSign     TRUE     -
    $> ppc.thresh          TRUE     -
    $> ppc.pcaComp         TRUE     -
    $> ppc.na.remove       TRUE     -
    $> ppc.k               TRUE     -
    $> ppc.fudge           TRUE     -
    $> ppc.numUnique       TRUE     -
    $> ppc.n.comp          TRUE     -
    $> method              TRUE     -
    $> nu                  TRUE     -
    $> predict.method      TRUE     -

`ppc.`というプレフィックスのついたものが前処理のパラメータで、他が`qda`関数のパラメータだ。主成分分析の閾値を`ppc.thresh`を使って調整する代わりに、主成分の数そのものを`ppc.pcaComp`を使って調整できる。さらに、`qda`関数に対しては、2種類の事後確率推定法(通常のプラグイン推定と不偏推定)を試してみよう。

今回は解像度10でグリッドサーチを行ってみよう。もっと解像度を高くしたくなるかもしれないが、今回はあくまでデモだ。

``` r
ps = makeParamSet(
  makeIntegerParam("ppc.pcaComp", lower = 1, upper = getTaskNFeats(sonar.task)),
  makeDiscreteParam("predict.method", values = c("plug-in", "debiased"))
)
ctrl = makeTuneControlGrid(resolution = 10)
res = tuneParams(lrn, sonar.task, rin, par.set = ps, control = ctrl, show.info = FALSE)
res
```

    $> Tune result:
    $> Op. pars: ppc.pcaComp=21; predict.method=plug-in
    $> mmce.test.mean=0.212

``` r
as.data.frame(res$opt.path)[1:3]
```

    $>    ppc.pcaComp predict.method mmce.test.mean
    $> 1            1        plug-in      0.5284334
    $> 2            8        plug-in      0.2311939
    $> 3           14        plug-in      0.2118703
    $> 4           21        plug-in      0.2116632
    $> 5           27        plug-in      0.2309869
    $> 6           34        plug-in      0.2739821
    $> 7           40        plug-in      0.2933057
    $> 8           47        plug-in      0.3029676
    $> 9           53        plug-in      0.3222912
    $> 10          60        plug-in      0.3505176
    $> 11           1       debiased      0.5579020
    $> 12           8       debiased      0.2502415
    $> 13          14       debiased      0.2503796
    $> 14          21       debiased      0.2550725
    $> 15          27       debiased      0.2792271
    $> 16          34       debiased      0.3128364
    $> 17          40       debiased      0.2982747
    $> 18          47       debiased      0.2839199
    $> 19          53       debiased      0.3224983
    $> 20          60       debiased      0.3799172

`"plug-in"`と`"debiased"`のいずれでも少なめ(27以下)の主成分が有効で、`"plug-in"`の方が若干エラー率が低いようだ。

独自の前処理ラッパーを書く
--------------------------

`makePreprocWrapperCaret`で不満があれば、`makePreprocWrapper`関数で独自の前処理ラッパーを作成できる。

ラッパーに関するチュートリアルでも説明しているが、ラッパーは**訓練**と**予測**という2つのメソッドを使って実装される。前処理ラッパーの場合は、メソッドは学習と予測の前に何をするかを指定するものであり、これは完全にユーザーが指定する。

以下に例として、訓練と予測の前にデータの中心化とスケーリングを行うラッパーの作成方法を示そう。k最近傍法やサポートベクターマシン、ニューラルネットワークなどは通常スケーリングされた特徴量を必要とする。多くの組み込みスケーリング手法は、データセットを事前にスケーリングし、テストデータセットもそれに従ってスケーリングされる。以下では、学習器にスケーリングオプションを追加し、`scale`関数と組み合わせる方法を示す。

今回この単純な例を選んだのはあくまで説明のためだ。中心化とスケーリングは`makePreprocWrapperCaret`でも可能だということに注意してほしい。

### 訓練関数の指定

**訓練**(ステップで使う)関数は以下の引数を持つ関数でなければならない。

-   `data`: 全ての特徴量と目的変数を列として含むデータフレーム。
-   `target`: `data`に含まれる目的変数の名前。
-   `args`: 前処理に関わるその他の引数とパラメータのリスト。

この関数は`$data`と`$control`を要素として持つリストを返す必要がある。`$data`は前処理されたデータセットを、`$control`には予測のために必要な全ての情報を格納する。

スケーリングのための訓練関数の定義例を以下に示す。これは数値型の特徴量に対して`scale`関数を呼び出し、スケーリングされたデータと関連するスケーリングパラメータを返す。

`args`は`scale`関数の引数である`center`と`scale`引数を含み、予測で使用するためにこれを`$control`スロットに保持する。これらの引数は、論理値または数値型の列の数に等しい長さの数値型ベクトルで指定する必要がある。`center`引数は数値を渡された場合にはその値を各データから引くが、`TRUE`が指定された場合には平均値を引く。`scale`引数は数値を渡されるとその値で各データを割るが、`TRUE`の場合は標準偏差か二乗平均平方根を引く(いずれになるかは`center`引数に依存する)。2つの引数のいずれかor両方に`TRUE`が指定された場合には、この値を予測の段階で使用するためには返り値の`$control`スロットに保持しておく必要があるという点に注意しよう。

``` r
trainfun = function(data, target, args = list(center, scale)){
  ## 数値特徴量を特定する
  cns = colnames(data)
  nums = setdiff(cns[sapply(data, is.numeric)], target)
  ## 数値特徴量を抽出し、scale関数を呼び出す
  x = as.matrix(data[, nums, drop = FALSE])
  x = scale(x, center = args$center, scale = args$scale)
  ## スケーリングパラメータを後で予測に使うためにcontrolに保持する
  control = args
  if(is.logical(control$center) && control$center){
    control$center = attr(x, "scaled:center")
  }
  if(is.logical(control$scale) && control$scale){
    control$scale = attr(x, "scaled:scale")
  }
  ## 結果をdataにまとめる
  data = data[, setdiff(cns, nums), drop = FALSE]
  data = cbind(data, as.data.frame(x))
  return(list(data = data, control = control))
}
```

### 予測関数の指定

**予測**(ステップで使う)関数は以下の引数を持つ必要がある。

-   `data`: 特徴量のみをもつデータフレーム。(予測ステップでは目的変数の値は未知なのが普通だ。)
-   `target`: 目的変数の名前。
-   `args`: 訓練関数に渡された`args`。
-   `control`: 訓練関数が返したもの。

この関数は前処理済みのデータを返す。

今回の例では、予測関数は数値特徴量を訓練ステージで`control`に保持されたパラメータを使ってスケーリングする。

``` r
predictfun = function(data, target, args, control){
  ## 数値特徴量の特定
  cns = colnames(data)
  nums = cns[sapply(data, is.numeric)]
  ## データから数値特徴量を抽出してscale関数を適用する
  x = as.matrix(data[, nums, drop = FALSE])
  x = scale(x, center = control$center, scale = control$scale)
  ## dataにまとめて返す
  data = data[, setdiff(cns, nums), drop = FALSE]
  data = cbind(data, as.data.frame(x))
  return(data)
}
```

### 前処理ラッパーの作成

以下では、ニューラルネットワークによる回帰(これは自前のスケーリングオプションを持たない)をベースの学習器として前処理ラッパーを作成する。

先に定義した**訓練**および**予測**関数を`makePreprocWrapper`関数の`train`と`predict`引数に渡す。`par.vals`には、訓練関数の`args`に渡すパラメータをリストとして渡す。

``` r
lrn = makeLearner("regr.nnet", trace = FALSE, decay = 1e-02)
lrn = makePreprocWrapper(lrn, train = trainfun, predict = predictfun,
                         par.vals = list(center = TRUE, scale = TRUE))
```

データセット`BostonHousing`を対象にして、スケーリングの有無による平均二乗誤差の違いを確認してみよう。

``` r
rdesc = makeResampleDesc("CV", iters = 3)

## スケーリングあり(上で前処理を指定した)
r = resample(lrn, bh.task, resampling = rdesc, show.info = FALSE)
r
```

    $> Resample Result
    $> Task: BostonHousing-example
    $> Learner: regr.nnet.preproc
    $> Aggr perf: mse.test.mean=  18
    $> Runtime: 0.137429

``` r
## 前処理無しの学習器を再度作る
lrn = makeLearner("regr.nnet", trace = FALSE, decay = 1e-02)
r = resample(lrn, bh.task, resampling = rdesc, show.info = FALSE)
r
```

    $> Resample Result
    $> Task: BostonHousing-example
    $> Learner: regr.nnet
    $> Aggr perf: mse.test.mean=41.5
    $> Runtime: 0.101203

### 前処理と学習器のパラメータを連結してチューニングする

前処理のオプションをどのように設定すれば特定のアルゴリズムに対して上手くいくのかということは、明確には分からないことが多い。`makePreprocWrapperCaret`の例で、既に前処理と学習器のパラメータを両方ともチューニングする方法を既に見た。

スケーリングの例では、ニューラルネットに対してスケーリングと中心化の両方を行うのが良いのか、いずれか片方なのか、あるいは行わないほうが良いのかという点を確認することができる。`center`と`scale`をチューニングするためには、適切な種類の`LearnerParam`をパラメータセットに追加する必要がある。

前述のように、`center`と`scale`には数値型か論理値型のいずれかを指定できるが、今回は論理値型のパラメータとしてチューニングしよう。

``` r
lrn = makeLearner("regr.nnet", trace = FALSE)
lrn = makePreprocWrapper(lrn, train = trainfun, predict = predictfun,
                         par.set = makeParamSet(
                           makeLogicalLearnerParam("center"),
                           makeLogicalLearnerParam("scale")
                         ),
                         par.vals = list(center = TRUE, scale = TRUE))
lrn
```

    $> Learner regr.nnet.preproc from package nnet
    $> Type: regr
    $> Name: ; Short name: 
    $> Class: PreprocWrapper
    $> Properties: numerics,factors,weights
    $> Predict-Type: response
    $> Hyperparameters: size=3,trace=FALSE,center=TRUE,scale=TRUE

今回はグリッドサーチで`nnet`の`decay`パラメータと`scale`の`center`と`scale`パラメータをチューニングする。

``` r
rdesc = makeResampleDesc("Holdout")
ps = makeParamSet(
  makeDiscreteLearnerParam("decay", c(0, 0.05, 0.1)),
  makeLogicalLearnerParam("center"),
  makeLogicalLearnerParam("scale")
)
crrl = makeTuneControlGrid()
res = tuneParams(lrn, bh.task, rdesc, par.set = ps, control = ctrl, show.info = FALSE)
res
```

    $> Tune result:
    $> Op. pars: decay=0.05; center=TRUE; scale=TRUE
    $> mse.test.mean=11.2

``` r
as.data.frame(res$opt.path)
```

    $>    decay center scale mse.test.mean dob eol error.message exec.time
    $> 1      0   TRUE  TRUE      57.95746   1  NA          <NA>     0.039
    $> 2   0.05   TRUE  TRUE      11.23583   2  NA          <NA>     0.042
    $> 3    0.1   TRUE  TRUE      15.44886   3  NA          <NA>     0.043
    $> 4      0  FALSE  TRUE      84.89302   4  NA          <NA>     0.019
    $> 5   0.05  FALSE  TRUE      16.63278   5  NA          <NA>     0.041
    $> 6    0.1  FALSE  TRUE      13.80628   6  NA          <NA>     0.043
    $> 7      0   TRUE FALSE      64.98619   7  NA          <NA>     0.029
    $> 8   0.05   TRUE FALSE      55.94930   8  NA          <NA>     0.040
    $> 9    0.1   TRUE FALSE      26.67453   9  NA          <NA>     0.048
    $> 10     0  FALSE FALSE      63.27422  10  NA          <NA>     0.023
    $> 11  0.05  FALSE FALSE      34.35454  11  NA          <NA>     0.044
    $> 12   0.1  FALSE FALSE      42.57609  12  NA          <NA>     0.043

### 前処理ラッパー関数

よい前処理ラッパーを作成したのであれば、それを関数としてカプセル化するのは良いアイデアだ。他の人も使えると便利だろうから`mlr`に追加して欲しい、というのであれば[Issues · mlr-org/mlr](https://github.com/mlr-org/mlr/issues)からコンタクトして欲しい。

``` r
makePreprocWrapperScale = function(learner, center = TRUE, scale = TRUE) {
  trainfun = function(data, target, args = list(center, scale)) {
    cns = colnames(data)
    nums = setdiff(cns[sapply(data, is.numeric)], target)
    x = as.matrix(data[, nums, drop = FALSE])
    x = scale(x, center = args$center, scale = args$scale)
    control = args
    if (is.logical(control$center) && control$center)
      control$center = attr(x, "scaled:center")
    if (is.logical(control$scale) && control$scale)
      control$scale = attr(x, "scaled:scale")
    data = data[, setdiff(cns, nums), drop = FALSE]
    data = cbind(data, as.data.frame(x))
    return(list(data = data, control = control))
  }
  predictfun = function(data, target, args, control) {
    cns = colnames(data)
    nums = cns[sapply(data, is.numeric)]
    x = as.matrix(data[, nums, drop = FALSE])
    x = scale(x, center = control$center, scale = control$scale)
    data = data[, setdiff(cns, nums), drop = FALSE]
    data = cbind(data, as.data.frame(x))
    return(data)
  }
  makePreprocWrapper(
    learner,
    train = trainfun,
    predict = predictfun,
    par.set = makeParamSet(
      makeLogicalLearnerParam("center"),
      makeLogicalLearnerParam("scale")
    ),
    par.vals = list(center = center, scale = scale)
  )
}

lrn = makePreprocWrapperScale("classif.lda")
train(lrn, iris.task)
```

    $> Model for learner.id=classif.lda.preproc; learner.class=PreprocWrapper
    $> Trained on: task.id = iris-example; obs = 150; features = 4
    $> Hyperparameters: center=TRUE,scale=TRUE
