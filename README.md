# 概要
ToTクエリにおける、確信度を考慮した検索をColBERTを用いて実装

# 実装
1.確信度を含む単語の検出
2.確信度の判定
3.係り受け関係の判定
4.確信度と重み調整関数の設定

今のところ、
1.2.は辞書ベース
3.は埋め込み作成時のアテンションを利用
4.は調整中
※追記: 1~3をGPT-4で行う方法を追加

# コード
https://github.com/stanford-futuredata/ColBERT を利用

全体の流れ
[エンコード]
retrieval.py
→search_all→_search_all_Q→encode (searcher.py)
→queryFromText→query (checkpoint.py)
A→tensorize (query_tokenization.py)
B→super.query (colbert.py)

[スコア計算]
retrieval.py
→search_all→_search_all_Q→dense_search (searcher.py)
→rank→score_pids (index_storage.py)
→colbert_score_packed (colbert.py)


1
トークナイズ直後に行う。tensorize内でdictを作り検出、confidence_layer(shape=[クエリ数, 128])を返り値に追加

2
辞書ベースなので1と同時に行う

3
アテンションを利用して係り受け関係を判定(super.queryでエンコードするついでにアテンションを取得)
参考: What Does BERT Look At? https://kasys.esa.io/posts/5512

これを利用して1,2で使ったconfidence_layerを書き換え(queryFromText)
ColBERTv2は12layer, 12headだが、そのうちlayer=7,head=0を利用

思ったよりは上手くいっていそう？
I’m trying to remember the name of a TV show that I [probably] watched as a kid. It had a group of friends, and I think they were always solving some kind of mystery, or [maybe] it was a problem in the neighborhood? I’m not sure if it was set in the city or the countryside, but I do remember that one of the characters was really quirky and always had some kind of gadget. Does this sound familiar?

Top 7 Attention Scores for probably:
1. Token: as | Score: 0.2622
2. Token: kid | Score: 0.1011
3. Token: watched | Score: 0.0891
4. Token: it | Score: 0.0431
5. Token: that | Score: 0.0214
6. Token: had | Score: 0.0150
7. Token: and | Score: 0.0086

Top 7 Attention Scores for maybe:
1. Token: was | Score: 0.2438
2. Token: in | Score: 0.1622
3. Token: it | Score: 0.1490
4. Token: problem | Score: 0.0772
5. Token: neighborhood | Score: 0.0423
6. Token: it | Score: 0.0223
7. Token: or | Score: 0.0151

追記:
1~3をGPT-4で行う方法
tensorize関数を書き換え。手順3については、確信度作成時に割り振りまで行われるため削除(コメントアウトしてる)
tensorizeでトークンの分割を行った後、OpenAI APIを利用してトークンごとに確信度を取得
→トークン列をハッシュ化してqueries_cache.jsonlに保存、2回目以降はそれを利用
- こっちの方がスコアが全然いい
- 150クエリで$10くらい

4
colbert_score_packedで本来行列積が計算されている
calculate_dot_product_with_confidence(colbert.py)で確信度の重みづけ
- 現状の重み付け関数
https://www.geogebra.org/graphing/zwupkakz

# 前処理など


# リンク
実験メモ(エラー関連) https://kasys.esa.io/posts/5511
実験メモ1 https://kasys.esa.io/posts/5509
実験メモ2 https://kasys.esa.io/posts/5510

(補足など)
retrieval時の(ハイパーパラメータ指定とかの)コマンド https://kasys.esa.io/posts/5777
データ(corpus, qrel)のナンバリングについて

# 検索の実行
コマンドライン引数によって確信度の適用などのオプションが設定可能。

- python retrieval.py
    - 確信度は非適用(デフォルトのcolbertと同じ)

- python retrieval.py --confidence_search True
    - 確信度適用、パラメータはデフォルトを利用

- python retrieval.py --confidence_search True --confidence_params 0 1 1
    - 確信度適用、パラメータも指定

## 補足・注意点など
- 重み付け関数の切り替えはコマンドライン引数に入れていないため、手動で切り替えが必要
    - ~/colbert/modeling/colbert.pyの222行目
scores_conf = apply_confidence(scores, confidence_layer, confidence_params)
を書き換える。apply_confidenceが関数A、apply_confidence2が関数B、apply_confidence3が関数Cに対応

- パラメータの数は重み付け関数によって異なるので注意。詳細は上記関数のコードを参照

- 新しいクエリを用いる場合、初回にOpenAI APIを利用した確信度の割り当てが行われる。~/colbert/modeling/tokenization/query_tokenization.pyのConfidenceCalculatorクラス内でAPIキーの設定が必要
    - クエリ文（とトークンへの分割）が同一の場合、2回目以降はキャッシュされるので必要なし

# 検索結果の評価
結果結果とqrelを用意して以下を実行
- python -m utility.evaluate.msmarco_passages --ranking <検索結果のpath> --qrels <qrelのpath>

- ColBERTの仕様上、TRECの文書idをそのまま処理することができなかった。TRECでは
文書idを1,2…といった連番にしなければいけなかった
    - そのため、文書idを置換した。それに伴い、qrelも書き換えた（クエリidは変更なし）
    - したがって、新しくコーパスやqrelファイルを利用する際は書き換えの必要がある


