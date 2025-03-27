全体の概要は https://kasys.esa.io/posts/5508 に記述

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


