import torch
from tqdm import tqdm
from openai import OpenAI
import re

# class ConfidenceCalculator用
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from colbert.modeling.hf_colbert import class_factory
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches, _split_into_batches_with_confidence_layer, _insert_prefix_token
from colbert.parameters import DEVICE

class ConfidenceCalculator:
    def __init__(self):
        self.cache_path = Path("/home/aki235/ColBERT_conf/data/TREC_TOT/dev2-2024/queries_cache_v1_実験2.jsonl")
        # メモリ効率のため、ハッシュ値のインデックスをメモリに保持
        self.hash_index = self._build_hash_index()
        self.api_key = 'API_KEY_HERE'
        self.client = OpenAI(api_key=self.api_key)
    
    def _build_hash_index(self) -> Dict[str, int]:
        # ハッシュ値から行番号へのマッピングを作成
        index = {}
        with open(self.cache_path, 'r') as f:
            for line_num, line in enumerate(f):
                entry = json.loads(line)
                index[entry['hash']] = line_num
        return index
    
    def get_hash(self, tokens: List[str]) -> str:
        # トークン分割パターンからハッシュ値を生成
        content = '|'.join(tokens)
        return hashlib.md5(content.encode()).hexdigest()
    
    # def create_confidence_prompt(self, tokens: List[str]):
    #     """Create prompt for confidence assignment"""
    #     return f"""Analyze these tokens from a tip-of-the-tongue search query and assign confidence scores (0-1) for how likely each token is to be "the word the user is trying to remember".

    # Tokens: {' '.join(tokens)}

    # Guidelines:
    # 1.0: Core terms the user might be searching for (specific nouns, key verbs, distinctive adjectives)
    # 0.8-0.9: Important contextual words
    # 0.4-0.7: Supporting content words
    # 0.1-0.3: Function words (articles, prepositions)
    # 0.0: Special tokens, punctuation, spaces

    # **Output ONLY a raw JSON array of numbers** representing confidence scores for each token, matching the exact number of input tokens. Strictly follow this format with no additional text, code blocks, or annotations.

    # Example format:
    # [
    #     ("[CLS]", 0.0), 
    #     ("Quick", 0.5), 
    #     ("brown", 0.8), 
    #     ("fox", 0.9), 
    #     ...
    # ]"""

    # def get_confidence_from_llm(self, tokens: List[str]):
    #     # GPT-4を使用してトークンごとの確信度を取得

    #     def fetch_confidences():
    #         response = self.client.chat.completions.create(
    #             model="gpt-4-turbo",
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant."},
    #                 {"role": "user", "content": self.create_confidence_prompt(tokens)}
    #             ],
    #             temperature=0.2  # 一貫性のため低めの温度を設定
    #         )
    #         response = response.choices[0].message.content.strip()
    #         confidences = [float(num) for num in re.findall(r"\(\s*\"[^\"]+\"\s*,\s*([0-9.]+)\s*\)", response)]
    #         return confidences

    #     # 初回の呼び出し
    #     confidences = fetch_confidences()

    #     # 長さが一致しない場合に再試行
    #     if len(confidences) != len(tokens):
    #         confidences = fetch_confidences()

    #     # 再試行しても長さが一致しなければデフォルト値を返す
    #     if len(confidences) != len(tokens):
    #         confidences = [0.0] * len(tokens)

    #     return confidences

    def create_confidence_prompt(self, tokens: List[str]):
        return f"""Analyze these tokens from a tip-of-the-tongue search query and assign confidence scores (0-1) for how confident the writer appears to be about each piece of information.

    Tokens:
    {tokens}

    Guidelines for assessing writer's confidence:
    1.0: Facts stated with absolute certainty (e.g., "movie", "planet" when clearly remembered)
    0.8-0.9: Information writer seems very confident about but not explicitly confirmed
    0.4-0.7: Details writer seems somewhat unsure about (often indicated by phrases like "I think", "probably", "might be")
    0.1-0.3: Information writer expresses significant uncertainty about
    0.0: Special tokens, punctuation, spaces, and information not related to confidence assessment

    Pay special attention to:
    - Hedging phrases ("I think", "probably", "maybe")
    - Time markers that suggest memory uncertainty ("back in the 90's")
    - Specific vs. vague descriptions
    - Corrections or alternatives provided
    - Questioning phrases ("what's the name?")

    **Output ONLY a JSON array in the following format**. Ensure the output array has exactly the same number of scores as there are tokens, and each score should be in the same order as the tokens.

    Expected format:
    [
        (token1, score1),
        (token2, score2),
        ...
    ]

    Example format:
    [
        ("[CLS]", 0.0),
        ("Quick", 0.5),
        ("brown", 0.8),
        ("fox", 0.9)
    ]"""

    def create_confidence_prompt2(self, tokens:List[str]):
        prompt = f"""Analyze these tokens from a tip-of-the-tongue search query and assign confidence scores (0.0~1.0 or -1.0) for how confident the writer appears to be about each piece of information.

    Tokens:
    {tokens}

    Guidelines for assigning confidence scores (0-1) or -1:

    1.0: Facts stated with absolute certainty (e.g., "movie", "planet" when clearly remembered).
    0.8 - 0.9: Writer appears very confident but not explicitly confirmed.
    0.4 - 0.7: Writer is somewhat unsure (indicated by phrases like "I think", "might be", "probably" or uncertain recall).
    0.1 - 0.3: Writer expresses significant uncertainty (strong hedging, guesswork).
    -1.0: Noise or extraneous tokens that do not contribute meaning to the final answer: Punctuation (e.g., commas, periods, dashes, parentheses), Tokens that convey confidence/uncertainty themselves ('probably', 'maybe', 'I think', 'unsure', 'guessing')

    Requirements:
    - Ensure the output array has **exactly the same number of elements as the input tokens**.
    - The output array must preserve the **original token order**.
    - Assign a score to **every token**, even if it is noise or irrelevant.

    Pay special attention to:
    - Hedging phrases ("I think", "probably", "maybe")
    - Time markers that suggest memory uncertainty ("back in the 90's")
    - Specific vs. vague descriptions
    - Corrections or alternatives provided
    - Questioning phrases ("what's the name?")

    **Output ONLY a JSON array in the following format**. 

    Expected format:
    [
        (token1, score1),
        (token2, score2),
        ...
    ]

    Example format:
    [
        ("[CLS]", -1.0),
        ("[unused0]", -1.0),
        ("Quick", 0.5),
        ("brown", 0.8),
        ("fox", 0.9)
    ]"""
        return prompt

    def parse_api_response(self, response_str):
        # 空白文字を削除
        response_str = response_str.strip()

        # 最初と最後の角括弧を削除
        if response_str.startswith('[') and response_str.endswith(']'):
            response_str = response_str[1:-1]

        # 各タプルを抽出
        # pattern = r'\("([^"]+)",\s*([\d.]+)\)'
        pattern = r'\("([^"]+)",\s*(-?[\d.]+)\)'
        matches = re.findall(pattern, response_str)

        # マッチした結果をタプルのリストに変換
        result = []
        for token, score in matches:
            result.append((token, float(score)))

        return result

    def check_match(self, tokens, confidences):
        min_length = min(len(tokens), len(confidences))

        # 一致の確認（短い方の長さまで）
        for i in range(min_length):
            token = tokens[i].replace('\\', '\\\\')
            conf_token, confidence = confidences[i]
            if token != conf_token:
                print(f"不一致: インデックス {i} - tokensの値: '{token}' / confidencesの値: '{conf_token}'")
                return [0.0] * len(tokens)

        # 両方のリストが完全に一致する場合
        if len(tokens) == len(confidences):
            print("全てのトークンが一致しています。")
            res = [float(conf[1]) for conf in confidences]
            return res
        # 長さが一致しない場合
        else:
            # confidencesに余分な要素がある場合の処理
            if len(confidences) > len(tokens):
                extra_confidences = confidences[len(tokens):]

                if all(conf[0] == "[MASK]" for conf in extra_confidences):
                    # 余分な要素がすべて[MASK]である場合
                    print("confidencesに余分な[MASK]があります。削除して再チェックを行います")
                    confidences = confidences[:len(tokens)]
                    return self.check_match(tokens, confidences)
                else:
                    # 余分な要素が[MASK]以外の場合
                    extra_conf_tokens = [conf[0] for conf in extra_confidences]
                    print(f"confidencesに余分な要素があります: {extra_conf_tokens}")
                    return [0.0] * len(tokens)

            # tokensに余分な要素がある場合の処理
            elif len(tokens) > len(confidences):
                extra_tokens = tokens[len(confidences):]

                if all(token == "[MASK]" for token in extra_tokens):
                    # 余分な要素がすべて[MASK]である場合
                    print("tokensに余分な[MASK]があります。confidencesをパディングして再チェックを行います")
                    for _ in range(len(tokens) - len(confidences)):
                        confidences.append(("[MASK]", 0.0))
                    return self.check_match(tokens, confidences)
                else:
                    # 余分な要素が[MASK]以外の場合
                    print(f"tokensに余分な要素があります: {extra_tokens}")
                    return [0.0] * len(tokens)

    def get_confidence_from_llm(self, tokens: List[str]):#最後にself追加
        # GPT-4を使用してトークンごとの確信度を取得

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.create_confidence_prompt(tokens)}
            ],
            temperature=0.2  # 一貫性のため低めの温度を設定
        )
        response = response.choices[0].message.content.strip()
        confidences = self.parse_api_response(response)

        return self.check_match(tokens, confidences)

    def get_cache_entry(self, hash_value: str) -> Optional[Dict]:
        # ハッシュ値に対応するキャッシュエントリを取得
        if hash_value not in self.hash_index:
            return None
        
        line_num = self.hash_index[hash_value]
        with open(self.cache_path, 'r') as f:
            for i, line in enumerate(f):
                if i == line_num:
                    return json.loads(line)
        return None
    
    def append_to_cache(self, hash_value:str, tokens: List[str], confidences: List[float]):
        # キャッシュファイルに新しいエントリを追加
        entry = {
            "hash": hash_value,
            "tokens": tokens,
            "confidences": confidences
        }
        with open(self.cache_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # メモリ上のindexを更新
        self.hash_index[hash_value] = len(self.hash_index)

    def get_confidence(self, tokens: List[str]) -> List[float]:
        # トークン列に対する確信度を取得(1つのクエリごと)
        hash_value = self.get_hash(tokens)

        # キャッシュから検索
        cache_entry = self.get_cache_entry(hash_value)
        if cache_entry is not None:
            if cache_entry["tokens"] == tokens:
                print("キャッシュ内に一致するクエリが見つかりました")
                return cache_entry["confidences"]
        
        # キャッシュにない場合はLLMで計算
        confidences = self.get_confidence_from_llm(tokens)
        print("新しくキャッシュを追加しました")
        # キャッシュに追加
        self.append_to_cache(hash_value, tokens, confidences)
        return confidences
 
class QueryTokenizer():
    def __init__(self, config: ColBERTConfig, verbose: int = 3):
        HF_ColBERT = class_factory(config.checkpoint)
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.verbose = verbose

        self.config = config
        self.query_maxlen = config.query_maxlen
        self.background_maxlen = 512 - self.query_maxlen + 1  # FIXME: Make this configurable

        self.Q_marker_token, self.Q_marker_token_id = config.query_token, self.tok.convert_tokens_to_ids(config.query_token_id)
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.pad_token,self.pad_token_id = self.tok.pad_token,self.tok.pad_token_id
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False).to(DEVICE)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids
    
    def tensorize(self, batch_text, bsize=None, context=None, full_length_search=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # Full length search is only available for single inference (for now)
        # Batched full length search requires far deeper changes to the code base
        assert(full_length_search == False or (type(batch_text) == list and len(batch_text) == 1))

        if full_length_search:
            # Tokenize each string in the batch
            un_truncated_ids = self.tok(batch_text, add_special_tokens=False).to(DEVICE)['input_ids']
            # Get the longest length in the batch
            max_length_in_batch = max(len(x) for x in un_truncated_ids)
            # Set the max length
            max_length = self.max_len(max_length_in_batch)
        else:
            # Max length is the default max length from the config
            max_length = self.query_maxlen

        # tokenize with max_length - 1 to add the marker id afterwards
        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=(max_length - 1)).to(DEVICE)

        ids = _insert_prefix_token(obj['input_ids'], self.Q_marker_token_id)
        mask = _insert_prefix_token(obj['attention_mask'], 1)

        # postprocess for the [MASK] augmentation
        ids[ids == self.pad_token_id] = self.mask_token_id

        if context is not None:
            assert len(context) == len(batch_text), (len(context), len(batch_text))

            obj_2 = self.tok(context, padding='longest', truncation=True,
                            return_tensors='pt', max_length=self.background_maxlen).to(DEVICE)

            ids_2, mask_2 = obj_2['input_ids'][:, 1:], obj_2['attention_mask'][:, 1:]  # Skip the first [SEP]

            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)

        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        # confidence_layerの生成
        tokens = [self.tok.convert_ids_to_tokens(ids[i]) for i in range(ids.shape[0])]
        print(f"tokensの長さ: {len(tokens)} * {len(tokens[0])}")
        print(f"tokensの例: {tokens[1]}")

        # confidence_layerの初期化(初期値を-1に設定)
        confidence_layer = torch.full_like(ids, -1, dtype=torch.float)

        # GPT-4による確信度割り当て
        confidence_calc = ConfidenceCalculator()
        for i, sentence_tokens in enumerate(tqdm(tokens, desc="Processing Sentences")):
            confidences = confidence_calc.get_confidence(sentence_tokens)

            for j, conf in enumerate(confidences):
                confidence_layer[i][j] = conf


        # リストベースの確信度割り当て
        # for i, sentence_ids in enumerate(ids):
        #     for j, token in enumerate(tokens[i]):
        #         confidence_layer[i][j] = self.get_confidence(token)

        # ここまで

        if bsize:
            batches = _split_into_batches_with_confidence_layer(ids, mask, confidence_layer, bsize)
            return batches
        
        if self.used is False:
            self.used = True

            firstbg = (context is None) or context[0]
            if self.verbose > 1:
                print()
                print("#> QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
                print(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
                print(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
                print(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
                print()
        
        return ids, mask, confidence_layer

    # Ensure that query_maxlen <= length <= 500 tokens
    def max_len(self, length):
        return min(500, max(self.query_maxlen, length))
    
    def get_confidence(self, token):
        # 確信度を持つ単語のリストとその確信度のマッピング
        confidence_dict = {
            "absolutely": 1.0,
            "definitely": 1.0,
            "undoubtedly": 1.0,
            "unquestionably": 1.0,
            "indubitably": 1.0,
            "certainly": 0.95,
            "undeniably": 0.95,
            "positively": 0.95,
            "unequivocally": 0.95,
            "irrefutably": 0.95,
            "incontrovertibly": 0.95,
            "indisputably": 0.95,
            "certain": 0.9,
            "surely": 0.9,
            "assuredly": 0.9,
            "decidedly": 0.9,
            "doubtless": 0.9,
            "unmistakably": 0.9,
            "conclusively": 0.9,
            "patently": 0.9,
            "unambiguously": 0.9,
            "manifestly": 0.85,
            "evidently": 0.8,
            "clearly": 0.8,
            "obviously": 0.8,
            "plainly": 0.8,
            "apparently": 0.75,
            "presumably": 0.75,
            "seemingly": 0.75,
            "likely": 0.75,
            "probably": 0.7,
            "ostensibly": 0.7,
            "outwardly": 0.7,
            "supposedly": 0.7,
            "conceivably": 0.6,
            "possibly": 0.6,
            "allegedly": 0.6,
            "reputedly": 0.6,
            "reportedly": 0.6,
            "perhaps": 0.55,
            "feasibly": 0.55,
            "plausibly": 0.55,
            "purportedly": 0.55,
            "maybe": 0.5,
            "potentially": 0.5,
            "theoretically": 0.5,
            "hypothetically": 0.5,
            "putatively": 0.5,
            "questionably": 0.45,
            "debatably": 0.45,
            "arguably": 0.45,
            "uncertainly": 0.4,
            "tentatively": 0.4,
            "speculatively": 0.4,
            "might": 0.4,
            "controversially": 0.4,
            "disputably": 0.4,
            "perchance": 0.35,
            "contestably": 0.35,
            "contentiously": 0.35,
            "could": 0.3,
            "doubtfully": 0.25,
            "improbably": 0.25,
            "unlikely": 0.2,
            "dubiously": 0.2,
            "suspiciously": 0.2,
            "problematically": 0.2,
            "enigmatically": 0.2,
            "implausibly": 0.15,
            "paradoxically": 0.15,
        }
        return confidence_dict.get(token.lower(), -1)
