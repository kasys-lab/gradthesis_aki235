import torch
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster


from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.modeling.colbert import ColBERT


def pool_embeddings_hierarchical(
    p_embeddings,
    token_lengths,
    pool_factor,
    protected_tokens: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_embeddings = p_embeddings.to(device)
    pooled_embeddings = []
    pooled_token_lengths = []
    start_idx = 0

    for token_length in tqdm(token_lengths, desc="Pooling tokens"):
        # Get the embeddings for the current passage
        passage_embeddings = p_embeddings[start_idx : start_idx + token_length]

        # Remove the tokens at protected_tokens indices
        protected_embeddings = passage_embeddings[:protected_tokens]
        passage_embeddings = passage_embeddings[protected_tokens:]

        # Cosine similarity computation (vector are already normalized)
        similarities = torch.mm(passage_embeddings, passage_embeddings.t())

        # Convert similarities to a distance for better ward compatibility
        similarities = 1 - similarities.cpu().numpy()

        # Create hierarchical clusters using ward's method
        Z = linkage(similarities, metric="euclidean", method="ward")
        # Determine the number of clusters we want in the end based on the pool factor
        max_clusters = (
            token_length // pool_factor if token_length // pool_factor > 0 else 1
        )
        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        # Pool embeddings within each cluster
        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(
                torch.tensor(cluster_labels == cluster_id, device=device)
            )[0]
            if cluster_indices.numel() > 0:
                pooled_embedding = passage_embeddings[cluster_indices].mean(dim=0)
                pooled_embeddings.append(pooled_embedding)

        # Re-add the protected tokens to pooled_embeddings
        pooled_embeddings.extend(protected_embeddings)

        # Store the length of the pooled tokens (number of total tokens - number of tokens from previous passages)
        pooled_token_lengths.append(len(pooled_embeddings) - sum(pooled_token_lengths))
        start_idx += token_length

    pooled_embeddings = torch.stack(pooled_embeddings)
    return pooled_embeddings, pooled_token_lengths


class Checkpoint(ColBERT):
    """
    Easy inference with ColBERT.

    TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name, colbert_config=None, verbose: int = 3):
        super().__init__(name, colbert_config)
        assert self.training is False

        self.verbose = verbose

        self.query_tokenizer = QueryTokenizer(self.colbert_config, verbose=self.verbose)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        self.amp_manager = MixedPrecisionManager(True)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q, head_attention = super().query(*args, **kw_args)
                
                if to_cpu:
                    return Q.cpu(), head_attention
                else:
                    return Q, head_attention
                
    # def query_2(self, *args, to_cpu=False, n, k, **kw_args):
    #     with torch.no_grad():
    #         with self.amp_manager.context():
    #             # 継承元の query 関数を呼び出して、attention_layer を取得
    #             Q, attention_layer = super().query(*args, **kw_args)
                
    #             # n層kヘッドのアテンションを取得
    #             head_attention = attention_layer[n][:, k, :, :]

    #             # Qとhead_attentionを返す
    #             if to_cpu:
    #                 return Q.cpu(), head_attention
    #             else:
    #                 return Q, head_attention

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)

                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    # GPT-4による確信度割り当て(係り受けも考慮済みなので、ここで計算する必要はない)    
    def queryFromText(
        self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False
    ):
        if bsize:
            batches = self.query_tokenizer.tensorize(
                queries,
                context=context,
                bsize=bsize,
                full_length_search=full_length_search,
            )

            all_embeddings = []
            all_confidence_layers = []

            for input_ids, attention_mask, confidence_layer in batches:
                embeddings, attention_layer = self.query(input_ids, attention_mask)
                
                all_embeddings.append(embeddings)
                all_confidence_layers.append(confidence_layer)

            return torch.cat(all_embeddings), torch.cat(all_confidence_layers)

        input_ids, attention_mask, confidence_layer = self.query_tokenizer.tensorize(
            queries, context=context, full_length_search=full_length_search
        )

        embeddings, attention_layer = self.query(input_ids, attention_mask)

        return embeddings, confidence_layer

    # 係り受け移動を行う場合(リストベースの確信度割り当て)
    # def queryFromText(
    #     self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False
    # ):
    #     # if bsize:
    #     #     batches = self.query_tokenizer.tensorize(
    #     #         queries,
    #     #         context=context,
    #     #         bsize=bsize,
    #     #         full_length_search=full_length_search,
    #     #     )
    #     #     batches = [
    #     #         self.query(input_ids, attention_mask, to_cpu=to_cpu)
    #     #         for input_ids, attention_mask in batches
    #     #     ]
    #     #     return torch.cat(batches)

    #     #
    #     n = 7 # layer
    #     k = 0 # head

    #     skip_tokens = {"[MASK]", "[SEP]", "a", "the", ".", ",", "i", "'", "m", "?"}
    #     for i in range(100):
    #         skip_tokens.add(f'[unused{i}]')

    #     if bsize:
    #         batches = self.query_tokenizer.tensorize(
    #             queries,
    #             context=context,
    #             bsize=bsize,
    #             full_length_search=full_length_search,
    #         )

    #         all_embeddings = []
    #         all_confidence_layers = []

    #         for input_ids, attention_mask, confidence_layer in batches:
    #             embeddings, attention_layer = self.query(input_ids, attention_mask)
                
    #             confidence_layer_propagated = self.propagate_confidence(confidence_layer, attention_layer, input_ids, skip_tokens, n, k)
                
    #             all_embeddings.append(embeddings)
    #             all_confidence_layers.append(confidence_layer_propagated)

    #         return torch.cat(all_embeddings), torch.cat(all_confidence_layers)

    #     input_ids, attention_mask, confidence_layer = self.query_tokenizer.tensorize(
    #         queries, context=context, full_length_search=full_length_search
    #     )

    #     embeddings, attention_layer = self.query(input_ids, attention_mask)

    #     confidence_layer_propagated = self.propagate_confidence(confidence_layer, attention_layer, input_ids, skip_tokens)

    #     return embeddings, confidence_layer_propagated

    # confidence_layerの係り受けを利用した移動
    def propagate_confidence(
    self, confidence_layer, attention_layer, input_ids, skip_tokens, n=7, k=0
):
        _confidence_layer = torch.full_like(confidence_layer, -1, dtype=torch.float)
        propagation_count = torch.zeros_like(confidence_layer, dtype=torch.int)

        def get_propagation_weight(attention_):
            return 1.0

        # 各クエリについてループ
        for i in range(confidence_layer.shape[0]):
            for j in range(confidence_layer.shape[1]):  
                if confidence_layer[i, j] != -1:
                    attention_scores = attention_layer[n][i, k, :, j]
                    token_attention_list = []

                    for t in range(confidence_layer.shape[1]):
                        target_token = self.query_tokenizer.tok.convert_ids_to_tokens(int(input_ids[i, t]))

                        if target_token in skip_tokens:
                            continue
                        if j == t:
                            continue

                        token_attention_list.append({
                            'target_token': target_token, # 修飾する先(英単語)
                            'attention_score': attention_scores[t].item(),
                            'index': t # 修飾する先(idx)
                        })

                    sorted_attention = sorted(token_attention_list, key=lambda x: x['attention_score'], reverse=True)

                    # print(f"Top 7 Attention Scores for {self.query_tokenizer.tok.convert_ids_to_tokens(int(input_ids[i, j]))}:")
                    # for rank, entry in enumerate(sorted_attention[:7], 1):
                    #     print(f"{rank}. Token: {entry['target_token']} | Score: {entry['attention_score']:.4f}")

                    for entry in sorted_attention[:7]:
                        target_idx = entry['index']
                        propagation_weight = get_propagation_weight(entry['attention_score'])

                        propagation_count[i, target_idx] += 1
                        if _confidence_layer[i, target_idx] == -1:
                            _confidence_layer[i, target_idx] = confidence_layer[i, j] * propagation_weight
                        else:
                            total_count = propagation_count[i, target_idx].item()
                            _confidence_layer[i, target_idx] = (_confidence_layer[i, target_idx] * (total_count - 1) + confidence_layer[i, j] * propagation_weight) / total_count

        return _confidence_layer


    # def find_best_n_k_for_probably(self, input_ids, attention_mask, query_token="certain", n_max=12, k_max=12):
    #     # 修飾対象とみなすトークン（動詞部分を修飾すると想定）
    #     # target_tokens = {"goes", "on", "a", "road", "trip", "to", "find", "a", "missing", "friend"}
    #     target_tokens = {"this", "approach", "will", "work", "well"}
    #     best_combinations = []

    #     # 各レイヤー(n)とヘッド(k)をループ
    #     for n in range(n_max):
    #         for k in range(k_max):
    #             # 特定のレイヤーとヘッドのアテンションを取得
    #             embeddings, head_attention = self.query_2(input_ids, attention_mask, n=n, k=k)
                
    #             # トークンを取得
    #             tokens = self.query_tokenizer.tok.convert_ids_to_tokens(input_ids[0])
                
    #             # probablyのインデックスを取得
    #             if query_token in tokens:
    #                 probably_idx = tokens.index(query_token)
                    
    #                 # probablyが他のトークンに与えるアテンションスコアを取得
    #                 attention_scores = head_attention[0, :, probably_idx].tolist()
                    
    #                 # 各トークンのスコアをリストに保存
    #                 token_attention_list = [
    #                     {"token": tokens[i], "attention_score": attention_scores[i]} 
    #                     for i in range(len(tokens)) if tokens[i] != self.query_tokenizer.mask_token
    #                 ]
                    
    #                 # アテンションスコアを高い順にソート
    #                 sorted_attention = sorted(token_attention_list, key=lambda x: x['attention_score'], reverse=True)
                    
    #                 # top_tokensを取得（上位10件）
    #                 top_tokens = [entry['token'] for entry in sorted_attention[:5]]
                    
    #                 # target_tokensと一致するトークンの数をカウント
    #                 match_count = sum(1 for token in top_tokens if token in target_tokens)
                    
    #                 # best_combinationsに結果を追加
    #                 best_combinations.append({
    #                     "n": n,
    #                     "k": k,
    #                     "top_tokens": top_tokens,
    #                     "match_count": match_count
    #                 })

    #     # target_tokensの一致数でソート
    #     best_combinations = sorted(best_combinations, key=lambda x: x['match_count'], reverse=True)
        
    #     # 上位結果を表示
    #     for result in best_combinations[:5]:  # 上位5件を表示
    #         print(f"Layer {result['n']}, Head {result['k']}:")
    #         print(f"Top Tokens: {', '.join(result['top_tokens'])}")
    #         print(f"Match Count: {result['match_count']}\n")

    # def display_top_attention_tokens(self, input_ids, attention_mask, n, k, word):
    #     # input_ids, attention_maskをデバイスに移動
    #     input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

    #     # BERTからアテンション出力を取得
    #     outputs = self.bert(input_ids, attention_mask=attention_mask, output_attentions=True)

    #     # 指定したn層kヘッドのアテンションを取得
    #     attentions = outputs.attentions[n]  # n層目のアテンション
    #     head_attention = attentions[:, k, :, :]  # kヘッド目のアテンション

    #     # 指定された単語(word)のトークンIDを取得
    #     word_id = self.query_tokenizer.tok.convert_tokens_to_ids(word)

    #     # wordに対応するinput_idsの位置を探す
    #     word_positions = (input_ids == word_id).nonzero(as_tuple=True)

    #     if len(word_positions[0]) == 0:
    #         print(f"'{word}' not found in the input.")
    #         return

    #     # wordのアテンションスコアを上位10トークンに対応させる
    #     for i in range(len(word_positions[0])):  # 同じクエリに複数の単語がある場合に対応
    #         query_index = word_positions[0][i].item()
    #         token_index = word_positions[1][i].item()

    #         # wordのアテンションスコアを取得
    #         attention_scores = head_attention[query_index, :, token_index]  # トークンのスコア

    #         # トークンに対応するアテンションスコアとその単語のリストを作成
    #         token_attention_list = []
    #         for j in range(input_ids.shape[1]):  # 各トークンに対するスコア
    #             target_token = self.query_tokenizer.tok.convert_ids_to_tokens(int(input_ids[query_index, j]))

    #             # MASKトークンを除外
    #             if target_token == self.query_tokenizer.mask_token:
    #                 continue  # MASKトークンを無視

    #             token_attention_list.append({
    #                 'token': target_token,
    #                 'score': attention_scores[j].item()
    #             })

    #         # アテンションスコアの高い順にソート
    #         sorted_attention = sorted(token_attention_list, key=lambda x: x['score'], reverse=True)

    #         # 上位10件を表示
    #         print(f"Top 10 attention scores for '{word}' (Query {query_index}, Token Position {token_index}):")
    #         for rank, entry in enumerate(sorted_attention[:10], 1):
    #             print(f"{rank}. {entry['token']} | Score: {entry['score']:.4f}")
    #         print("\n" + "-" * 50 + "\n")

    #     return outputs
    
    def docFromText(
        self,
        docs,
        bsize=None,
        keep_dims=True,
        to_cpu=False,
        showprogress=False,
        return_tokens=False,
        pool_factor=1,
        protected_tokens=0,
        clustering_mode: str = "hierarchical",
    ):
        assert keep_dims in [True, False, "flatten"]
        assert clustering_mode in ["hierarchical"]

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(
                docs, bsize=bsize
            )

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = "return_mask" if keep_dims == "flatten" else keep_dims
            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                for input_ids, attention_mask in tqdm(
                    text_batches, disable=not showprogress
                )
            ]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == "flatten":
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = (
                    torch.cat(D)[reverse_indices],
                    torch.cat(mask)[reverse_indices],
                )

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)
                D = D[mask.bool().flatten()].cpu()

                if pool_factor > 1:
                    print(f"Clustering tokens with a pool factor of {pool_factor}")
                    D, doclens = pool_embeddings_hierarchical(
                        D,
                        doclens,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                    )

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)

        assert False, "Implement scoring"

    def score(self, Q, D, mask=None, lengths=None):
        assert False, "Call colbert_score"
        # EVENTUALLY: Just call the colbert_score function!

        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.device).unsqueeze(-1)

        scores = D @ Q
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(
        bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
    )

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, : x.size(1)] = x
        offset = endpos

    return output


"""
TODO:

def tokenize_and_encode(checkpoint, passages):
    embeddings, token_ids = checkpoint.docFromText(passages, bsize=128, keep_dims=False, showprogress=True, return_tokens=True)
    tokens = [checkpoint.doc_tokenizer.tok.convert_ids_to_tokens(ids.tolist()) for ids in token_ids]
    tokens = [tokens[:tokens.index('[PAD]') if '[PAD]' in tokens else -1] for tokens in tokens]
    tokens = [[tok for tok in tokens if tok not in checkpoint.skiplist] for tokens in tokens]

    return embeddings, tokens

"""
