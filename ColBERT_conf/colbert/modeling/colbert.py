from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT
from colbert.parameters import DEVICE

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load

from tqdm import tqdm

class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token = self.raw_tokenizer.pad_token_id


    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze

        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        nway = self.colbert_config.nway
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)

        return torch.nn.CrossEntropyLoss()(scores, labels)

    # def query(self, input_ids, attention_mask):
    #     input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
    #     Q = self.bert(input_ids, attention_mask=attention_mask)[0]
    #     print(f"Q_shepe: {Q.shape}")
    #     Q = self.linear(Q)

    #     mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
    #     Q = Q * mask

    #     return torch.nn.functional.normalize(Q, p=2, dim=2)

    def query(self, input_ids, attention_mask):
        
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        
        # Forward pass to get the BERT outputs including attentions
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_attentions=True)
        
        # Get the attention for the specific layer (n) and head (k)
        attention_layer = outputs.attentions  # Get attention for layer n

        # Proceed with your original embedding processing
        Q = outputs[0]  # [0] gets the final hidden states
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2), attention_layer

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        if self.colbert_config.similarity == 'l2':
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """
    print(f"We're in colbert_score.")
    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, confidence_layer, D_packed, D_lengths, config=ColBERTConfig(), confidence_search=False, confidence_params=[]):
    """
        Works with a single query only.
    """
    print(f"aaa params: {confidence_params}")

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T
    scores_conf = scores

    if confidence_search == True:
        print(f"confidence_search is ON. params:{confidence_params}")
        scores_conf = apply_confidence(scores, confidence_layer, confidence_params)#####ここで重み付け関数を変更
    else:
        print(f"confidence_search is OFF.")
        scores_conf = scores

    # if torch.all(confidence_layer == -1):
    #     pass# print("===confidence_layer is all -1===")
    # else:
    #     print("===confidence_layer is valid===")
    #     if torch.equal(scores_conf, scores):
    #         print("scores_conf EQUAL scores")
    #     else:
    #         print("scores_conf NOT_EQUAL scores")
    # print(f"scores_without_conf: {scores.shape}")
    # print(f"scores_with_conf: {scores_conf.shape}")

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(scores_conf, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores_conf, D_lengths)

# 確信度が低いとき、スコアを高くする(最初の手法)
def apply_confidence(scores_, confidence_layer, confidence_params):
    # データ型とデバイスを一致
    confidence_layer = confidence_layer.to(dtype=scores_.dtype, device=scores_.device)
    print(f"apply_confidence is called.")
    assert confidence_params==[] or len(confidence_params)==3, "The list confidence_params should contain either 3 elements or be empty."
    if confidence_params==[]:
        a = 0
        b = 1
        c = 1
    else:
        a = confidence_params[0]
        b = confidence_params[1]
        c = confidence_params[2]

    # パラメータを計算
    paramators = torch.where(
        confidence_layer == -1,
        torch.tensor(1.0, device=confidence_layer.device, dtype=scores_.dtype),
        (32 ** (confidence_layer - 0.7 + a)) * b
    )  # 形状: confidence_layerと同じ形, [128]とか

    # スコアを計算(ブロードキャスト)
    base_scores = torch.sign(scores_) * (torch.abs(scores_) ** paramators) * c
    scores_conf = torch.where(
        confidence_layer == -1,
        torch.tensor(0.0, device=scores_.device, dtype=scores_.dtype),
        base_scores
    )
    
    return scores_conf

# 確信度が低いとき、高いスコアを下げる(二番目の手法)
def apply_confidence2(scores_, confidence_layer, confidence_params):
    # データ型とデバイスを一致
    confidence_layer = confidence_layer.to(dtype=scores_.dtype, device=scores_.device)
    print(f"apply_confidence2 is called.")
    assert confidence_params==[] or len(confidence_params)==3, "The list confidence_params should contain either 3 elements or be empty."

    if confidence_params==[]:
        a = 1
        b = 1
        c = 1
    else:
        a = confidence_params[0]
        b = confidence_params[1]
        c = confidence_params[2]###b,cのパラメータは今のところつかってない、入力との整合性のため残してる

    # パラメータを計算
    paramators = torch.where(
        confidence_layer == -1,
        torch.tensor(1.0, device=confidence_layer.device, dtype=scores_.dtype),#2.0は後から修正する
        confidence_layer * a
    )

    # スコアを計算
    scores_conf = torch.minimum(scores_, paramators)

    return scores_conf


# 確信度が一定以下のものはスコア0にする(=トリミングのようなもの)
def apply_confidence3(scores_, confidence_layer, confidence_params):
    # データ型とデバイスを一致
    confidence_layer = confidence_layer.to(dtype=scores_.dtype, device=scores_.device)
    print(f"apply_confidence3 is called.")
    assert confidence_params==[] or len(confidence_params)==3, "The list confidence_params should contain either 3 elements or be empty."


    # スコアを計算
    k = 0.4
    scores_conf = torch.where(confidence_layer <= k, torch.tensor(0.0, device=scores_.device, dtype=scores_.dtype), scores_)

    return scores_conf