# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from typing import Dict, List, Optional
from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.fairseq_decoder import FairseqDecoder
import torch
from fairseq.models.nat.nat import NATransformerDecoder
import logging
import random
from lunanlp import torch_seed
import numpy as np
from fairseq.models.nat import FairseqNATSharedDecoder, FairseqNATModel, ensemble_decoder
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from fairseq.modules import MultiheadAttention
logger = logging.getLogger(__name__)

def judge_finished(c_list, idx_list):
    for i in range(len(c_list)):
        if c_list[i][idx_list[i]] == 2 or c_list[i][idx_list[i]] == 1:
            return False
    return True

def judge_same(c_list, idx_list):
    for i in range(len(c_list)-1):
        if c_list[i][idx_list[i]] != c_list[i+1][idx_list[i+1]]:
            return False
    return True

def judge_small(idx_list, Len):
    for i in range(len(idx_list)):
        if idx_list[i] < Len - 1:
            return True
    return False

def add_hash(c_list, idx_list1, hash_list):
    for i in range(len(c_list)):
        if c_list[i][idx_list1[i]] not in hash_list[i].keys():
            hash_list[i][c_list[i][idx_list1[i]]] = idx_list1[i]
    return hash_list

def find_same(c_list, idx_list1, hash_list):
    for i in range(len(c_list)):
        flag = True
        for j in range(len(hash_list)):
            if i != j:
                if c_list[i][idx_list1[i]] not in hash_list[j].keys():
                    flag = False
                    break
        if flag:
            for j in range(len(hash_list)):
                if i != j:
                   idx_list1[j] = hash_list[j][c_list[i][idx_list1[i]]]
            break
    return idx_list1, flag

def delete_repeat(c_list, s_list):
    for i in range(len(c_list)):
        c_new = np.ones((c_list[i].shape[0]))
        s_new = np.ones((s_list[i].shape[0]))
        c_new[0] = c_list[i][0]
        s_new[0] = s_list[i][0]
        j, k = 1, 1
        while j < c_list[i].shape[0] and c_list[i][j] != 2:
            if c_list[i][j] == c_new[k-1]:
                s_new[k] = max(s_list[i][j], s_list[i][j-1])
                j += 1
            else:
                c_new[k] = c_list[i][j]
                s_new[k] = s_list[i][j]
                j += 1
                k += 1
        c_new[k] = 2
        c_list[i] = c_new
        s_list[i] = s_new
    return c_list, s_list

def combine(c, s):
    c_clone = c.clone()
    c = c.cpu().numpy()
    s = s.cpu().numpy()
    res = torch.ones((c.shape[1])).cpu().numpy()
    num = c.shape[0]
    Len = c.shape[1]
    c_list = [c[i] for i in range(num)]
    s_list = [s[i] for i in range(num)]
    idx_list = [1 for i in range(num)]
    l = 1
    res[0] = 0
    c_list, s_list = delete_repeat(c_list, s_list)
    segment = 0
    common_mask = torch.zeros((c.shape[1])).type_as(c_clone).bool()
    common_mask[0] = True
    common_start = []
    common_end = []
    while judge_finished(c_list, idx_list) and l < Len:
        for i in range(num):
            while c_list[i][idx_list[i]] == res[l - 1]:
                idx_list[i] += 1
        if judge_same(c_list, idx_list):
            res[l] = c_list[0][idx_list[0]]
            common_mask[l] = True
            l += 1
            idx_list = [i+1 for i in idx_list]
        else:
            segment += 1
            hash_list = [{c_list[i][idx_list[i]]: idx_list[i]} for i in range(num)]
            idx_list1 = [i+1 for i in idx_list]
            while judge_small(idx_list1, Len):
                hash_list = add_hash(c_list, idx_list1, hash_list)
                idx_list1, flag = find_same(c_list, idx_list1, hash_list)
                if flag:
                    break
                idx_list1 = [i + 1 if i < Len-1 else i for i in idx_list1]
            sum_list = [s_list[i][idx_list[i]-1:idx_list1[i]+1].mean() for i in range(num)]  #idx_list[i]-1:idx_list1[i]+1从开头的前一个到最后的后一个
            max_idx = sum_list.index(max(sum_list))
            common_start.append(l)
            res[l:l + idx_list1[max_idx] - idx_list[max_idx]] = c_list[max_idx][idx_list[max_idx]:idx_list1[max_idx]]
            common_end.append(l + idx_list1[max_idx] - idx_list[max_idx])
            l = l + idx_list1[max_idx] - idx_list[max_idx]
            idx_list = idx_list1
    if l < Len:
        common_mask[l] = True
        res[l] = 2
    res = torch.from_numpy(res).cuda().type_as(c_clone)
    #print(segment)
    return res.unsqueeze(0).repeat(num, 1), segment, common_mask, common_start, common_end

def combine_all(c_all, s_all, beam_size):
    res, segment, common_mask, common_start, common_end = combine(c_all[0:beam_size], s_all[0:beam_size])
    segmets = [segment]
    common_start_list = [common_start]
    common_end_list = [common_end]
    s = beam_size
    while s < c_all.shape[0]:
        res_tmp, segment, common_mask_tmp, common_start, common_end = combine(c_all[s:s + beam_size], s_all[s:s + beam_size])
        res = torch.cat((res, res_tmp), 0)
        common_mask = torch.cat((common_mask, common_mask_tmp), 0)
        segmets.append(segment)
        common_start_list.append(common_start)
        common_end_list.append(common_end)
        s += beam_size
    return res, segmets, common_mask, common_start_list, common_end_list

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def judge_stop(l):
    for i in range(len(l)):
        if not l[i]:
            return False
    return True

def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("glat_fliter")
class Glat_Fliter(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.glat_dict = {}
        
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--restore-decoder-from",
            default="off",
            action="store",
        )
        parser.add_argument(
            '--dropout-anneal',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dropout-anneal-end-ratio',
            type=float,
            default=0
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, train_ratio=None, **kwargs
    ):
        if train_ratio is not None:
            self.encoder.train_ratio = train_ratio
            self.decoder.train_ratio = train_ratio
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # print(encoder_out.keys())--->dict_keys(['encoder_out', 'encoder_padding_mask', 'encoder_embedding', 'encoder_states', 'src_tokens', 'src_lengths'])
        # print(type(encoder_out))
        # --> dict

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )
        tgt_ori = tgt_tokens
        prev_output_tokens_ori = prev_output_tokens.clone()
        src_tokens_ori = src_tokens.clone()
        rand_seed = random.randint(0, 19260817)
        glat_info = {}
        ori_acc = None
        ori_p = None
        if glat and tgt_tokens is not None:
            if "context_p" in glat:
                #with torch.no_grad():  
                with torch_seed(rand_seed):
                    word_ins_out_ori = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                    )
                pred_tokens = word_ins_out_ori.argmax(-1)
                nonpad_positions = ~tgt_tokens.eq(self.pad)
                same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1) #每个样本中正确的位置数量。
                seq_lens = (nonpad_positions).sum(1)  #计算了每个样本中非填充标记的总数量，即序列长度
                same_num = same_num.type(torch.float)  #转成float否则下面进行除法时会变为0
                seq_lens = seq_lens.type(torch.float)

                ori_acc = same_num / seq_lens  #计算第一轮decode结果的准确率
                output_logit = F.softmax(word_ins_out_ori, -1)  #每个位置上都转换成概率分布
                ori_p = output_logit.gather(2, tgt_tokens[:, :, None]).squeeze(-1)  #获取模型对tgt的预测概率
                ori_p[~nonpad_positions] = 0
                ori_p = ori_p.sum(1) / seq_lens

                keep_prob = ((seq_lens - same_num) / seq_lens * glat['context_p']).unsqueeze(-1)  #得到了每个标记被错误预测的概率, 
                #keep_prob = ((1 - ori_p) * glat['context_p']).unsqueeze(-1)

                # keep: True, drop: False
                keep_word_mask = (
                        torch.rand(prev_output_tokens.shape, device=word_ins_out_ori.device) < keep_prob).bool()  #random选取要mask的值

                keep_word_mask = keep_word_mask & tgt_tokens.ne(self.pad)  #只有在 keep_word_mask 为 True 且目标标记不是填充标记时，对应位置才会在输出中保留，其他位置将被丢弃

                glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask,
                                                                            0) + tgt_tokens.masked_fill(
                    ~keep_word_mask, 0)  #模型的前一个输出 prev_output_tokens 中需要保留的位置设为 0，并将对应的目标标记 tgt_tokens 复制到这些位置，以生成新的输出
                glat_tgt_tokens = tgt_tokens.masked_fill(keep_word_mask, self.pad)  #将目标标记中需要保留的位置设为填充标记 self.pad

                at_mask = keep_word_mask 

                prev_output_tokens, tgt_tokens = glat_prev_output_tokens, glat_tgt_tokens

                glat_info = {
                    "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                    "glat_context_p": glat['context_p'],
                    "glat_keep": keep_prob.mean().item()
                }

        with torch_seed(rand_seed):
            prev_output_tokens_all = prev_output_tokens
            
            '''
            if glat and tgt_tokens is not None and "context_p" in glat:
                B, L, target_L = src_tokens.shape[0], src_tokens.shape[1], prev_output_tokens.shape[1]
                src_tokens_list = src_tokens.tolist()
                for i in range(B):
                    for j in range(L-1, -1, -1):
                        if src_tokens_list[i][j] == 1: # 删除pad，以免pad不同的长度造成影响
                            src_tokens_list[i].pop(j)
                        else:
                            break
                prev_output_tokens_list = prev_output_tokens.tolist()
                best_prev_output_tokens_list = []
                flag = True # 是否都存在dict中
                for i in range(B):
                    if str(src_tokens_list[i]) in self.glat_dict:
                        best_prev_output_tokens_list.append(self.glat_dict[str(src_tokens_list[i])].copy())
                    else:
                        for j in range(target_L-1, -1, -1):
                            if prev_output_tokens_list[i][j] == 1: # 删除pad，以免pad不同的长度造成影响
                                prev_output_tokens_list[i].pop(j)
                            else:
                                break
                        self.glat_dict[str(src_tokens_list[i])] = prev_output_tokens_list[i].copy()
                        flag = False
                if flag:
                    ori_Len_list = []
                    Len_list = []
                    for i in range(B):
                        if len(best_prev_output_tokens_list[i]) > target_L:
                            print("scr:", src_tokens_list[i])
                            print(best_prev_output_tokens_list[i]) #可能会有相同的源句对应不同长度的译文
                            flag = False
                            break
                        if len(best_prev_output_tokens_list[i]) < target_L:
                            best_prev_output_tokens_list[i] += ([1] * (target_L - len(best_prev_output_tokens_list[i])))
                            
                if flag:
                    best_prev_output_tokens = torch.tensor(best_prev_output_tokens_list).type_as(prev_output_tokens) 
                    prev_output_tokens_all = torch.cat((prev_output_tokens, best_prev_output_tokens), dim=0)
                    encoder_out["encoder_out"][0] = encoder_out["encoder_out"][0].repeat(1, 2, 1)
                    encoder_out["encoder_padding_mask"][0] = encoder_out["encoder_padding_mask"][0].repeat(2, 1)
            '''
        
            word_ins_out_all = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens_all,
                encoder_out=encoder_out,
            )
            
            word_ins_out = word_ins_out_all
            '''
            if glat and tgt_tokens is not None and "context_p" in glat:
                if flag:
                    word_ins_out = word_ins_out_all[:int(word_ins_out_all.shape[0] / 2), :, :]
                    word_ins_out_best = word_ins_out_all[int(word_ins_out_all.shape[0] / 2):, :, :]
                    
                    # 计算本次glat后的loss
                    nonpad_positions = tgt_tokens.ne(self.pad)
                    seq_lens = (nonpad_positions).sum(1)
                    output_logit = F.softmax(word_ins_out, -1)
                    p = output_logit.gather(2, tgt_tokens[:, :, None]).squeeze(-1)
                    p[~nonpad_positions] = 0
                    p = p.sum(1) / seq_lens
                    # 计算原始最佳的glat的loss
                    best_tgt_tokens = tgt_ori
                    best_tgt_tokens[best_prev_output_tokens.ne(self.unk) & best_prev_output_tokens.ne(self.eos) & best_prev_output_tokens.ne(self.bos)] = self.pad
                    nonpad_positions_best = best_tgt_tokens.ne(self.pad)
                    seq_lens_best = (nonpad_positions_best).sum(1)
                    best_output_logit = F.softmax(word_ins_out_best, -1)
                    best_p = best_output_logit.gather(2, best_tgt_tokens[:, :, None]).squeeze(-1)
                    best_p[~nonpad_positions] = 0
                    best_p = best_p.sum(1) / seq_lens_best
                                        
                    # loss最小的glat结果
                    mask = ((p - best_p) > -0.03) # 当前的概率与最大值相差不大，则替换(增强随机性)
                    glat_info["glat_replace"] = mask.float().mean().item()
                    tgt_tokens[~mask] = best_tgt_tokens[~mask]
                    word_ins_out[~mask] = word_ins_out_best[~mask]
                    p[~mask] = best_p[~mask]
                    
                    # 如果不提醒也能取得足够好的p，则使用不提醒的p
                    ori_p = ori_p.type_as(p)
                    ori_mask = (ori_p >= p)
                    glat_info["ori_keep"] = ori_mask.float().mean().item()
                    tgt_tokens[ori_mask] = tgt_ori[ori_mask]
                    word_ins_out[ori_mask] = word_ins_out_ori[ori_mask]
                    p[ori_mask] = ori_p[ori_mask]
                    
                    # 过滤loss过大的句子
                    # fliter_mask = p < 0.6
                    # tgt_tokens[fliter_mask] = self.pad
                    # glat_info["fliter_mask"] = fliter_mask.float().mean().item()
                    
                    # 更新dict
                    src_tokens = src_tokens_ori[mask]
                    prev_output_tokens = prev_output_tokens[mask]
                    B, L, target_L = src_tokens.shape[0], src_tokens.shape[1], prev_output_tokens.shape[1]
                    src_tokens_list = src_tokens.tolist()
                    for i in range(B):
                        for j in range(L-1, -1, -1):
                            if src_tokens_list[i][j] == 1: # 删除pad，以免pad不同的长度造成影响
                                src_tokens_list[i].pop(j)
                            else:
                                break
                    prev_output_tokens_list = prev_output_tokens.tolist()
                    for i in range(B):
                        for j in range(target_L-1, -1, -1):
                            if prev_output_tokens_list[i][j] == 1: # 删除pad，以免pad不同的长度造成影响
                                prev_output_tokens_list[i].pop(j)
                            else:
                                break
                        if len(prev_output_tokens_list[i]) != len(self.glat_dict[str(src_tokens_list[i])]):
                            print("scr:", src_tokens_list[i])
                            print("prev1:", prev_output_tokens_list[i])
                            print("prev2:", self.glat_dict[str(src_tokens_list[i])])
                        else:
                            self.glat_dict[str(src_tokens_list[i])] = prev_output_tokens_list[i].copy()
           
                    #不提示不需要进行存储，因为每一次都会进行比较
                    src_tokens = src_tokens_ori[ori_mask]
                    prev_output_tokens = prev_output_tokens_ori[ori_mask]
                    B, L, target_L = src_tokens.shape[0], src_tokens.shape[1], prev_output_tokens.shape[1]
                    src_tokens_list = src_tokens.tolist()
                    for i in range(B):
                        for j in range(L-1, -1, -1):
                            if src_tokens_list[i][j] == 1: # 删除pad，以免pad不同的长度造成影响
                                src_tokens_list[i].pop(j)
                            else:
                                break
                    prev_output_tokens_list = prev_output_tokens.tolist()
                    for i in range(B):
                        for j in range(target_L-1, -1, -1):
                            if prev_output_tokens_list[i][j] == 1: # 删除pad，以免pad不同的长度造成影响
                                prev_output_tokens_list[i].pop(j)
                            else:
                                break
                        self.glat_dict[str(src_tokens_list[i])] = prev_output_tokens_list[i].copy()
            '''       
            
            sub_p_mask = None
            nonpad_positions = tgt_tokens.ne(self.pad)
            seq_lens = (nonpad_positions).sum(1)

            if ori_p is not None:
                output_logit = F.softmax(word_ins_out, -1)
                p = output_logit.gather(2, tgt_tokens[:, :, None]).squeeze(-1)  #从 output_logit 中提取 tgt 对应的概率值 p, 得到形状与 tgt_tokens 相同的概率张量。
                p[~nonpad_positions] = 0  #将填充标记对应的概率置零，以便后续计算。
                p = p.sum(1) / seq_lens
                #ori_mask  = (p <= ori_p) & (ori_p > ori_p.mean())
                #mask_window = (p < p.mean()) & ((p-ori_p) < (p-ori_p).mean())  #过滤数据
                
                ori_mask  = (p <= ori_p) & (ori_p > 0.7)
                mask_window = (p < 0.7) & ((p-ori_p) < 0.20)  #过滤数据
                
                word_ins_out[ori_mask] = word_ins_out_ori[ori_mask]
                tgt_tokens[mask_window] = self.pad
                tgt_tokens[ori_mask] = tgt_ori[ori_mask]
                glat_info["ori_keep"] = ori_mask.float().mean().item()
                glat_info["fliter_mask"] = mask_window.float().mean().item()
    
            '''
            if ori_acc is not None:
                pred_tokens = word_ins_out.argmax(-1)
                nonpad_positions = ~tgt_tokens.eq(self.pad)
                same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)
                seq_lens = (nonpad_positions).sum(1)
                same_num = same_num.type(torch.float)  #转成float否则下面进行除法时会变为0
                seq_lens = seq_lens.type(torch.float)
                now_acc = same_num / seq_lens
                ori_mask  = (now_acc < ori_acc) & (ori_acc > ori_acc.mean())
                mask_window = (now_acc < now_acc.mean()) & ((now_acc-ori_acc) < (now_acc-ori_acc).mean())  #过滤数据
                #print(ori_mask.float().mean(), mask_window.float().mean())
                word_ins_out[ori_mask] = word_ins_out_ori[ori_mask]
                tgt_tokens[mask_window] = self.pad
                tgt_tokens[ori_mask] = tgt_ori[ori_mask]
                glat_info["ori_keep"] = ori_mask.float().mean().item()
                glat_info["fliter_mask"] = mask_window.float().mean().item()
            '''

        # pred_tokens = tgt_ori
        # tgt_AT = torch.ones_like(tgt_ori).type_as(tgt_ori)
        # tgt_AT[:, :-1] = tgt_ori[:, 1:]
        # word_ins_out_rescore = self.decoder.forward_rescore(pred_tokens, encoder_out, normalize=False)


        if train_ratio is None:
            train_ratio = 0


        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens, 
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 1,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }
        # print("the ret type",type(ret))
        # --> dict

        if glat_info is not None:
            ret.update(glat_info)
        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, reranker=None, encoder_input=None, batch=16, tgt_tokens=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        num = int(output_tokens.shape[0] / batch)
        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        
        word_ins_out = self.decoder(
                            normalize=True,
                            prev_output_tokens=output_tokens,
                            encoder_out=encoder_out,
                            step=step,
                        )
        _scores, _tokens = word_ins_out.max(-1)


        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        
        if history is not None:
            history.append(output_tokens.clone())



        # def at_ir(self, res, encoder_out, num):
        #     # at_mask = (common_mask == False) & res.ne(self.pad)  #目前还需要使用AT部分重新生成的部分
        #     # common_start_list = common_start_list
        #     # common_end_list = common_end_list
        #     encoder_out["encoder_out"][0] = encoder_out["encoder_out"][0][:, 0::num, :]
        #     encoder_out["encoder_padding_mask"][0] = encoder_out["encoder_padding_mask"][0][0::num, :]
        #     left = [1 for i in range(res.shape[0])] #指向新的
        #     flags = [False for i in range(res.shape[0])] 
        #     res_clone = res.clone()
        #     left_clone = [1 for i in range(res.shape[0])]   #指向旧的
        #     while not judge_stop(flags):
        #         output_logits_rescore = self.decoder.forward_rescore(normalize=True, prev_output_tokens=res,
        #                                                         encoder_out=encoder_out, )
        #         scores, tokens = output_logits_rescore.max(-1)
        #         scores = scores.exp()
        #         reranking_scores = output_logits_rescore.gather(2, res[:, 1:, None]).squeeze(-1).exp()
        #         for i in range(tokens.shape[0]):
        #             while left[i] < tokens.shape[1] - 1 and res[i, left[i]]!=2: 
        #                 #if tokens[i, left[i]] != res[i, left[i]+1]:  #找到第一个不同的位置
        #                 if scores[i, left[i]] > reranking_scores[i, left[i]] + 0.5:  #AT的置信度比NAT高时替换，不需要是AT预测的概率分布中最大的
        #                     #print(scores[i, left[i]], reranking_scores[i, left[i]], )
        #                     if tokens[i, left[i]] != 2:  #不是终止符
        #                         res[i, left[i]+1] = tokens[i, left[i]]
        #                         for j in range(1, 3):  
        #                             if left[i]+1+j >= res_clone.shape[1]:
        #                                 break
        #                             if res[i, left[i]+1] == res_clone[i, left[i]+1+j]: #如果AT预测出的结果在后续的NAT中出现，则将NAT后续的句子拼到现在的结果后面
        #                                 res[i, left[i]+1:-j] = res_clone[i, left[i]+1+j:]
        #                             if res[i, left[i]+1] == res_clone[i, left[i]+1-j]: #如果AT预测出的结果在之前的NAT中出现，则将NAT后续的句子拼到现在的结果后面
        #                                 res[i, left[i]+1:] = res_clone[i, left[i]+1-j:-j]
        #                     else:
        #                         res[i, left[i]+1] = tokens[i, left[i]]
        #                         res[i, left[i]+1:] = 1
        #                         flags[i] = True   #迭代完成
        #                     break
        #                 left[i] += 1
        #             if left[i] == tokens.shape[1] - 1 or res[i, left[i]]==2: #过长或者已经到终止符
        #                 flags[i] = True    
        #     return res.repeat(num, 1)

        # if num > 1:
            # res, segments, common_mask, common_start_list, common_end_list = combine_all(output_tokens, output_scores, num)
            # output_tokens = res
        #output_tokens = at_ir(self, output_tokens[0::num], encoder_out=encoder_out, num=num)

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

        
    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length + 8)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length + 8
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        # print(DecoderOut(
        #     output_tokens=initial_output_tokens,
        #     output_scores=initial_output_scores,
        #     attn=None,
        #     step=0,
        #     max_step=0,
        #     history=None,
        # ).keys())--->>dict_keys(['encoder_out', 'encoder_padding_mask', 'encoder_embedding', 'encoder_states', 'src_tokens', 'src_lengths'])
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )
    
    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length + 8)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length + 8
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


class NATransformerDecoder(FairseqNATSharedDecoder):
        def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
            super().__init__(
                args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
            )
            self.dictionary = dictionary
            self.bos = dictionary.bos()
            self.unk = dictionary.unk()
            self.eos = dictionary.eos()

            self.encoder_embed_dim = args.encoder_embed_dim
            self.sg_length_pred = getattr(args, "sg_length_pred", False)
            self.pred_length_offset = getattr(args, "pred_length_offset", False)
            self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
            self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
            self.embed_length = Embedding(256, self.encoder_embed_dim, None)
            #self.resocre_layers = torch.nn.ModuleList([self.build_decoder_layer(args) for i in range(1)])
            #self.at_output_layer = torch.nn.Linear(self.encoder_embed_dim, self.dictionary.__len__(), bias=False)
            
        def build_decoder_layer(self, args, no_encoder_attn=False):
            layer = TransformerDecoderLayer(args, no_encoder_attn)
            #layer = TransformerDecoderQMaskLayer(args, no_encoder_attn)  #使用DisCo,以免信息泄露
            if getattr(args, "checkpoint_activations", False):
                layer = checkpoint_wrapper(layer)
            return layer

        def buffered_future_mask(self, tensor):
            dim = tensor.size(0)
            # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
            if (
                    self._future_mask.size(0) == 0
                    or (not self._future_mask.device == tensor.device)
                    or self._future_mask.size(0) < dim
            ):
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
                )
            self._future_mask = self._future_mask.to(tensor)
            return self._future_mask[:dim, :dim]

        @ensemble_decoder
        def forward(self, normalize, encoder_out, prev_output_tokens, step=0, at_mask=None, **unused):
            features, _ = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=(step == 0) & self.src_embedding_copy,
                at_mask=at_mask,
            )
            decoder_out = self.output_layer(features)
            # print(decoder_out)
            return F.log_softmax(decoder_out, -1) if normalize else decoder_out

        def forward_by_layer(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
            _, all_features = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=(step == 0) & self.src_embedding_copy,
            )
            all_layer_output_logits = all_features['inner_states'][1:][self.inference_decoder_layer]
            return F.log_softmax(self.output_layer(all_layer_output_logits.transpose(0, 1)), -1) \
                if normalize else self.output_layer(all_layer_output_logits.transpose(0, 1))

        @ensemble_decoder
        def forward_length(self, normalize, encoder_out):
            enc_feats = encoder_out["encoder_out"][0]  # T x B x C
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
            else:
                src_masks = None
            enc_feats = _mean_pooling(enc_feats, src_masks)
            if self.sg_length_pred:
                enc_feats = enc_feats.detach()
            length_out = F.linear(enc_feats, self.embed_length.weight)
            return F.log_softmax(length_out, -1) if normalize else length_out

        def forward_rescore(self, prev_output_tokens, encoder_out=None, embedding_copy=False, normalize=False):
            # embedding
            if embedding_copy:
                src_embd = encoder_out["encoder_embedding"][0]
                if len(encoder_out["encoder_padding_mask"]) > 0:
                    src_mask = encoder_out["encoder_padding_mask"][0]
                else:
                    src_mask = None
                src_mask = (
                    ~src_mask
                    if src_mask is not None
                    else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
                )

                if not self.softcopy:
                    x, decoder_padding_mask = self.forward_embedding(
                        prev_output_tokens,
                        self.forward_copying_source(
                            src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                        ),
                    )
                else:
                    x = self.forward_softcopying_source(src_embd, src_mask, prev_output_tokens.ne(self.padding_idx))
                    decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)

            else:
                x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

            x = x.transpose(0, 1)
            self_attn_mask = self.buffered_future_mask(x)
            # print(self_attn_mask)
            for layer in self.resocre_layers:
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"][0]
                    if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                    else None,
                    encoder_out["encoder_padding_mask"][0]
                    if (
                            encoder_out is not None
                            and len(encoder_out["encoder_padding_mask"]) > 0
                    )
                    else None,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=decoder_padding_mask,
                )
                
            if self.layer_norm:
                x = self.layer_norm(x)

            x = self.output_layer(x)

            return F.log_softmax(x.transpose(0, 1), -1) if normalize else x.transpose(0, 1)


        def forward_at_layers(self, prev_output_tokens, encoder_out=None, encoder_padding_mask=None, normalize=False):
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
            x = x.transpose(0, 1)
            self_attn_mask = self.buffered_future_mask(x)
            for layer in self.resocre_layers:
                x, attn, _ = layer(
                    x,
                    encoder_out,
                    encoder_padding_mask,
                    self_attn_mask=None,
                    self_attn_padding_mask=decoder_padding_mask,
                )
                
            if self.layer_norm:
                x = self.layer_norm(x)

            x = self.at_output_layer(x)

            return F.log_softmax(x.transpose(0, 1), -1) if normalize else x.transpose(0, 1)


        def encode_forward(self, prev_output_tokens):
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
            x = x.transpose(0, 1)
            for i, layer in enumerate(self.layers):       
                x, attn, _ = layer(
                    x,
                    self_attn_mask=None,
                    self_attn_padding_mask=decoder_padding_mask,
                )

            return x, decoder_padding_mask


        def extract_features(
                self,
                prev_output_tokens,
                encoder_out=None,
                early_exit=None,
                embedding_copy=False,
                at_mask=None,
                **unused
        ):
            """
            Similar to *forward* but only return features.

            Inputs:
                prev_output_tokens: Tensor(B, T)
                encoder_out: a dictionary of hidden states and masks

            Returns:
                tuple:
                    - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                    - a dictionary with any model-specific outputs
                the LevenshteinTransformer decoder has full-attention to all generated tokens
            """
            # embedding
            if embedding_copy:
                src_embd = encoder_out["encoder_embedding"][0]
                if len(encoder_out["encoder_padding_mask"]) > 0:
                    src_mask = encoder_out["encoder_padding_mask"][0]
                else:
                    src_mask = None
                src_mask = (
                    ~src_mask
                    if src_mask is not None
                    else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
                )

                x, decoder_padding_mask = self.forward_embedding(
                    prev_output_tokens,
                    self.forward_copying_source(
                        src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                    ),
                )

            else:

                x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            x_kv = x.clone()

            attn = None
            inner_states = [x]
            '''
            self_attn_mask = self.buffered_future_mask(x).unsqueeze(0).repeat(prev_output_tokens.shape[0], 1, 1) #因为每一句话的mask不同
            zeros = torch.zeros_like(self_attn_mask).type_as(self_attn_mask)
            if at_mask is None:
                self_attn_mask = zeros
            else:
                zeros[at_mask] = self_attn_mask[at_mask] 
                self_attn_mask = zeros

            zeros = zeros.repeat(8, 1, 1)
            for i in range(self_attn_mask.shape[0]):
                zeros[i*8:i*8+8, :, :] = self_attn_mask[i, :, :]
            self_attn_mask = zeros
            #self_attn_mask = self_attn_mask.unsqueeze(0).repeat(8, 1, 1) # 多头注意力机制，需要乘以 head_num
            #print(self_attn_mask)
            '''
            # decoder layers
            for i, layer in enumerate(self.layers):

                # early exit from the decoder.
                if (early_exit is not None) and (i >= early_exit):
                    break
                
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"][0]
                    if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                    else None,
                    encoder_out["encoder_padding_mask"][0]
                    if (
                            encoder_out is not None
                            and len(encoder_out["encoder_padding_mask"]) > 0
                    )
                    else None,
                    self_attn_mask=None,
                    self_attn_padding_mask=decoder_padding_mask,
                )
                '''
                x, attn, _ = layer(
                    x,
                    x_kv,
                    encoder_out["encoder_out"][0]
                    if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                    else None,
                    encoder_out["encoder_padding_mask"][0]
                    if (
                            encoder_out is not None
                            and len(encoder_out["encoder_padding_mask"]) > 0
                    )
                    else None,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=decoder_padding_mask,
                )
                '''
                inner_states.append(x)

            if self.layer_norm:
                x = self.layer_norm(x)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            if self.project_out_dim is not None:
                x = self.project_out_dim(x)

            return x, {"attn": attn, "inner_states": inner_states}

        def forward_embedding(self, prev_output_tokens, states=None):
            # embed positions
            positions = (
                self.embed_positions(prev_output_tokens)
                if self.embed_positions is not None
                else None
            )

            # embed tokens and positions
            if states is None:
                x = self.embed_scale * self.embed_tokens(prev_output_tokens)
                if self.project_in_dim is not None:
                    x = self.project_in_dim(x)
            else:
                x = states

            if positions is not None:
                x += positions
            if self.dropout_anneal:
                x = self.dropout_module(x, self.train_ratio)
            else:
                x = self.dropout_module(x)
            decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
            return x, decoder_padding_mask

        def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
            length_sources = src_masks.sum(1)
            length_targets = tgt_masks.sum(1)
            mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
                ~tgt_masks, 0
            )
            copied_embedding = torch.gather(
                src_embeds,
                1,
                mapped_inputs.unsqueeze(-1).expand(
                    *mapped_inputs.size(), src_embeds.size(-1)
                ),
            )
            return copied_embedding

        def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
            enc_feats = encoder_out["encoder_out"][0]  # T x B x C
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
            else:
                src_masks = None
            if self.pred_length_offset:
                if src_masks is None:
                    src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                        enc_feats.size(0)
                    )
                else:
                    src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
                src_lengs = src_lengs.long()

            if tgt_tokens is not None:
                # obtain the length target
                tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
                if self.pred_length_offset:
                    length_tgt = tgt_lengs - src_lengs + 128
                else:
                    length_tgt = tgt_lengs
                length_tgt = length_tgt.clamp(min=0, max=255)

            else:
                # predict the length target (greedy for now)
                # TODO: implementing length-beam
                pred_lengs = length_out.max(-1)[1]
                if self.pred_length_offset:
                    length_tgt = pred_lengs - 128 + src_lengs
                else:
                    length_tgt = pred_lengs

            return length_tgt


class TransformerDecoderQMaskLayer(TransformerDecoderLayer):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """
    def __init__(self, args, no_encoder_attn=False):
        super().__init__(args, no_encoder_attn)

    def forward(
        self,
        x,
        x_kv,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        need_self_attn: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        
        #print(self_attn_mask)
        x, self_attn = self.self_attn(
            query=x,
            key=x_kv,
            value=x_kv,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_self_attn,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        if need_self_attn:
            attn = self_attn

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@register_model_architecture(
    "glat_fliter", "glat_fliter"
    # , "glat_at" #跑en-de的时候改一下
)
def base_architecture0(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "glat_fliter", "glat_fliter_base"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "glat_fliter", "glat_fliter_big"
)
def big_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture(
    "glat_fliter", "glat_fliter_16e6d"
)
def glat_16e6d_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)

