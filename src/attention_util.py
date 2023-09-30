import copy

import torch
import torch.nn.functional as F


def src_seq_att(eatt, xatt, src_ids, dummy=False, normalize=False):
    if dummy or normalize:
        eatt_dum = dummy_attention_by_batch(eatt.unsqueeze(0))[0]
        xatt_dum = dummy_attention_by_batch(xatt.unsqueeze(0))[0]

    if dummy:
        eatt = copy.deepcopy(eatt_dum)
        xatt = copy.deepcopy(xatt_dum)

    cont_ids = context_ids(src_ids, eatt, include_last_ids=False)
    x2cont = coverage(eatt, src_ids=src_ids, tgt_ids=cont_ids)
    x2x = coverage(eatt, src_ids=src_ids, tgt_ids=src_ids)
    ent = entropy(eatt, src_ids=src_ids, tgt_ids=torch.cat((src_ids, cont_ids), 0), normalize=True)
    cont2x = coverage(eatt, src_ids=cont_ids, tgt_ids=src_ids)
    x2eos = att2eos(eatt, src_ids=src_ids)
    tgtseq2x = coverage(xatt, tgt_ids=src_ids)

    if normalize and src_ids is not None:
        x2cont /= coverage(eatt_dum, src_ids=src_ids, tgt_ids=cont_ids)
        cont2x /= coverage(eatt_dum, src_ids=cont_ids, tgt_ids=src_ids)
        x2x /=coverage(eatt_dum, src_ids=src_ids, tgt_ids=src_ids)
        ent /= entropy(eatt_dum, src_ids=src_ids, tgt_ids=torch.cat((src_ids, cont_ids), 0), normalize=True)
        x2eos /= att2eos(eatt_dum, src_ids=src_ids)
        tgtseq2x /= coverage(xatt_dum, tgt_ids=src_ids)

    att_tuple = tuple([torch.mean(att) for att in [x2cont, cont2x, ent, x2x, x2eos, tgtseq2x]])

    return att_tuple


def tgt_seq_att(xatt, datt, tgt_ids, dummy=False, normalize=False):
    if dummy or normalize:
        datt_dum = dummy_attention_by_batch(datt.unsqueeze(0))[0]
        xatt_dum = dummy_attention_by_batch(xatt.unsqueeze(0))[0]

    if dummy:
        datt = copy.deepcopy(datt_dum)
        xatt = copy.deepcopy(xatt_dum)

    src_seqids = context_ids([], xatt, include_last_ids=False)[1:] # omitting 1st token = language tag
    y2srcseq_ent = entropy(xatt, src_ids=tgt_ids, tgt_ids=src_seqids, normalize=True)
    y2eos = att2eos(xatt, src_ids=tgt_ids)

    cont_ids = context_ids(tgt_ids, datt, include_last_ids=True)
    y2cont = coverage(datt, src_ids=tgt_ids, tgt_ids=cont_ids)
    y2y = coverage(datt, src_ids=tgt_ids, tgt_ids=tgt_ids)
    tgt_seqids = torch.arange(datt.size(2))[1:]
    ent = entropy(datt, src_ids=tgt_ids, tgt_ids=tgt_seqids, normalize=True)

    if normalize and tgt_ids is not None:
        y2srcseq_ent /= entropy(xatt_dum, src_ids=tgt_ids, tgt_ids=src_seqids, normalize=True)
        y2eos /= att2eos(xatt_dum, src_ids=tgt_ids)
        y2cont /= coverage(datt_dum, src_ids=tgt_ids, tgt_ids=cont_ids)
        y2y /= coverage(datt_dum, src_ids=tgt_ids, tgt_ids=tgt_ids)
        ent /= entropy(datt_dum, src_ids=tgt_ids, tgt_ids=tgt_seqids, normalize=True)

    # aggh, aggl = agg_func(agg_heads), agg_func(agg_layers)

    att_tuple = tuple([torch.mean(att) for att in [y2srcseq_ent, y2eos, y2cont, y2y, ent]])

    return att_tuple


def dummy_attention_by_batch(attention):
    '''
    uniform attention along non-zero attention values
    '''
    dummy = torch.where(attention > 0, 1, 0) # (bsz, nlayers, nheads, # src_tokens, # tgt_tokens)
    target_sum = torch.sum(attention, dim=-1) #  (bsz, nlayers, nheads, # src_tokens)
    target_uniform = target_sum / dummy.sum(-1)
    target_uniform = target_uniform.unsqueeze(-1).expand(-1, -1,-1,-1, attention.size(-1)) # (bsz, nlayers, nheads, # src_tokens, # tgt_tokens)
    dummy_attention = dummy * target_uniform
    return dummy_attention


def att2eos(attention, src_ids=None):
    if src_ids is not None:
        attention = attention[:, :, src_ids, :]
    eos_ids = torch.where(attention > 0, 1, 0)
    eos_ids = torch.sum(eos_ids, dim=-1)[0][0] - 1
    eos_att = attention[:, :, :, eos_ids.long()]
    eos_att = torch.sum(eos_att, dim=-1).sum(dim=-1)
    return eos_att


def context_ids(token_ids, attention, include_last_ids=False):

    att = torch.where(attention > 0, 1, 0)
    eos_id = torch.sum(att, dim=-1)[0][0][0] # eos_id in tgt_seq
    if not include_last_ids:
         eos_id -= 1
    cont_ids = torch.arange(eos_id.item())
    cont_ids = torch.tensor([cid for cid in cont_ids if cid not in token_ids]).long()
    return cont_ids

def entropy(attention, src_ids=None, tgt_ids=None, normalize=True):
    if src_ids is not None:
        attention = attention[:, :, src_ids, :]
    if tgt_ids is not None:
        attention = attention[:, :, :, tgt_ids]
    idx = attention != 0
    entropy = attention.clone().detach()
    if normalize:
        entropy = F.normalize(entropy, p=1, dim=-1)
    entropy[idx] = -entropy[idx] * torch.log(entropy[idx])
    entropy = torch.sum(entropy, dim=-1).sum(dim=-1)
    return entropy


def coverage(attention, src_ids=None, tgt_ids=None):
    '''
    attention: [ nlayers, nheads, seq_len, seq_len]
    src_ids: token ids of a source segment, torch 1-d tensor, default=None
    tgt_ids: token ids of a target segment, torch 1-d tensor, default=None
    '''
    #  (nlayers, nheads, # src_tokens, # tgt tokens)
    if src_ids is not None:
        attention = attention[:, :, src_ids, :]
    if tgt_ids is not None:
        attention = attention[:, :, :, tgt_ids]
    # sum along src tokens (for attention received by tgt tokens)
    coverage = torch.sum(attention, dim=2)
    # coverage of segment, summed by target tokens
    coverage = torch.sum(coverage, dim=-1)
    return coverage # (nlayers, nheads)


def confidence(attention, src_ids=None, tgt_ids=None, mask_last_ids=False):
    if src_ids is not None:
        attention = attention[:, :, src_ids, :]

    if tgt_ids is not None:
        attention = attention[:, :, :, tgt_ids]

    if mask_last_ids:
        eos_ids = torch.where(attention > 0, 1, 0)
        eos_ids = torch.sum(eos_ids, dim=-1)[0][0] - 1
        tgt_ids = torch.stack([torch.arange(eos_id.item()) for eos_id in eos_ids.long()])
        attention = attention[:, :, :, tgt_ids]

    confidence = torch.max(attention, dim=-1).values
    # confidence of segment, summed by source tokens
    confidence = torch.sum(confidence, dim=-1)
    return confidence  # (nlayers, nheads)


def variance(attention, src_ids=None, tgt_ids=None):
    nlayers, nheads = attention.size(0), attention.size(1)
    if src_ids is not None:
        attention = attention[:, :, src_ids, :]
        position = torch.range(1, attention.size(-1)).repeat(nlayers, nheads, src_ids.size(0), 1)
    else:
        src_len = attention.size(2)
        position = torch.range(1, attention.size(-1)).repeat(nlayers, nheads, src_len, 1)

    position = position.to(attention.get_device())

    # (nlayers, nheads, # source tokens)
    expected_position = torch.sum(position * attention, dim=-1)
    # (nlayers, nheads, # source tokens, # target_tokens)
    expected_position = expected_position.unsqueeze(-1).expand(-1, -1, -1, position.size(-1))

    normalized_attention = F.normalize(attention, p=1, dim=-1)
    variance = - normalized_attention * (expected_position - position)**2

    if tgt_ids is not None:
        variance = variance[:, :, :, tgt_ids]

    # sum along source and target tokens
    variance = variance.sum(dim=-1).sum(dim=-1)

    return variance # (nlayers, nheads)





