import copy
import torch
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, Dataset
import pandas as pd
import numpy as np
import csv
import os
from pprint import pprint
from transformers.tokenization_utils_base import BatchEncoding
from data_correction import correct_table

dcopy = copy.deepcopy

lang_code = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
             "ms": "zsm_Latn", "ja": "jpn_Jpan", "nl": "nld_Latn", "es": "spa_Latn", "ar": "arb_Arab",
             "da": "dan_Latn", "hi": "hin_Deva", "pt": "por_Latn", "et": "est_Latn", "po": "pol_Latn",
             "pl": "pol_Latn", "ro": "ron_Latn", "ne": "npi_Deva", "si": "sin_Sinh", "mr": "mar_Deva",
             "cs": "ces_Latn", "km": "khm_Khmr", "ps": "pbt_Arab"}

source_labels = ["TrtS", "FixS"]
target_labels = ["TrtT", "FixT", "Dur"]

def to_device(inpt, device):
    return {k: inp.to(device) for k, inp in inpt.items()}

def has_numbers(input_str):
    if isinstance(input_str, str):
        return any(char.isdigit() for char in input_str)
    else:
        return False


def extension_by_level(src_level, tgt_level):
    if src_level == "token":
        return "st"
    if tgt_level == "token":
        return "tt"
    if src_level == "segment" or tgt_level == "segment":
        return "ag"
    else:
        return "sg"


class LineByLineCRITTDatasetByNLLBmGPT(IterableDataset):
    def __init__(self, tokenizer, tokenizer_lm, session, table_file, label_columns, src_level, tgt_level):
        super(IterableDataset, self).__init__()
        self.tokenizer = tokenizer # translation model tokenizer, nllb tokenizer
        self.tokenizer_lm = tokenizer_lm # monolingual lm tokenizer, mGPT tokenizer
        self.table_file = table_file
        self.session = session
        self.table = pd.read_csv(table_file, sep='\t').astype("string")

        self.labels = [label for label in label_columns if label in self.table.columns]

        self.src_level, self.tgt_level = src_level, tgt_level

        self.table, self.stid, self.ttid, self.stok_col, self.ttok_col, \
            self.sentence_table, self.s2t_sent_ids, self.t2s_sent_ids  = self.reformat_table_columns(
            self.table, src_level, tgt_level, self.table_file, drop_duplicates=True,
            keep_srcid_val=lambda x: has_numbers(x), keep_tgtid_val=lambda x: has_numbers(x))

        self.table = self.table.to_dict('records')
        try:
            self.src_lang = self.table[0]['SL']
            self.tgt_lang = self.table[0]['TL']
        except IndexError:
            print(session["study"], session["session_id"])
        self.get_sentence_offset()
        self.warnings = set()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        total_load = len(self.table)
        if worker_info is None:
            offset = 0
            load = total_load
        else:
            load = int(math.ceil(total_load / worker_info.num_workers))
            worker_id = worker_info.id
            offset = worker_id * load
        return iter(self.process_one_line(row) for row in self.table[offset: min(total_load, offset + load)])


    def print_warnings(self):
        if len(self.warnings) < 1:
            return
        else:
            print(self.session["study"], self.session['session_id'])
            print('\n'.join(self.warnings))
            print('\n')

    def change_fpath_ext(self, table_path, seg_level, source=True):
        if seg_level == "token" and source:
            ext = "st"
        elif seg_level == "token" and not source:
            ext = "tt"
        elif seg_level == "segment":
            ext = "ag"
        elif seg_level == "sentence":
            ext = "sg"
        else:
            raise Exception("Table extension not determined")

        table_dir = os.path.dirname(table_path)
        fname = f"{os.path.basename(table_path).split('.')[0]}.{ext}"

        return os.path.join(table_dir, fname)

    def load_sentence_table(self, table_path, filter_boundary_crosses=True, keep_srcid_val=lambda x: has_numbers(x),
                               keep_tgtid_val=lambda x: has_numbers(x)):
        sent_table_path = self.change_fpath_ext(table_path, "sentence")
        sentence_table = pd.read_csv(sent_table_path, sep='\t').astype("string")
        sentence_table = correct_table(sentence_table, sent_table_path)
        sentence_table = sentence_table[sentence_table['STseg'].apply(keep_srcid_val)]
        sentence_table = sentence_table[sentence_table['TTseg'].apply(keep_tgtid_val)]

        if filter_boundary_crosses:
            sentence_table = sentence_table[sentence_table['STseg'].apply(lambda val: len(val.split('+')) < 2)]
            sentence_table = sentence_table[sentence_table['TTseg'].apply(lambda val: len(val.split('+')) < 2)]

        return sentence_table


    def reformat_token_table(self, table, source=True, drop_duplicate_cols=[]):
        ## DATA CORRECTION: reorder tokens in target token tables with first '---' e.g., ENJA15/P32_T3
        if 'TTseg' in table.columns and not has_numbers(table.iloc[0]['TTseg']):
            table['Id'] = table['Id'].apply(lambda x: str(int(x) - 1))

        ## DATA CORRECTION: remove '---' in target token tables, retain only rows with valid segid and
        seg_col = 'TTseg' if not source else 'STseg'
        tok_col = 'SToken' if source else 'TToken'
        table = table[table[seg_col].apply(lambda x: has_numbers(x))]
        table = table[table[tok_col].apply(lambda tok: tok != "---")]

        if len(drop_duplicate_cols) > 0:
            table = table.drop_duplicates(subset=drop_duplicate_cols)
        return table


    def reformat_table_columns(self, table, src_seg_level, tgt_seg_level, table_path, drop_duplicates=False,
                               keep_srcid_val=lambda x: has_numbers(x),
                               keep_tgtid_val=lambda x: has_numbers(x)):
        def merge_token_table(table, table_path):
            token_table_path = self.change_fpath_ext(table_path, "token", source=True)
            token_table = pd.read_csv(token_table_path, sep='\t').astype("string")
            token_table = correct_table(token_table, token_table_path)
            merge_cols = ['SGroup', 'TGroup', 'STseg']
            if 'STid' in token_table.columns:
                tsid_col, ttid_col = 'STid', 'TTid'
            else:
                tsid_col, ttid_col = 'SGid', 'TGid'
            table = table.merge(token_table[['SGroup', 'TGroup', 'STseg', tsid_col, ttid_col]],
                                on=merge_cols, how="left")
            return table, tsid_col, ttid_col

        def get_sentence_pairs(table, table_path, filter_boundary_crosses=True):
            sentence_table = self.load_sentence_table(table_path, filter_boundary_crosses=filter_boundary_crosses,
                                                           keep_srcid_val=keep_srcid_val, keep_tgtid_val=keep_tgtid_val)

            have_rows = True if len(sentence_table) > 0 else False

            if "STseg" in table.columns:
                uniq_stsegs = set(sentence_table.STseg.values) if have_rows else set()
                table = table[table["STseg"].apply(lambda val: val in uniq_stsegs)]


            if "TTseg" in table.columns:
                uniq_ttsegs = set(sentence_table.TTseg.values) if have_rows else set()
                table = table[table["TTseg"].apply(lambda val: val in uniq_ttsegs)]

            sentence_table = sentence_table.drop_duplicates(subset=['STseg', 'TTseg'])

            s2t_sent_ids = dict(zip(sentence_table.STseg.values, sentence_table.TTseg.values)) if have_rows else {}
            t2s_sent_ids = dict(zip(sentence_table.TTseg.values, sentence_table.STseg.values)) if have_rows else {}

            return table, sentence_table, s2t_sent_ids, t2s_sent_ids


        sid_col, stok_col = None, None
        tid_col, ttok_col = None, None

        if src_seg_level == "token":
            sid_col, stok_col = "Id", "SToken"
        if tgt_seg_level == "token":
            tid_col, ttok_col = "Id", "TToken"

        if 'SAG' in table.columns and 'TAG' in table.columns:
            table = table.rename(columns={'SAG': 'SGroup', 'TAG': 'TGroup'})

        if src_seg_level == "segment":
            if 'STid' in table.columns:
                sid_col = 'STid'
            elif 'SGid' in table.columns:
                sid_col = 'SGid'
            else:
                table, tsid_col, ttid_col = merge_token_table(table, table_path)
                sid_col = tsid_col
            stok_col = "SGroup"

        if tgt_seg_level == "segment":
            if 'TTid' in table.columns:
                tid_col = 'TTid'
            elif 'TGid' in table.columns:
                tid_col = 'TGid'
            else:
                table, tsid_col, ttid_col = merge_token_table(table, table_path)
                tid_col = ttid_col
            ttok_col = "TGroup"

        if src_seg_level == "sentence" and 'STseg' in table.columns:
            sid_col = "STseg"

        if tgt_seg_level == "sentence" and "TTseg" in table.columns:
            tid_col = "TTseg"


        if len(table) > 0:

            if tgt_seg_level == "token":
                table = self.reformat_token_table(table, source=False)
            elif src_seg_level == "token":
                table = self.reformat_token_table(table, source=True)


        ## DATA CORRECTION: filter invalid id values
        if sid_col is not None:
            if keep_srcid_val is not None:
                table = table[table[sid_col].apply(keep_srcid_val)]

        if tid_col is not None:
            if keep_tgtid_val is not None:
                table = table[table[tid_col].apply(keep_tgtid_val)]


        ## DATA CORRECTION: manually correct inaccurate values, see data_correction.py
        table = correct_table(table, table_path)

        ## DATA CORRECTION: remove rows that cross sentence boundary
        table, sentence_table, s2t_sent_ids, t2s_sent_ids = get_sentence_pairs(table, table_path, filter_boundary_crosses=True)

        ## DATA CORRECTION: drop duplicates
        if drop_duplicates:
            cols = [col for col in [sid_col, tid_col] if col is not None]
            if len(cols) > 0:
                table = table.drop_duplicates(subset=cols)

        return table, sid_col, tid_col, stok_col, ttok_col, sentence_table, s2t_sent_ids, t2s_sent_ids


    def load_segoffset(self, table_path, keep_srcid_val=lambda x: has_numbers(x),
                               keep_tgtid_val=lambda x: has_numbers(x), source=True):
        '''Get segment offset via token tables'''
        tgt_table_path = self.change_fpath_ext(table_path, "token", source=source)
        tt_table = pd.read_csv(tgt_table_path, sep='\t').astype("string")
        tt_table = correct_table(tt_table,tgt_table_path)
        seg_col = 'TTseg' if not source else 'STseg'
        tok_col = 'SToken' if source else 'TToken'
        #### manually correct target token table with empty first row and inconsistent token ids
        if not source and not has_numbers(tt_table.iloc[0]['TTseg']):
            tt_table['Id'] = tt_table['Id'].apply(lambda x: str(int(x) - 1))
        ####

        tt_table = tt_table[tt_table[seg_col].apply(keep_tgtid_val)]
        tt_table = tt_table[tt_table[tok_col].apply(lambda tok: tok != "---")]

        segoffset = {}
        for strseg in set(tt_table[seg_col].values):
            tokenids = tt_table[tt_table[seg_col] == strseg]['Id'].apply(lambda x: int(x) - 1).values
            segoffset[int(strseg) - 1] = min(tokenids)
        return segoffset



    def get_sentence_offset(self):

        # rid of redundant sentences
        self.session['source_tokens'] = list(
            map(list, dict.fromkeys([tuple([tok for tok in seq if tok is not None]) for seq in self.session['source_tokens']])))
        self.session['target_tokens'] = list(
            map(list, dict.fromkeys([tuple([tok for tok in seq if tok is not None]) for seq in self.session['target_tokens']])))

        if len(self.sentence_table) > 0:
            strid2sent = dict(zip(self.sentence_table.TTseg.values, self.sentence_table.String.values))
            id2sent = {int(strid): sent for strid, sent in strid2sent.items() if strid in self.t2s_sent_ids}
            self.session['target_tokens'] = [[] for _ in range(max(id2sent.keys()))]
            for id, sent in id2sent.items():
                sent = [tok.rstrip() for tok in sent.split('_')]
                sent = [tok for tok in sent if tok != "---" and tok is not None]
                self.session['target_tokens'][id-1] = sent


        self.src_tokid2segid = [i for i, tokens in enumerate(self.session['source_tokens']) for _ in tokens]
        self.tgt_tokid2segid = [i for i, tokens in enumerate(self.session['target_tokens']) for _ in tokens]
        self.max_src_tokid = len(self.src_tokid2segid) - 1
        self.max_tgt_tokid = len(self.tgt_tokid2segid) - 1

        self.src_seg_offset = self.load_segoffset(self.table_file, source=True) # seg (sentence) offset by seg index
        self.tgt_seg_offset = self.load_segoffset(self.table_file, source=False)


    def get_token_id_sequence(self, row):
        def process_none_ids(sequence, token_ids, remove_oob_tokids=True):
            if remove_oob_tokids:
                maxid = len(sequence) - 1
                token_ids = list(filter(lambda i: i <= maxid, token_ids))
            if None not in sequence:
                return sequence, token_ids
            else:
                new_token_ids = dcopy(token_ids)
                new_sequence = []
                for curr_tokid, tok in enumerate(sequence):
                    if tok is not None:
                        new_sequence.append(tok)
                    else:
                        for i, tid in enumerate(token_ids):
                            if tid > curr_tokid:
                                new_token_ids[i] -= 1
                return new_sequence, new_token_ids


        def get_valid_token_id(tokens, source=True, sent_id=None):
            id_col = self.stid if source else self.ttid
            max_tok_id = self.max_src_tokid if source else self.max_tgt_tokid
            # tokid2segid = self.src_tokid2segid if source else self.tgt_tokid2segid
            seg_offset = self.src_seg_offset if source else self.tgt_seg_offset
            sentences = self.session['source_tokens'] if source else self.session['target_tokens']

            if has_numbers(str(row[id_col])):
                token_ids = np.array([int(i) - 1 for i in str(row[id_col]).split('+')])
                if not isinstance(tokens, str) or any(list(map(lambda sid: sid > max_tok_id, token_ids))):
                    return None, None, None
                # sent_id = tokid2segid[token_ids[0]] if sent_id is None else sent_id
                rel_token_ids = token_ids - seg_offset[sent_id]
                sentence = sentences[sent_id]
                sentence, rel_token_ids = process_none_ids(sentence, rel_token_ids, remove_oob_tokids=False)  # offset ids for sequences with None

                return sentence, sent_id, rel_token_ids

            return None, None, None

        def get_sent_id_by_row(source=True):
            if source:
                if 'STseg' in row and row['STseg'] in self.s2t_sent_ids:
                    return int(row['STseg']) - 1
                elif 'TTseg' in row and row['TTseg'] in self.t2s_sent_ids:
                    return int(self.t2s_sent_ids[row['TTseg']]) - 1
                else:
                    self.warnings.add("SOURCE SENTENCE ID cannot be found")
                    return None
            else:
                if 'TTseg' in row and row['TTseg'] in self.t2s_sent_ids:
                    return int(row['TTseg']) - 1
                elif 'STseg' in row and row['STseg'] in self.s2t_sent_ids:
                    return int(self.s2t_sent_ids[row['STseg']]) - 1
                else:
                    self.warnings.add("TARGET SENTENCE ID cannot be found")
                    return None


        src_tokens, tgt_tokens, src_sent, tgt_sent = None, None, None, None
        src_rel_token_ids, tgt_rel_token_ids = None, None
        src_sent_id = get_sent_id_by_row(source=True)
        tgt_sent_id = get_sent_id_by_row(source=False)

        if self.src_level != "sentence" and src_sent_id is not None:
            src_tokens = row[self.stok_col]
            src_sent, src_sent_id, src_rel_token_ids = get_valid_token_id(src_tokens, source=True, sent_id=src_sent_id)

        if self.tgt_level != "sentence" and tgt_sent_id is not None:
            tgt_tokens = row[self.ttok_col]
            tgt_sent, tgt_sent_id, tgt_rel_token_ids = get_valid_token_id(tgt_tokens, source=False, sent_id=tgt_sent_id)

        if self.src_level == "sentence" and src_sent_id is not None:
            src_sent = self.session['source_tokens'][src_sent_id]
            src_tokens = '_'.join(src_sent)
            src_rel_token_ids = list(range(len(src_sent)))

        if self.tgt_level == "sentence" and tgt_sent_id is not None:
            tgt_sent = self.session['target_tokens'][tgt_sent_id]
            tgt_tokens = '_'.join(tgt_sent)
            tgt_rel_token_ids = list(range(len(tgt_sent)))

        return src_tokens, src_rel_token_ids, tgt_tokens, tgt_rel_token_ids, src_sent, tgt_sent, src_sent_id, tgt_sent_id


    def extract_label_values(self, row, src_sent_id, tgt_sent_id):
        def get_sentence_label_value(sent_id, label, source=True):
            seg_col = "STseg" if source else "TTseg"
            sent_rows = self.sentence_table[self.sentence_table[seg_col] == str(sent_id+1)]
            if len(sent_rows) > 0 and label in sent_rows.columns:
                return float(sent_rows.iloc[0][label])
            else:
                return np.nan

        label_vals = {k: float(row[k]) for k in self.labels}

        if self.src_level == "sentence" and self.tgt_level != "sentence":
            for label in source_labels:
                if label in label_vals:
                    label_vals[label] = get_sentence_label_value(src_sent_id, label, source=True)
        elif self.src_level != "sentence" and self.tgt_level == "sentence":
            for label in target_labels:
                if label in label_vals:
                    label_vals[label] = get_sentence_label_value(tgt_sent_id, label, source=True)

        return [label_vals[label] for label in self.labels]


    def get_segment_position_quantile(self, token_ids, sentence):
        if token_ids is not None and sentence is not None:
            mean_token_pos = np.mean(np.array(token_ids))
            return mean_token_pos / len(sentence)
        else:
            return np.nan

    def process_one_line(self, row):
        token_id_sequence = self.get_token_id_sequence(row)
        if token_id_sequence is not None:

            src_tokens, src_rel_token_ids, tgt_tokens, tgt_rel_token_ids, src_seq, tgt_seq, \
                src_segid, tgt_segid  = token_id_sequence

            if src_rel_token_ids is not None and any([i < 0 for i in src_rel_token_ids]):
                print(row)
                print(f"source: {src_segid} {src_rel_token_ids}")

            elif tgt_rel_token_ids is not None and any([i < 0 for i in tgt_rel_token_ids]):
                print(row)
                print(f"target: {tgt_segid} {tgt_rel_token_ids}")

            elif all([item is not None for item in [src_tokens, src_rel_token_ids, src_segid, tgt_segid, src_seq, tgt_seq]]) and \
                all([len(item) > 0 for item in [src_tokens, src_rel_token_ids, src_seq, tgt_seq]]) :

                src_posq = self.get_segment_position_quantile(src_rel_token_ids, src_seq)
                tgt_posq = self.get_segment_position_quantile(tgt_rel_token_ids, tgt_seq)

                src_tokens, src_pos_ids, src_ids, src_att = self.tokenize_sequence(
                    src_tokens, src_rel_token_ids, src_seq, self.src_lang)

                tgt_tokens, tgt_pos_ids, tgt_ids, tgt_att = self.tokenize_sequence(
                    tgt_tokens, tgt_rel_token_ids, tgt_seq, self.tgt_lang)

                src_pos_ids_lm, src_ids_lm, src_att_lm = self.tokenize_sequence_lm(src_rel_token_ids, src_seq)
                tgt_pos_ids_lm, tgt_ids_lm, tgt_att_lm = self.tokenize_sequence_lm(tgt_rel_token_ids, tgt_seq)

                label_vals = self.extract_label_values(row, src_segid, tgt_segid)

                return (src_tokens, tgt_tokens, src_pos_ids, tgt_pos_ids, src_posq, tgt_posq, src_segid, tgt_segid,
                        src_ids, tgt_ids, src_att, tgt_att, src_pos_ids_lm, tgt_pos_ids_lm, src_ids_lm, tgt_ids_lm,
                        src_att_lm, tgt_att_lm, label_vals)

        return None


    def collate_encode_fn(self, lines):
        lines = [line for line in lines if line is not None]

        if lines:
            src_tokens,  tgt_tokens, src_token_ids, tgt_token_ids, src_posqs, tgt_posqs, src_segids, tgt_segids, \
                src_encoder_ids, tgt_encoder_ids, src_atts, tgt_atts, src_pos_ids_lm, tgt_pos_ids_lm, \
                src_encoder_ids_lm, tgt_encoder_ids_lm, src_atts_lm, tgt_atts_lm, ys = zip(*lines)

            src_seq_ids = pad_sequence(list(src_encoder_ids), batch_first=True, padding_value=self.tokenizer.pad_token_id)
            tgt_seq_ids = pad_sequence(list(tgt_encoder_ids), batch_first=True, padding_value=self.tokenizer.pad_token_id)
            src_atts, tgt_atts = pad_sequence(list(src_atts), batch_first=True), pad_sequence(list(tgt_atts), batch_first=True)

            src_encodings = BatchEncoding(data={'input_ids': src_seq_ids, 'attention_mask': src_atts,
                                                'token_type_ids': torch.zeros_like(src_seq_ids)})
            tgt_encodings = BatchEncoding(data={'input_ids': tgt_seq_ids, 'attention_mask': tgt_atts,
                                                'token_type_ids': torch.zeros_like(tgt_seq_ids)})

            src_seq_ids_lm = pad_sequence(list(src_encoder_ids_lm), batch_first=True, padding_value=1)
            tgt_seq_ids_lm = pad_sequence(list(tgt_encoder_ids_lm), batch_first=True, padding_value=1)
            src_atts_lm, tgt_atts_lm = pad_sequence(list(src_atts_lm), batch_first=True), pad_sequence(list(tgt_atts_lm), batch_first=True)

            src_encodings_lm = BatchEncoding(data={'input_ids': src_seq_ids_lm, 'attention_mask': src_atts_lm})
            tgt_encodings_lm = BatchEncoding(data={'input_ids': tgt_seq_ids_lm, 'attention_mask': tgt_atts_lm})

            ys = [torch.tensor(y, dtype=torch.float) for y in ys]
            ys = torch.stack(ys)  # [bsz, #labels]

            return src_tokens, src_token_ids, tgt_tokens, tgt_token_ids, src_posqs, tgt_posqs, src_segids, tgt_segids, \
                src_encodings, tgt_encodings, src_pos_ids_lm, tgt_pos_ids_lm, src_encodings_lm, tgt_encodings_lm, ys
        else:
            return None


    def get_rel_ids(self, tokenized_ids, target_ids, special_token_offset):
        new_target_ids = []

        for tid in target_ids:
            new_offset = sum([tokids.size(0) for tokids in tokenized_ids[:tid]])
            for i in range(len(tokenized_ids[tid])):
                new_target_ids.append(new_offset + i + special_token_offset)
        return new_target_ids

    def tokenize_sequence(self, src_tokens, src_rel_token_ids, src_seq, lang):
        # for nllb, a translation model
        processed_src_seq = src_seq
        src_ids = [
            self.tokenizer(tok, return_tensors='pt', padding=False, add_special_tokens=False)["input_ids"].view(-1)
            for tok in processed_src_seq]
        if src_rel_token_ids is not None:
            src_rel_token_ids = self.get_rel_ids(src_ids, src_rel_token_ids, 1)    # token id by tokenizer scheme
        src_ids = [torch.tensor([self.tokenizer.lang_code_to_id[lang_code[lang]]])] + src_ids + \
                  [torch.tensor([self.tokenizer.eos_token_id])]

        src_ids = torch.cat(src_ids).view(-1).to(dtype=torch.long)
        src_att = torch.ones_like(src_ids)

        return src_tokens, src_rel_token_ids, src_ids, src_att


    def tokenize_sequence_lm(self, src_rel_token_ids, src_seq):
        # for mGPT, a monolingual langauge model
        src_ids = [self.tokenizer_lm(tok, return_tensors='pt', padding=False, add_special_tokens=False)["input_ids"].view(-1)
            for tok in src_seq]
        if src_rel_token_ids is not None:
            src_rel_token_ids = self.get_rel_ids(src_ids, src_rel_token_ids, 0)
        try:
            src_ids = torch.cat(src_ids).view(-1).to(dtype=torch.long)
        except:
            print(src_seq)
            print(src_rel_token_ids)
        src_att = torch.ones_like(src_ids)
        return src_rel_token_ids, src_ids, src_att



class LineByLineQEDataset(IterableDataset):
    def __init__(self, tokenizer, src_lang, tgt_lang, combine_files=None, src_col=None, tgt_col=None, sc_col=None,
                 source_file=None, target_file=None, score_file=None, tokenized=False, quoting=0):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        self.combine_files = combine_files
        self.tokenized = tokenized

        if combine_files is not None:
            source_lines, target_lines, scores = [], [], []
            for combine_file in combine_files:
                sep = '\t' if combine_file.split('.')[-1] == 'tsv' else ','
                try:
                    df = pd.read_csv(combine_file, sep=sep, quoting=quoting)
                except:
                    print("BAD LINES")
                    print(combine_file)
                    df = pd.read_csv(combine_file, sep=sep, quoting=quoting, on_bad_lines='skip')
                if src_col not in df.columns:
                    for col in ['source', 'src', 'original']:
                        if col in df.columns:
                            src_col = col
                            break
                if tgt_col not in df.columns:
                    for col in ['target', 'tgt', 'translation', 'translations', 'mt']:
                        if col in df.columns:
                            tgt_col = col
                            break
                try:
                    source_lines += list(df[src_col].values)
                    target_lines += list(df[tgt_col].values)
                    scores += list(df[sc_col].astype('float').values)
                except:
                    print("src / tgt /sc columns not found")
                    print(combine_file)
                    print(df.columns)

        else:

            with open(source_file, 'r') as f:
                source_lines = [line.split('\t')[0] for line in f.read().rstrip().split('\n')]

            with open(target_file, 'r') as f:
                target_lines = [line.split('\t')[0] for line in f.read().rstrip().split('\n')]

            with open(score_file, 'r') as f:
                scores = f.read().rstrip().split('\n')
            scores = [float(score) for score in scores]

        self.data = [(src, tgt, sc) for src, tgt, sc in zip(source_lines, target_lines, scores) if isinstance(src, str) and isinstance(tgt, str)]



    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        total_load = len(self.data)
        if worker_info is None:
            offset = 0
            load = total_load
        else:
            load = int(math.ceil(total_load / worker_info.num_workers))
            worker_id = worker_info.id
            offset = worker_id * load
        return iter([self.process_one_pair(pair) for pair in self.data[offset: min(total_load, offset + load)]])


    def process_one_pair(self, trans_pair):
        return trans_pair

    def collate_encode_fn(self, trans_tup):
        src_lines, tgt_lines, scs = zip(*trans_tup)
        try:
            inputs = self.tokenizer(src_lines, text_target=tgt_lines, is_split_into_words=self.tokenized, return_tensors='pt', padding=True, truncation=True)
        except:
            print("UNABLE TO TOKENIZE")
            print(self.combine_files)
            print(src_lines)
            print(tgt_lines)
            return None

        # select only non-special tokens (without language token in the beginning and eos token)
        src_ids = [torch.arange(1, (ids!=self.tokenizer.pad_token_id).sum() - 1, dtype=torch.int32) for ids in inputs.input_ids]
        tgt_ids = [torch.arange(1, (ids != self.tokenizer.pad_token_id).sum() - 1, dtype=torch.int32) for ids in inputs.labels]

        return src_lines, tgt_lines, inputs, src_ids, tgt_ids, torch.tensor(scs, dtype=torch.float)



class LineByLineQEWordLevelDataset(IterableDataset):

    def __init__(self, tokenizer, src_lang, tgt_lang, source_file=None, target_file=None, tag_file=None,
                 alignment_file=None):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        with open(source_file, 'r') as f:
            source_lines = f.read().rstrip().split('\n')
        self.source_lines = [line.split(' ') for line in source_lines]

        with open(target_file, 'r') as f:
            target_lines = f.read().rstrip().split('\n')
        self.target_lines = [line.split(' ') for line in target_lines]

        with open(tag_file, 'r') as f:
            tags = f.read().rstrip().split('\n')
        self.tags = [tag.split(' ') for tag in tags]

        with open(alignment_file, 'r') as f:
            alignments = f.read().rstrip().split('\n')
        self.alignments = [line.split(' ') for line in alignments]

        self.data = []
        self.non_aligned_tgt_tokens = []

        for seg_id, (tgt_line, src_line, tgs, algs) in enumerate(zip(self.target_lines, self.source_lines, self.tags, self.alignments)):
            algs = [al.split('-') for al in algs]
            algs_dict = {}
            for al in algs:
                src_id, tgt_id = int(al[0]), int(al[1])
                algs_dict[tgt_id] = algs_dict.get(tgt_id, []) + [src_id]
            for tgt_id, (tgt_tok, tg) in enumerate(zip(tgt_line, tgs)):
                if tgt_id in algs_dict:
                    src_ids = algs_dict[tgt_id]
                    src_toks = [src_line[id] for id in src_ids]
                    tg = 1 if tg == 'BAD' else 0
                    self.data.append((seg_id, tgt_tok, tgt_id, src_toks, src_ids, tg))
                else:
                    self.non_aligned_tgt_tokens.append(tgt_tok)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        total_load = len(self.data)
        if worker_info is None:
            offset = 0
            load = total_load
        else:
            load = int(math.ceil(total_load / worker_info.num_workers))
            worker_id = worker_info.id
            offset = worker_id * load
        return iter([self.process_one_pair(pair) for pair in self.data[offset: min(total_load, offset + load)]])


    def process_one_pair(self, row):
        seg_id, tgt_tok, tgt_id, src_toks, src_ids, tg = row
        tgt_seq, src_seq = self.target_lines[seg_id], self.source_lines[seg_id]
        tgt_tokens, tgt_ids, tgt_encoded, tgt_att = self.tokenize_sequence([tgt_tok], [tgt_id], tgt_seq, self.tgt_lang)
        src_tokens, src_ids, src_encoded, src_att = self.tokenize_sequence(src_toks, src_ids, src_seq, self.src_lang)
        return tgt_tokens, tgt_ids, tgt_encoded, tgt_att, src_tokens, src_ids, src_encoded, src_att, seg_id, tg

    def collate_encode_fn(self, lines):
        lines = [line for line in lines if line is not None]

        if lines:
            tgt_tokens, tgt_ids, tgt_encoded, tgt_atts, src_tokens, src_ids, src_encoded, src_atts, seg_id, tgs = zip(*lines)

            src_seq_ids = pad_sequence(list(src_encoded), batch_first=True,
                                       padding_value=self.tokenizer.pad_token_id)
            tgt_seq_ids = pad_sequence(list(tgt_encoded), batch_first=True,
                                       padding_value=self.tokenizer.pad_token_id)
            src_atts, tgt_atts = pad_sequence(list(src_atts), batch_first=True), pad_sequence(list(tgt_atts),
                                                                                              batch_first=True)

            src_encodings = BatchEncoding(data={'input_ids': src_seq_ids, 'attention_mask': src_atts,
                                                'token_type_ids': torch.zeros_like(src_seq_ids)})
            tgt_encodings = BatchEncoding(data={'input_ids': tgt_seq_ids, 'attention_mask': tgt_atts,
                                                'token_type_ids': torch.zeros_like(tgt_seq_ids)})

            return src_tokens, src_ids, tgt_tokens, tgt_ids, src_encodings, tgt_encodings, seg_id, tgs
        else:
            return None


    def tokenize_sequence(self, src_tokens, src_rel_token_ids, src_seq, lang):
        def get_rel_ids(tokenized_ids, target_ids, special_token_offset):
            new_target_ids = []

            for tid in target_ids:
                new_offset = sum([tokids.size(0) for tokids in tokenized_ids[:tid]])
                for i in range(len(tokenized_ids[tid])):
                    new_target_ids.append(new_offset + i + special_token_offset)
            return new_target_ids

        processed_src_seq = src_seq
        src_ids = [
            self.tokenizer(tok, return_tensors='pt', padding=False, add_special_tokens=False)["input_ids"].view(-1)
            for tok in processed_src_seq]
        if src_rel_token_ids is not None:
            src_rel_token_ids = get_rel_ids(src_ids, src_rel_token_ids, 1)    # token id by tokenizer scheme
        src_ids = [torch.tensor([self.tokenizer.lang_code_to_id[lang_code[lang]]])] + src_ids + \
                  [torch.tensor([self.tokenizer.eos_token_id])]

        src_ids = torch.cat(src_ids).view(-1).to(dtype=torch.long)
        src_att = torch.ones_like(src_ids)

        return src_tokens, src_rel_token_ids, src_ids, src_att
