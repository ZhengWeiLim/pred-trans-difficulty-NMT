import torch, sys
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from scipy import stats
import scipy.special as special
from torch.utils.data import IterableDataset, Dataset, DataLoader
import torch.optim as optim
import pickle, os, copy, json
from torchmetrics import ExplainedVariance, R2Score, PearsonCorrCoef, SpearmanCorrCoef
torch.manual_seed(0)

seg2seg_tup = [("token", "segment"), ("token", "sentence"), ("segment", "token"), ("segment", "segment"),
                   ("segment", "sentence"), ("sentence", "token"), ("sentence", "segment"), ("sentence", "sentence")]
seg2seg = ['-'.join(s2s) for s2s in seg2seg_tup]
seg_order = {"token": 0, "segment": 1, "sentence": 2}

attentions, methods = ["eatt", "xatt", "datt"], ["cov", "conf", "var", "ent", "eos"]
nmt_features = ['xent', 'xmi']
att_features = {}
nlayers = 12
nheads = 16

attribute_methods = [(att, met) for att in attentions for met in methods if not (att == 'datt' and  met == 'eos')]

layer_head = [f"{layer:02d}{head:02d}" for layer in range(1, nlayers + 1) for head in range(1, nheads + 1)]

for att, met in attribute_methods:
    features = [f"{att}_{met}_{lh}" for lh in layer_head]
    nmt_features += features
    att_features[f"{att}_{met}"] = features

mean_layer_features = {att_met: [f"{att_met}_meanL_meanH"] for att_met in
                       att_features}
max_layer_features = {att_met: [f"{att_met}_meanL_maxH"] for att_met in
                      att_features}



max_enc_features = [ft for att, met in attribute_methods for ft in max_layer_features[f"{att}_{met}"] if att == 'eatt']
max_dec_features = [ft for att, met in attribute_methods for ft in max_layer_features[f"{att}_{met}"] if att == 'datt']

max_cross_conf_ent_eos = max_layer_features[f"xatt_conf"] + max_layer_features[f"xatt_ent"] + max_layer_features[f"xatt_eos"]
max_cross_cov_var = [ft for met in ["cov", "var"] for ft in max_layer_features[f"xatt_{met}"]]

odist = ['xmi', 'xent']

sides = ['src', 'tgt']
side2labels = {'tgt': ['log_Dur', 'log_TrtT'], 'src': ['log_TrtS']}
levels = list(seg_order.keys())
sseg2tseg, tseg2sseg = {}, {}
for src_level, tgt_level in seg2seg_tup:
    sseg2tseg[src_level] = sseg2tseg.get(src_level, []) + [tgt_level]
    tseg2sseg[tgt_level] = tseg2sseg.get(tgt_level, []) + [src_level]


def get_side_level_mean_max_features():
    def add_suffix(lst, suffix_lst):
        return [f'{val}_{"_".join(suffix_lst)}' for val in lst]

    max_features = {'src': {level: ["src_seg_len", "src_posq"] if level != 'sentence' else ["src_seg_len"]  for level in levels},
                    'tgt': {level: ["tgt_seg_len", "tgt_posq"]  if level != 'sentence' else ["tgt_seg_len"] for level in levels}}
    mean_features = {'src': {level: ["src_seg_len", "src_posq"]  if level != 'sentence' else ["src_seg_len"] for level in levels},
                     'tgt': {level: ["tgt_seg_len", "tgt_posq"] if level != 'sentence' else ["tgt_seg_len"] for level in levels}}

    for side in sides:
        max_features[side]['sentence'].append('transquest_score')
        mean_features[side]['sentence'].append('transquest_score')

    for level in levels:
        max_src_features = copy.deepcopy(max_enc_features)
        for tlevel in sseg2tseg[level]:
            max_src_features += add_suffix(max_dec_features + max_cross_conf_ent_eos + max_cross_cov_var + odist,
                                           [tlevel])
        max_features["src"][level] += max_src_features
        mean_features["src"][level] += [ft.replace('max', 'mean') for ft in max_src_features]

        max_tgt_features = copy.deepcopy(odist + max_dec_features + max_cross_conf_ent_eos)
        for slevel in tseg2sseg[level]:
            new = add_suffix(max_cross_cov_var + max_enc_features, [slevel])
            max_tgt_features += new
        max_features["tgt"][level] += max_tgt_features
        mean_features["tgt"][level] += [ft.replace('max', 'mean') for ft in max_tgt_features]

    return mean_features, max_features


class ProcessDataset(Dataset):
    def __init__(self, data, feature_cols, label):
        super(ProcessDataset, self).__init__()
        self.data = data[feature_cols + [label]].dropna()
        self.label = label
        self.features = feature_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row[self.features].values, row[self.label]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)


def evaluate(model, data, loss_fn, device):
    model.eval()
    total_loss = 0
    ys, y_preds = [], []
    with torch.no_grad():
        for X, y in data:
            y = y.float().to(device)
            y_pred = model(X.float().to(device)).view(-1)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            y_preds.append(y_pred)
            ys.append(y)
        return total_loss / len(data), torch.cat(y_preds), torch.cat(ys)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, s2s, train_dfs, dev_dfs, test_dfs, features, label, loss_fn, lr, bsz, n_epochs, device="cpu", suffix=None,
          save_dir='model', num_workers=1, l1_penalty=0.1, l2_penalty=0.1):
    models = []
    expl_var_, r2score_  = ExplainedVariance().to(device), R2Score().to(device)
    pearson_, spearman_ = PearsonCorrCoef().to(device), SpearmanCorrCoef().to(device)
    def get_corr_sig(corr_func, corr_name, x, y):
        r = corr_func(x,y).item()
        sig = ttest_finish(len(x), corr_name, r)
        return r, sig

    metrics = {
        "explained_variance": lambda a, b: expl_var_(a, b).item(),
        "r2score": lambda a, b: r2score_(a, b).item(),
        "pearson": lambda a, b: get_corr_sig(pearson_, "pearson", a, b),
        "spearman": lambda a, b: get_corr_sig(spearman_, "spearman", a, b)
    }

    best_scores = {met_name: [] for met_name in metrics}
    best_scores['coef'], best_scores['aic'], best_scores['bic'] = [], [], []
    best_scores['train_loss'], best_scores['val_loss'], best_scores['test_loss'] = [], [], []

    for i, (train_df, dev_df, test_df) in enumerate(zip(train_dfs, dev_dfs, test_dfs)):

        train_dt = ProcessDataset(train_df, features, label)
        train_dataloader = DataLoader(train_dt, bsz, shuffle=True, num_workers=num_workers)
        val_dt = ProcessDataset(dev_df, features, label)
        val_dataloader = DataLoader(val_dt, bsz)
        test_dt = ProcessDataset(test_df, features, label)
        test_dataloader = DataLoader(test_dt, bsz)

        train_sz, val_sz = len(train_dt), len(val_dt)

        if train_sz == 0 or val_sz == 0:
            print(f"Train / val {i} dataset is empty, training skipped")
            return

        model.apply(init_weights)
        device = torch.device(device)
        optimizer = optim.ASGD(model.parameters(), lr=lr, weight_decay=l2_penalty)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        best_loss = np.inf
        best_params = None
        train_loss = []
        valid_loss = []
        best_epoch = None
        best_curr_scores = {met_name: None for met_name in metrics}
   
        print(f"\nTraining for {label} given {s2s}")
        print(f"Train data size: {train_sz}, validation data size: {val_sz}")

        early_stopper = EarlyStopper(patience=3, min_delta=0)
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0

            for bid, batch in enumerate(train_dataloader):
                X, y = batch
                y = y.float().to(device)
                y_pred = model(X.float().to(device)).view(-1)

                # L1 penalty
                l1_regularization = 0.
                for param in model.parameters():
                    l1_regularization += param.abs().sum()

                loss = loss_fn(y_pred, y) + l1_penalty * l1_regularization

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss.item()

            val_loss, val_preds, val_labels = evaluate(model, val_dataloader, loss_fn, device)
            if val_loss < best_loss or best_params is None:
                best_params = copy.deepcopy(model.state_dict())
                best_loss = val_loss
                best_epoch = epoch+1
                for met_name, met_func in metrics.items():
                    best_curr_scores[met_name] = met_func(val_preds, val_labels)

            train_loss.append(total_loss / train_sz)
            valid_loss.append(val_loss)
            scheduler.step()

            val_explained_var = metrics["explained_variance"](val_preds, val_labels)
            val_r2score = metrics["r2score"](val_preds, val_labels)
            print(
                f"Epoch {epoch + 1}: train loss={total_loss / train_sz:.4f}, validation loss={val_loss:.4f}, explained variance={val_explained_var:.4f}, r2={val_r2score:.4f}")

            if early_stopper.early_stop(val_loss):
                print(f"Stopping early at epoch {epoch+1}... val loss = {val_loss} has stopped decreasing for at least 3 epochs")
                break

        if suffix is not None:
            model_fname = f"{s2s}_ep={best_epoch}_run={i}_{suffix}.pt"
        else:
            model_fname = f"{s2s}_ep={best_epoch}_run={i}.pt"

        model_path = os.path.join(save_dir, model_fname)
        torch.save(best_params, model_path)
        model.load_state_dict(best_params)
        models.append(copy.deepcopy(model))

        test_loss, test_preds, test_labels = evaluate(model, test_dataloader, loss_fn, device)
        test_scores = {met_name: met_func(test_preds, test_labels) for met_name, met_func in metrics.items()}

        val_loss, val_preds, _ = evaluate(model, val_dataloader, loss_fn, device)
        train_loss, train_preds, _ = evaluate(model, train_dataloader, loss_fn, device)


        for met_name, sc in test_scores.items():
            best_scores[met_name].append(sc)

        best_scores['coef'].append(dict(zip(features, model.linear.weight.view(-1).tolist())))

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        k = sum([np.prod(p.size()) for p in model_parameters]) # num_params
        n = len(test_preds) + len(val_preds) + len(train_preds)
        mse_loss = (train_loss*len(train_preds) + val_loss*len(val_preds) + test_loss*len(test_preds)) / n
        best_scores['aic'].append(2*k + n * np.log(mse_loss)) # 2k + n ln (RSS/m)
        best_scores['bic'].append(n * np.log(mse_loss) + k * np.log(n)) # n ln (RSS/n) + k ln (n)
        best_scores['train_loss'].append(train_loss)
        best_scores['val_loss'].append(val_loss)
        best_scores['test_loss'].append(test_loss)

        print("Training complete!\n")

        loss_fname = f"train_val_loss_{model_fname.split('.')[0]}.json"
        loss_dict = {"train": train_loss, "valid": valid_loss}
        loss_dict.update(best_curr_scores)
        with open(os.path.join(save_dir, loss_fname), 'w') as f:
            f.write(json.dumps(loss_dict))

    test_fname = f"test_score_{s2s}.json"
    with open(os.path.join(save_dir, test_fname), 'w') as f:
        f.write(json.dumps(best_scores))

    return models, best_scores


def average_predict_df(models, df, feature_cols, label, device, suffix="pred"):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y_preds = []
    for model in models:
        y_pred = model(X.to(device)).view(-1)
        y_preds.append(y_pred)

    y_mean_pred = torch.stack(y_preds).mean(dim=0)
    df[f"{label}_{suffix}"] = y_mean_pred.tolist()
    return df


def train_and_predict_by_levels(data, train_dt, dev_dt, test_dt, omodel_dict, feature_dict, label_dict, loss_fn, lr, bsz, n_epochs,
                                device="cpu", pred_suffix="pred_max", savef_suffix=None, save_dir='model', num_workers=1, penalty=0.1):
    trained_models, best_scores = {}, {}
    for side, level_dt in data.items():
        for level, df in level_dt.items():
            for label in label_dict[side]:
                s2s = f'{side}-{level}-{label}'
                if level in feature_dict[side]:
                    current_features = feature_dict[side][level]
                    model_dict = copy.deepcopy(omodel_dict)
                    trained_models[s2s], best_scores[s2s] = train(
                        model_dict[side][level], s2s, train_dt[side][level][label], dev_dt[side][level][label], test_dt[side][level][label], current_features,
                        label, loss_fn, lr, bsz, n_epochs, device=device, suffix=savef_suffix, save_dir=save_dir, num_workers=num_workers,
                        l1_penalty=penalty, l2_penalty=penalty
                    )

                    data[side][level] = average_predict_df(trained_models[s2s], df, current_features, label, device,
                                                       suffix=pred_suffix)
    return data, trained_models, best_scores


def get_train_dev_test(data, feature_dict, labels, major_train_studies, major_test_studies, random_state=42, nsets=5,
                   save_dir=None):
    random.seed(random_state)

    train_data = {side: {level: {label: [] for label in labels[side]} for level in data[side]} for side in data}
    dev_data = {side: {level: {label: [] for label in labels[side]} for level in data[side]} for side in data}
    test_data = {side: {level: {label: [] for label in labels[side]} for level in data[side]} for side in data}

    for side, level_dt in data.items():
        for level, df in level_dt.items():
            otrain_df = df[df['study'].apply(lambda x: x in major_train_studies)]
            otest_df = df[df['study'].apply(lambda x: x in major_test_studies)]
            for label in labels[side]:
                features = feature_dict[side][level]
                test_df = otest_df[features + [label, "study", "text", "src_segid"]].dropna()
                test_df = test_df.astype({"text": str, "src_segid": str})
                test_df["study-text-seg"] = test_df["study"] + '-' + test_df["text"] + '-' + test_df["src_segid"]
                texts = list(test_df["study-text-seg"].unique())

                train_df = otrain_df[features+ [label, "study", "text", "src_segid"]].dropna()
                train_df = train_df.astype({"text": str, "src_segid": str})
                train_df["text-seg"] = train_df["text"] + '-' + train_df["src_segid"]
                train_texts = list(train_df["text-seg"].unique())

                for i in range(nsets):
                    test_text_ids = random.sample(texts, 10)
                    dev_text_ids = random.sample(test_text_ids, 5)
                    test0 = test_df[test_df["study-text-seg"].apply(lambda x: x in test_text_ids and x not in dev_text_ids)][features + [label]]
                    dev0 = test_df[test_df["study-text-seg"].apply(lambda x: x in dev_text_ids)][features + [label]]
                    train1 = test_df.drop(test0.index).drop(dev0.index)[features + [label]]

                    train_test_text_ids = random.sample(train_texts, 6)
                    train_dev_text_ids = random.sample(train_test_text_ids, 3)
                    test1 = train_df[train_df["text-seg"].apply(lambda x: x in train_test_text_ids and x not in train_dev_text_ids)][features + [label]]
                    dev1 = train_df[train_df["text-seg"].apply(lambda x: x in train_dev_text_ids)][features + [label]]
                    train0 = train_df.drop(test1.index).drop(dev1.index)[features + [label]]

                    new_train = train0.append(train1)
                    new_test = test0.append(test1)
                    new_dev= dev0.append(dev1)

                    if save_dir is not None:
                        new_train.to_csv(os.path.join(save_dir, f'train-{side}-{level}-{label}_{i}.csv'), index=False)
                        new_test.to_csv(os.path.join(save_dir, f'test-{side}-{level}-{label}_{i}.csv'), index=False)
                        new_dev.to_csv(os.path.join(save_dir, f'dev-{side}-{level}-{label}_{i}.csv'), index=False)

                    train_data[side][level][label].append(new_train)
                    test_data[side][level][label].append(new_test)
                    dev_data[side][level][label].append(new_dev)

    return train_data, dev_data, test_data


class LinearRegression(nn.Module):
    def __init__(self, ninput):
        super().__init__()
        self.linear = nn.Linear(ninput, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, ninput):
        super().__init__()
        self.linear = nn.Linear(ninput, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))



def read_train_dev_test(dir, nsets=5):
    train_data = {side: {level: {label: [] for label in side2labels[side]} for level in levels} for side in sides}
    dev_data = {side: {level: {label: [] for label in side2labels[side]} for level in levels} for side in sides}
    test_data = {side: {level: {label: [] for label in side2labels[side]}  for level in levels} for side in sides}

    for side in sides:
        for level in levels:
            for label in side2labels[side]:
                for i in range(nsets):
                    train_i = pd.read_csv(os.path.join(dir, f'train-{side}-{level}-{label}_{i}.csv'))
                    train_data[side][level][label].append(train_i)
                    test_i = pd.read_csv(os.path.join(dir, f'test-{side}-{level}-{label}_{i}.csv'))
                    test_data[side][level][label].append(test_i)
                    dev_i = pd.read_csv(os.path.join(dir, f'dev-{side}-{level}-{label}_{i}.csv'))
                    dev_data[side][level][label].append(dev_i)

    return train_data, dev_data, test_data

# modified from scipy source code
def ttest_finish(n_obs, corr_type, r):
    if corr_type == "spearman":
        dof = n_obs - 2
        t = r * np.sqrt((dof/((r+1.0)*(1.0-r))))
        pval = special.stdtr(dof, -np.abs(t))*2
        return pval
    elif corr_type == "pearson":
        n = n_obs
        ab = n/2 - 1
        dist = stats.beta(ab, ab, loc=-1, scale=2)
        prob = 2*dist.sf(abs(r))
        return prob



def ablate_features(features, ablated_features):
    ffeatures = copy.deepcopy(features)
    ffeatures = list(filter(lambda ft: all([tp not in ft for tp in ablated_features]), ffeatures))
    return ffeatures