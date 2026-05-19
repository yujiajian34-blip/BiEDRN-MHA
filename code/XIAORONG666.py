import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import re, gc, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from collections import Counter
from tqdm import tqdm


# ==========================================
# 1. 全局配置
# ==========================================
SEED, MAX_LEN, VOCAB_SIZE = 42, 128, 25000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILE = r'D:\INJECT\data30w\combined_dataset.parquet'
SAVE_DIR  = r'D:\INJECT\XIAORONG_CHULI_UD'
LOG_FILE  = os.path.join(SAVE_DIR, "edrn_ablation_results.csv")
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab_fixed.json")

BATCH_SIZE = 512
EPOCHS = 50

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 2. 严格确定性
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ==========================================
# 3. FGM
# ==========================================
class FGM:
    def __init__(self, model):
        self.model, self.backup = model, {}

    def attack(self, epsilon=0.3, emb_name='emb'):
        for name, p in self.model.named_parameters():
            if p.requires_grad and emb_name in name:
                self.backup[name] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    p.data.add_(epsilon * p.grad / norm)

    def restore(self, emb_name='emb'):
        for name, p in self.model.named_parameters():
            if p.requires_grad and emb_name in name:
                p.data = self.backup[name]
        self.backup = {}


# ==========================================
# 4. EDRN 核心算子
# ==========================================
@torch.jit.script
def edrn_step(gates_t, pd_t, alpha_prev, units: int, d_max: int):
    z, r, g, o = torch.split(gates_t, units, dim=-1)
    z, r, o = torch.sigmoid(z), torch.sigmoid(r), torch.sigmoid(o)

    p_d = torch.softmax(pd_t.view(-1, units, d_max), dim=-1)

    alpha_shift = torch.zeros_like(alpha_prev)
    alpha_shift[:, :, :-1] = alpha_prev[:, :, 1:]

    alpha_new = z.unsqueeze(-1) * alpha_shift \
              + r.unsqueeze(-1) * torch.tanh(g).unsqueeze(-1) * p_d

    h = o * torch.tanh(torch.sum(alpha_new, dim=-1))
    return alpha_new, h


# ==========================================
# 5. EDRN Layer
# ==========================================
class BiEDRNLayer(nn.Module):
    def __init__(self, in_dim, units=64, d_max=2):
        super().__init__()
        self.units, self.d_max = units, d_max

        self.f_proj = nn.Linear(in_dim, 4 * units + units * d_max)
        self.f_m_proj = nn.Linear(units, 4 * units + units * d_max, bias=False)

        self.b_proj = nn.Linear(in_dim, 4 * units + units * d_max)
        self.b_m_proj = nn.Linear(units, 4 * units + units * d_max, bias=False)

    def forward(self, x):
        b, s, _ = x.size()

        fm = torch.zeros(b, self.units, device=x.device)
        fa = torch.zeros(b, self.units, self.d_max, device=x.device)

        fx = self.f_proj(x)
        f_outs = []
        for t in range(s):
            gt, pt = torch.split(
                fx[:, t, :] + self.f_m_proj(fm),
                [4 * self.units, self.units * self.d_max], dim=-1)
            fa, fm = edrn_step(gt, pt, fa, self.units, self.d_max)
            f_outs.append(fm.unsqueeze(1))

        bm = torch.zeros(b, self.units, device=x.device)
        ba = torch.zeros(b, self.units, self.d_max, device=x.device)

        bx = self.b_proj(torch.flip(x, [1]))
        b_outs = []
        for t in range(s):
            gt, pt = torch.split(
                bx[:, t, :] + self.b_m_proj(bm),
                [4 * self.units, self.units * self.d_max], dim=-1)
            ba, bm = edrn_step(gt, pt, ba, self.units, self.d_max)
            b_outs.append(bm.unsqueeze(1))

        return torch.cat(
            [torch.cat(f_outs, 1),
             torch.flip(torch.cat(b_outs, 1), [1])],
            dim=-1
        )


class EDRNLayer(nn.Module):
    def __init__(self, in_dim, units=64, d_max=2):
        super().__init__()
        self.units, self.d_max = units, d_max
        self.proj = nn.Linear(in_dim, 4 * units + units * d_max)
        self.m_proj = nn.Linear(units, 4 * units + units * d_max, bias=False)

    def forward(self, x):
        b, s, _ = x.size()
        m = torch.zeros(b, self.units, device=x.device)
        a = torch.zeros(b, self.units, self.d_max, device=x.device)

        xs = self.proj(x)
        outs = []
        for t in range(s):
            gt, pt = torch.split(
                xs[:, t, :] + self.m_proj(m),
                [4 * self.units, self.units * self.d_max], dim=-1)
            a, m = edrn_step(gt, pt, a, self.units, self.d_max)
            outs.append(m.unsqueeze(1))
        return torch.cat(outs, dim=1)


# ==========================================
# 6. EDRN 消融模型
# ==========================================
class EDRN_Full(nn.Module):
    def __init__(self, units=256, d_max=16):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.edrn = BiEDRNLayer(64, units, d_max)
        self.mha = nn.MultiheadAttention(2 * units, 8, batch_first=True)
        self.clf = nn.Linear(2 * units, 1)

    def forward(self, x):
        h = self.edrn(self.emb(x))
        h, _ = self.mha(h, h, h)
        return self.clf(h.max(dim=1)[0])


class EDRN_NoMHA(nn.Module):
    def __init__(self, units=256, d_max=16):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.edrn = BiEDRNLayer(64, units, d_max)
        self.clf = nn.Linear(2 * units, 1)

    def forward(self, x):
        h = self.edrn(self.emb(x))
        return self.clf(h.max(dim=1)[0])


class EDRN_Uni(nn.Module):
    def __init__(self, units=256, d_max=16):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.edrn = EDRNLayer(64, units, d_max)
        self.clf = nn.Linear(units, 1)

    def forward(self, x):
        h = self.edrn(self.emb(x))
        return self.clf(h.max(dim=1)[0])


class EDRN_NoDelay(nn.Module):
    def __init__(self, units=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.edrn = BiEDRNLayer(64, units, d_max=1)
        self.mha = nn.MultiheadAttention(2 * units, 8, batch_first=True)
        self.clf = nn.Linear(2 * units, 1)

    def forward(self, x):
        h = self.edrn(self.emb(x))
        h, _ = self.mha(h, h, h)
        return self.clf(h.max(dim=1)[0])


# ==========================================
# 7. 数据 & 指标
# ==========================================
def calculate_all_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return p, r, acc, f1, fpr, fnr


def get_data():
    df = pd.read_parquet(DATA_FILE)
    _, df_s = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df['label'])
    tr_df, ts_df = train_test_split(df_s, test_size=0.2, random_state=SEED, stratify=df_s['label'])

    def tokenize(text):
        return re.findall(r'\w+|[^\w\s]', str(text).lower())

    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            word_idx = json.load(f)
    else:
        counts = Counter()
        for t in tr_df['text'].values:
            counts.update(tokenize(t))
        word_idx = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB_SIZE - 2))}
        word_idx["<PAD>"], word_idx["<OOV>"] = 0, 1
        with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(word_idx, f, ensure_ascii=False)

    def encode(texts):
        enc = np.zeros((len(texts), MAX_LEN), dtype=np.int32)
        for i, t in enumerate(texts):
            toks = tokenize(t)[:MAX_LEN]
            enc[i, :len(toks)] = [word_idx.get(tok, 1) for tok in toks]
        return torch.from_numpy(enc)

    tr_ld = DataLoader(
        TensorDataset(
            encode(tr_df['text'].values),
            torch.tensor(tr_df['label'].values, dtype=torch.float32).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True)

    ts_ld = DataLoader(
        TensorDataset(
            encode(ts_df['text'].values),
            torch.tensor(ts_df['label'].values, dtype=torch.float32).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=False)

    return tr_ld, ts_ld


# ==========================================
# 8. 消融实验主流程
# ==========================================
def run_ablation():
    tr_ld, ts_ld = get_data()

    models = [
        ("EDRN-Full",    EDRN_Full(256, 16), True),
        ("EDRN-NoMHA",   EDRN_NoMHA(256, 16), True),
        ("EDRN-Uni",     EDRN_Uni(256, 16), True),
        ("EDRN-NoDelay", EDRN_NoDelay(256), True),
        ("EDRN-NoFGM",   EDRN_Full(256, 16), False),
    ]

    cols = ['Model','Params','Epoch','TrainLoss','ValLoss','P','R','Acc','F1','FPR','FNR']
    pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

    for name, model, use_fgm in models:
        set_seed(SEED)
        model = model.to(DEVICE)
        fgm = FGM(model)

        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sch = optim.lr_scheduler.OneCycleLR(
            opt, 2e-3, steps_per_epoch=len(tr_ld), epochs=EPOCHS)
        crit = nn.BCEWithLogitsLoss()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[START] {name} | Params: {params:,}")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            tr_loss = 0

            for x, y in tqdm(tr_ld, leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()

                loss = crit(model(x), y)
                loss.backward()

                if use_fgm:
                    fgm.attack()
                    crit(model(x), y).backward()
                    fgm.restore()

                opt.step()
                sch.step()
                tr_loss += loss.item()

            model.eval()
            ps, ls = [], []
            with torch.no_grad():
                for x, y in ts_ld:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    ps.extend((torch.sigmoid(model(x)) > 0.5).cpu().numpy().flatten())
                    ls.extend(y.cpu().numpy().flatten())

            p, r, acc, f1, fpr, fnr = calculate_all_metrics(ls, ps)
            row = [name, params, epoch, tr_loss/len(tr_ld), 0, p, r, acc, f1, fpr, fnr]
            pd.DataFrame([row]).to_csv(LOG_FILE, mode='a', header=False, index=False)

            if epoch % 10 == 0 or epoch == 1:
                print(f"[{name} E{epoch}] F1={f1:.4f} FPR={fpr:.4f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    run_ablation()
