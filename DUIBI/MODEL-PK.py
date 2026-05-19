import os

# [核心] 必须在 import torch 前设置，强制 cuBLAS 使用固定工作空间
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import re, torch, gc, random, numpy as np, pandas as pd, json
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. 环境与核心配置 (严格复现版)
# ==========================================
SEED, MAX_LEN, VOCAB_SIZE = 42, 128, 25000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 请根据实际情况确认以下路径 ---
DATA_FILE = r'D:\INJECT\data30w\combined_dataset.parquet'
SAVE_DIR = r'D:\INJECT\XIAORONG_CHULI_UD'
LOG_FILE = os.path.join(SAVE_DIR, "baseline_comparison_results.csv")
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab_fixed.json")

os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 512
EPOCHS = 50


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 强制所有算子使用确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)


# ==========================================
# 2. 核心算法组件
# ==========================================
class FGM:
    def __init__(self, model):
        self.model, self.backup = model, {}

    def attack(self, epsilon=0.3, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: param.data = self.backup[name]
        self.backup = {}


@torch.jit.script
def edrn_step(gates_t, pd_t, alpha_prev, units: int, d_max: int):
    z, r, g, o = torch.split(gates_t, units, dim=-1)
    z, r, o = torch.sigmoid(z), torch.sigmoid(r), torch.sigmoid(o)
    p_d = torch.softmax(pd_t.view(-1, units, d_max), dim=-1)
    alpha_shift = torch.zeros_like(alpha_prev)
    alpha_shift[:, :, :-1] = alpha_prev[:, :, 1:]
    alpha_new = (z.unsqueeze(-1) * alpha_shift) + (r.unsqueeze(-1) * torch.tanh(g).unsqueeze(-1) * p_d)
    return alpha_new, o * torch.tanh(torch.sum(alpha_new, dim=-1))


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
        fm, fa = torch.zeros(b, self.units, device=x.device), torch.zeros(b, self.units, self.d_max, device=x.device)
        fx = self.f_proj(x)
        f_outs = []
        for t in range(s):
            gt, pt = torch.split(fx[:, t, :] + self.f_m_proj(fm), [4 * self.units, self.units * self.d_max], dim=-1)
            fa, fm = edrn_step(gt, pt, fa, self.units, self.d_max)
            f_outs.append(fm.unsqueeze(1))
        bm, ba = torch.zeros(b, self.units, device=x.device), torch.zeros(b, self.units, self.d_max, device=x.device)
        bx = self.b_proj(torch.flip(x, [1]))
        b_outs = []
        for t in range(s):
            gt, pt = torch.split(bx[:, t, :] + self.b_m_proj(bm), [4 * self.units, self.units * self.d_max], dim=-1)
            ba, bm = edrn_step(gt, pt, ba, self.units, self.d_max)
            b_outs.append(bm.unsqueeze(1))
        return torch.cat([torch.cat(f_outs, 1), torch.flip(torch.cat(b_outs, 1), [1])], dim=-1)


class EDRN_MHA_Tuned(nn.Module):
    def __init__(self, units=256, d_max=16):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.edrn = BiEDRNLayer(64, units, d_max)
        self.mha = nn.MultiheadAttention(2 * units, 8, batch_first=True)
        self.clf = nn.Linear(2 * units, 1)

    def forward(self, x):
        h = self.edrn(self.emb(x))
        out, _ = self.mha(h, h, h)
        return self.clf(out.max(dim=1)[0])


# ==========================================
# 3. 基线模型
# ==========================================
class PLSTMModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.clf = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.clf(h[-1])


class LSTMModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.clf = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.clf(h[-1])


class BiLSTMModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.clf = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.clf(torch.cat([h[-2], h[-1]], dim=-1))


class BiGRUModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
        self.clf = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.clf(torch.cat([h[-2], h[-1]], dim=-1))


class TransformerModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=hidden, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.clf = nn.Linear(64, 1)

    def forward(self, x):
        x = self.transformer(self.emb(x))
        return self.clf(x.max(dim=1)[0])


class CNNModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.convs = nn.ModuleList([nn.Conv1d(64, hidden // 3, k) for k in [3, 4, 5]])
        self.clf = nn.Linear((hidden // 3) * 3, 1)

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return self.clf(torch.cat(x, 1))


# ==========================================
# 4. 数据与指标
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
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到数据集文件: {DATA_FILE}")

    df = pd.read_parquet(DATA_FILE)
    _, df_s = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df['label'])
    tr_df, ts_df = train_test_split(df_s, test_size=0.2, random_state=SEED, stratify=df_s['label'])

    def tokenize(text):
        return re.findall(r'\w+|[^\w\s]', str(text).lower())

    # --- 严格词表加载 ---
    if os.path.exists(VOCAB_PATH):
        print(f"📦 加载既有词表: {VOCAB_PATH}")
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            word_idx = json.load(f)
    else:
        print("⚠️ 词表不存在，正在重新生成...")
        counts = Counter()
        for t in tr_df['text'].values: counts.update(tokenize(t))
        word_idx = {w: i + 2 for i, (w, c) in enumerate(counts.most_common(VOCAB_SIZE - 2))}
        word_idx["<PAD>"], word_idx["<OOV>"] = 0, 1
        with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(word_idx, f, ensure_ascii=False)

    def encode(texts):
        enc = np.zeros((len(texts), MAX_LEN), dtype=np.int32)
        for i, t in enumerate(texts):
            tokens = tokenize(str(t))[:MAX_LEN]
            enc[i, :len(tokens)] = [word_idx.get(tk, 1) for tk in tokens]
        return torch.from_numpy(enc)

    tr_ld = DataLoader(TensorDataset(encode(tr_df['text'].values),
                                     torch.tensor(tr_df['label'].values, dtype=torch.float32).unsqueeze(1)),
                       batch_size=BATCH_SIZE, shuffle=True)
    ts_ld = DataLoader(TensorDataset(encode(ts_df['text'].values),
                                     torch.tensor(ts_df['label'].values, dtype=torch.float32).unsqueeze(1)),
                       batch_size=BATCH_SIZE, shuffle=False)
    return tr_ld, ts_ld


# ==========================================
# 5. 执行对比
# ==========================================
def run_comparison():
    # 彻底清除可能存在的旧 Checkpoint 干扰
    old_ckpt = os.path.join(SAVE_DIR, "current_task_checkpoint.pth")
    if os.path.exists(old_ckpt):
        os.remove(old_ckpt)
        print(f"🧹 已清理旧断点文件以确保从零开始训练。")

    tr_ld, ts_ld = get_data()

    # 按照特定顺序初始化模型
    models_to_compare = [
        ("EDRN_MHA_U256_D16", EDRN_MHA_Tuned(256, 16)),
        ("PLSTM_Base", PLSTMModel(512)),
        ("LSTM_Base", LSTMModel(512)),
        ("BiLSTM_Base", BiLSTMModel(512)),
        ("BiGRU_Base", BiGRUModel(512)),
        ("Transformer_Base", TransformerModel(512)),
        ("CNN_Base", CNNModel(512))
    ]

    cols = ['Model', 'Params', 'Epoch', 'TrainLoss', 'ValLoss', 'P', 'R', 'Acc', 'F1', 'FPR', 'FNR']
    # 如果已存在对比日志，备份并新建
    if os.path.exists(LOG_FILE):
        os.rename(LOG_FILE, LOG_FILE + ".bak")

    pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

    for name, model in models_to_compare:
        set_seed(SEED)  # 每一个模型训练前，强制重置所有随机数状态
        model = model.to(DEVICE)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[START] {name} | Params: {params:,}")

        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sch = optim.lr_scheduler.OneCycleLR(opt, 2e-3, steps_per_epoch=len(tr_ld), epochs=EPOCHS)
        crit = nn.BCEWithLogitsLoss()
        fgm = FGM(model)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            tr_l = 0
            for x, y in tqdm(tr_ld, desc=f"{name} E{epoch}", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                fgm.attack()
                crit(model(x), y).backward()
                fgm.restore()
                opt.step()
                sch.step()
                tr_l += loss.item()

            model.eval()
            ps, ls = [], []
            with torch.no_grad():
                for x, y in ts_ld:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x)
                    ps.extend(torch.sigmoid(out).cpu().numpy().flatten() > 0.5)
                    ls.extend(y.cpu().numpy().flatten())

            p, r, acc, f1, fpr, fnr = calculate_all_metrics(ls, ps)
            res = [name, params, epoch, tr_l / len(tr_ld), 0, p, r, acc, f1, fpr, fnr]
            pd.DataFrame([res]).to_csv(LOG_FILE, mode='a', header=False, index=False)

            if epoch % 10 == 0 or epoch == 1:
                print(f"[{name} E{epoch}] F1: {f1:.4f} | FPR: {fpr:.4f}")

        del model, opt, sch, fgm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_comparison()