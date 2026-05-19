import os, re, torch, gc, random, time, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. 环境与核心配置
# ==========================================
SEED, MAX_LEN, VOCAB_SIZE = 42, 128, 25000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = r'D:\INJECT\data30w\combined_dataset.parquet'
SAVE_DIR = r'D:\INJECT\XIAORONG_MHA_TUNING'
LOG_FILE = os.path.join(SAVE_DIR, "tuning_results_live.csv")
# [NEW] 定义模型断点保存路径
CKPT_PATH = os.path.join(SAVE_DIR, "current_task_checkpoint.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

U_SPACE = [32, 64, 128, 256]
D_SPACE = [2, 4, 8, 16]
BATCH_SIZE = 512
EPOCHS = 50


def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed);
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)


# ==========================================
# 2. 核心算法组件 (保持一致)
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
    def __init__(self, units, d_max):
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
# 3. 数据预处理与评价逻辑
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

    counts = Counter()
    for t in tr_df['text'].values: counts.update(tokenize(t))
    word_idx = {w: i + 2 for i, (w, c) in enumerate(counts.most_common(VOCAB_SIZE - 2))}
    word_idx["<PAD>"], word_idx["<OOV>"] = 0, 1

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
# 4. 实验运行逻辑 (Grid Search + Resumption)
# ==========================================
def run_tuning():
    tr_ld, ts_ld = get_data()
    cols = ['U', 'D', 'epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'accuracy', 'f1', 'fpr', 'fnr']

    # [NEW] 初始化或读取现有进度
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)
        completed_tasks = set()
    else:
        log_df = pd.read_csv(LOG_FILE)
        # 只有完成全部 EPOCHS 的才算完成
        task_counts = log_df.groupby(['U', 'D']).size().reset_index(name='count')
        completed_tasks = set(zip(task_counts[task_counts['count'] >= EPOCHS]['U'],
                                  task_counts[task_counts['count'] >= EPOCHS]['D']))

    for u in U_SPACE:
        for d in D_SPACE:
            if (u, d) in completed_tasks:
                print(f"⏩ 跳过已完成组: U={u}, D={d}")
                continue

            print(f"\n🚀 启动/恢复试验组: U={u}, D={d}")
            model = EDRN_MHA_Tuned(u, d).to(DEVICE)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            sch = optim.lr_scheduler.OneCycleLR(opt, 2e-3, steps_per_epoch=len(tr_ld), epochs=EPOCHS)
            crit = nn.BCEWithLogitsLoss()
            fgm = FGM(model)

            # [NEW] 检查是否有该任务的中途断点
            start_epoch = 1
            if os.path.exists(CKPT_PATH):
                ckpt = torch.load(CKPT_PATH)
                if ckpt['U'] == u and ckpt['D'] == d:
                    model.load_state_dict(ckpt['model_state'])
                    opt.load_state_dict(ckpt['optimizer_state'])
                    sch.load_state_dict(ckpt['scheduler_state'])
                    start_epoch = ckpt['epoch'] + 1
                    print(f"🔄 检测到中途断点，从 Epoch {start_epoch} 恢复")
                else:
                    os.remove(CKPT_PATH)  # 非当前任务的旧断点直接删除

            for epoch in range(start_epoch, EPOCHS + 1):
                model.train()
                tr_loss, pbar = 0, tqdm(tr_ld, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
                for x, y in pbar:
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
                    tr_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

                model.eval()
                val_loss, ps, ls = 0, [], []
                with torch.no_grad():
                    for x, y in ts_ld:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        out = model(x)
                        val_loss += crit(out, y).item()
                        ps.extend(torch.sigmoid(out).cpu().numpy().flatten() > 0.5)
                        ls.extend(y.cpu().numpy().flatten())

                p, r, acc, f1, fpr, fnr = calculate_all_metrics(ls, ps)
                metrics_row = [u, d, epoch, tr_loss / len(tr_ld), val_loss / len(ts_ld), p, r, acc, f1, fpr, fnr]
                pd.DataFrame([metrics_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)

                # [NEW] 每轮保存断点
                torch.save({
                    'U': u, 'D': d, 'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': opt.state_dict(),
                    'scheduler_state': sch.state_dict()
                }, CKPT_PATH)

                if epoch % 10 == 0 or epoch == 1:
                    print(f"[E{epoch}] F1: {f1:.4f} | FNR: {fnr:.4f} | Loss: {val_loss / len(ts_ld):.4f}")

            # [NEW] 任务组完成，删除当前断点文件
            if os.path.exists(CKPT_PATH): os.remove(CKPT_PATH)
            del model;
            gc.collect();
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_tuning()