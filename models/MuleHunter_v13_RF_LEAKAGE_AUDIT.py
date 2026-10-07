#!/usr/bin/env python
"""MuleHunter v11 - clean SAML-D ablation.
Requires MuleHunter_v10_1_patched_base.py in the same folder.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
 f1_score, roc_auc_score, average_precision_score, confusion_matrix,
 classification_report, precision_recall_curve, roc_curve)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse only the tested SAML-D data/graph utilities from v10.1.
from MuleHunter_v10_1_patched_base import (
    seed_everything, find_archive, scan_dataset, collect_train_accounts,
    build_train_graph, learn_payment_type_mapping,
    learn_markov_transition_model, extract_samples
)

SEED=42
TX_NAMES=["amount","log_amount","sender_out_count","sender_in_count",
"receiver_out_count","receiver_in_count","sender_counterparties",
"receiver_counterparties","sender_out_amount","receiver_in_amount",
"cross_bank","cross_currency","same_account","payment_type_code"]

class Encoder(nn.Module):
    def __init__(self,d,h=32,z=16):
        super().__init__(); self.a=nn.Linear(d*2,h); self.b=nn.Linear(h*2,z)
    @staticmethod
    def agg(x,s,d):
        o=torch.zeros_like(x); deg=torch.zeros(x.size(0),device=x.device)
        o.index_add_(0,d,x[s]); deg.index_add_(0,d,torch.ones(len(d),device=x.device))
        return o/deg.clamp_min(1).unsqueeze(1)
    def forward(self,x,s,d):
        n=self.agg(x,s,d); h=F.relu(self.a(torch.cat([x,n],1)))
        n2=self.agg(h,s,d); return self.b(torch.cat([h,n2],1))

class EdgeModel(nn.Module):
    def __init__(self,node_dim,extra_dim,z=16):
        super().__init__(); self.enc=Encoder(node_dim,32,z)
        self.head=nn.Sequential(nn.Linear(z*2+extra_dim,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self,x,s,d,e,extra):
        z=self.enc(x,s,d); ez=torch.cat([z[e[:,0]],z[e[:,1]]],1)
        return self.head(torch.cat([ez,extra],1)),z

def fit_graph(node,gs,gd,Xtr,ytr,etr,Xv,yv,ev,Xt,et,mode,epochs,device):
    # X contains 13 transaction features + true Markov score as final column.
    txtr,txv,txt=Xtr[:,:-1],Xv[:,:-1],Xt[:,:-1]
    mtr,mv,mt=Xtr[:,-1],Xv[:,-1],Xt[:,-1]
    sc=StandardScaler(); sc.fit(txtr)
    txtr=sc.transform(txtr).astype(np.float32); txv=sc.transform(txv).astype(np.float32); txt=sc.transform(txt).astype(np.float32)
    if mode=="graph_only":
        A=np.zeros((len(txtr),1),np.float32); B=np.zeros((len(txv),1),np.float32); C=np.zeros((len(txt),1),np.float32)
    elif mode=="graph_mlp": A,B,C=txtr,txv,txt
    else: A=np.c_[txtr,mtr]; B=np.c_[txv,mv]; C=np.c_[txt,mt]
    nx=torch.tensor(node,dtype=torch.float32,device=device); s=torch.tensor(gs,dtype=torch.long,device=device); d=torch.tensor(gd,dtype=torch.long,device=device)
    ex=torch.tensor(A,dtype=torch.float32,device=device); vx=torch.tensor(B,dtype=torch.float32,device=device); qx=torch.tensor(C,dtype=torch.float32,device=device)
    y=torch.tensor(ytr,dtype=torch.long,device=device); vy=torch.tensor(yv,dtype=torch.long,device=device)
    ee=torch.tensor(etr,dtype=torch.long,device=device); ve=torch.tensor(ev,dtype=torch.long,device=device); qe=torch.tensor(et,dtype=torch.long,device=device)
    model=EdgeModel(node.shape[1],A.shape[1]).to(device)
    c=np.bincount(ytr,minlength=2); w=torch.tensor([1.,float(c[0]/max(c[1],1))],device=device)
    loss_fn=nn.CrossEntropyLoss(weight=w); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    tl,vl=[] ,[]
    for ep in range(epochs):
        model.train(); opt.zero_grad(); lo,_=model(nx,s,d,ee,ex); loss=loss_fn(lo,y); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad(): lv,_=model(nx,s,d,ve,vx); vloss=loss_fn(lv,vy)
        tl.append(float(loss)); vl.append(float(vloss)); print(f"{mode} epoch {ep+1}/{epochs} train={loss.item():.4f} val={vloss.item():.4f}")
    with torch.no_grad():
        model.eval(); lv,z=model(nx,s,d,ve,vx); lt,_=model(nx,s,d,qe,qx)
    return torch.softmax(lv,1)[:,1].cpu().numpy(),torch.softmax(lt,1)[:,1].cpu().numpy(),tl,vl,z.cpu().numpy()

def fit_graph_custom(node, gs, gd,
                     Xtr, ytr, etr,
                     Xv, yv, ev,
                     Xt, et,
                     extra_tr, extra_v, extra_t,
                     epochs, device, label="Hybrid"):
    """
    Edge-level GraphSAGE classifier with explicitly supplied stacking inputs.

    extra_* are the exact non-graph features supplied to the MLP head.
    For the proposed model these are:
        standardized transaction features + Markov score + RF probability.
    """
    scaler = StandardScaler()
    scaler.fit(Xtr)
    txtr = scaler.transform(Xtr).astype(np.float32)
    txv = scaler.transform(Xv).astype(np.float32)
    txt = scaler.transform(Xt).astype(np.float32)

    A = np.c_[txtr, extra_tr].astype(np.float32)
    B = np.c_[txv, extra_v].astype(np.float32)
    C = np.c_[txt, extra_t].astype(np.float32)

    nx = torch.tensor(node, dtype=torch.float32, device=device)
    s = torch.tensor(gs, dtype=torch.long, device=device)
    d = torch.tensor(gd, dtype=torch.long, device=device)
    ex = torch.tensor(A, dtype=torch.float32, device=device)
    vx = torch.tensor(B, dtype=torch.float32, device=device)
    qx = torch.tensor(C, dtype=torch.float32, device=device)

    y = torch.tensor(ytr, dtype=torch.long, device=device)
    vy = torch.tensor(yv, dtype=torch.long, device=device)

    ee = torch.tensor(etr, dtype=torch.long, device=device)
    ve = torch.tensor(ev, dtype=torch.long, device=device)
    qe = torch.tensor(et, dtype=torch.long, device=device)

    model = EdgeModel(node.shape[1], A.shape[1]).to(device)

    c = np.bincount(ytr, minlength=2)
    w = torch.tensor(
        [1., float(c[0] / max(c[1], 1))],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=.001,
        weight_decay=1e-4,
    )

    tl, vl = [], []

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        lo, _ = model(nx, s, d, ee, ex)
        loss = loss_fn(lo, y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            lv, _ = model(nx, s, d, ve, vx)
            vloss = loss_fn(lv, vy)

        tl.append(float(loss.item()))
        vl.append(float(vloss.item()))

        print(
            f"{label} epoch {ep+1}/{epochs} "
            f"train={loss.item():.4f} "
            f"val={vloss.item():.4f}"
        )

    model.eval()
    with torch.no_grad():
        lv, z = model(nx, s, d, ve, vx)
        lt, _ = model(nx, s, d, qe, qx)

    pv = torch.softmax(lv, 1)[:, 1].cpu().numpy()
    pt = torch.softmax(lt, 1)[:, 1].cpu().numpy()

    return pv, pt, tl, vl, z.cpu().numpy()


def fit_rf_oof(X, y, folds=5):
    """
    Leakage-controlled RF stacking features.

    Each training row receives a probability generated by an RF that did
    not train on that row. After OOF probabilities are created, a final RF
    is fit on all training data for validation/test inference.
    """
    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(X, y), 1
    ):
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=SEED + fold,
            n_jobs=-1,
        )

        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(
            X[va_idx]
        )[:, 1]

        print(
            f"RF OOF fold {fold}/{folds} complete"
        )

    final_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    return (
        oof,
        final_model,
    )


def thr(y,p):
    pr,re,t=precision_recall_curve(y,p)
    if not len(t): return .5
    f=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12); return float(t[np.argmax(f)])

def score(name,y,p,t):
    q=(p>=t).astype(int)
    return {"Model":name,"Accuracy":accuracy_score(y,q),"Precision":precision_score(y,q,zero_division=0),"Recall":recall_score(y,q,zero_division=0),"F1":f1_score(y,q,zero_division=0),"ROC-AUC":roc_auc_score(y,p),"PR-AUC":average_precision_score(y,p)},q

def plots(out,y,probs,pred,t,tl,vl):
    out=Path(out)
    plt.figure(figsize=(7,5))
    for n,p in probs.items():
        fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr,label=f"{n} AUC={roc_auc_score(y,p):.3f}")
    plt.plot([0,1],[0,1],'--',label='Random'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('SAML-D ROC'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'roc_curve.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5))
    for n,p in probs.items():
        pr,re,_=precision_recall_curve(y,p); plt.plot(re,pr,label=f"{n} AP={average_precision_score(y,p):.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('SAML-D Precision-Recall'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'pr_curve.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(tl,label='Training Loss'); plt.plot(vl,label='Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Hybrid Loss'); plt.tight_layout(); plt.savefig(out/'training_validation_loss.png',dpi=180); plt.close()
    cm=confusion_matrix(y,pred); plt.figure(figsize=(5,4)); plt.imshow(cm); plt.title('Hybrid Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.xticks([0,1],['Normal','Laundering']); plt.yticks([0,1],['Normal','Laundering']);
    for i in range(2):
        for j in range(2): plt.text(j,i,int(cm[i,j]),ha='center',va='center')
    plt.colorbar(); plt.tight_layout(); plt.savefig(out/'confusion_matrix.png',dpi=180); plt.close()
    rows=[]
    for x in np.linspace(.01,.99,99):
        q=(probs['Hybrid']>=x).astype(int); rows.append({'threshold':x,'precision':precision_score(y,q,zero_division=0),'recall':recall_score(y,q,zero_division=0),'f1':f1_score(y,q,zero_division=0)})
    td=pd.DataFrame(rows); td.to_csv(out/'threshold_analysis.csv',index=False); plt.figure(figsize=(7,5)); plt.plot(td.threshold,td.precision,label='Precision'); plt.plot(td.threshold,td.recall,label='Recall'); plt.plot(td.threshold,td.f1,label='F1'); plt.axvline(t,ls='--',label=f'Validation threshold={t:.3f}'); plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend(); plt.tight_layout(); plt.savefig(out/'threshold_analysis.png',dpi=180); plt.close()

def dashboard(out,table):
    h=f'''<!doctype html><html><head><meta charset="utf-8"><title>MuleHunter v12</title><style>body{{font-family:Arial;background:#f4f6f8;margin:0;color:#172033}}header{{background:#172033;color:#fff;padding:25px 5%}}main{{width:92%;max-width:1450px;margin:25px auto}}.p{{background:#fff;padding:20px;margin:20px 0;border-radius:12px}}.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:center}}th{{background:#eee}}</style></head><body><header><h1>MuleHunter v12</h1><p>Leakage-controlled RF + GraphSAGE + Markov stacking</p></header><main><div class="p"><b>Protocol:</b> temporal split; train-only graph; sender/receiver GraphSAGE; validation-only threshold; held-out test.</div><div class="p"><h2>Model Comparison</h2>{table.to_html(index=False,float_format=lambda x:f"{x:.4f}")}</div><div class="g"><div class="p"><h2>Loss</h2><img src="training_validation_loss.png"></div><div class="p"><h2>ROC</h2><img src="roc_curve.png"></div><div class="p"><h2>PR</h2><img src="pr_curve.png"></div><div class="p"><h2>Confusion Matrix</h2><img src="confusion_matrix.png"></div><div class="p"><h2>Threshold</h2><img src="threshold_analysis.png"></div></div></main></body></html>'''
    p=out/'results_dashboard.html'; p.write_text(h,encoding='utf-8'); return p

def run_rf_audit(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names,
    output,
    random_state=SEED,
):
    """
    RF validity audit.

    This does NOT delete or hide the strong RF result. It tests whether the
    performance is driven by suspiciously target-proxy features or train/test
    distribution shifts.

    Outputs:
      - rf_feature_importance.csv
      - rf_permutation_importance.csv
      - rf_single_feature_auc.csv
      - feature_distribution_audit.csv
      - rf_audit_summary.txt
      - rf_audit.json
    """
    audit_dir = output / "rf_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    Xtr = np.asarray(X_train, dtype=np.float32)
    Xv = np.asarray(X_val, dtype=np.float32)
    Xt = np.asarray(X_test, dtype=np.float32)

    names = list(feature_names)
    if len(names) != Xtr.shape[1]:
        names = [f"feature_{i}" for i in range(Xtr.shape[1])]

    # 1. Train a clean RF using training features only.
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(Xtr, y_train)

    pv = rf.predict_proba(Xv)[:, 1]
    pt = rf.predict_proba(Xt)[:, 1]

    # 2. Standard metrics at validation-selected threshold.
    threshold = thr(y_val, pv)
    pred = (pt >= threshold).astype(int)

    auc = float(roc_auc_score(y_test, pt))
    pr = float(average_precision_score(y_test, pt))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    prec = float(precision_score(y_test, pred, zero_division=0))
    rec = float(recall_score(y_test, pred, zero_division=0))
    acc = float(accuracy_score(y_test, pred))

    # 3. RF feature importance.
    fi = pd.DataFrame({
        "feature": names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fi.to_csv(
        audit_dir / "rf_feature_importance.csv",
        index=False,
    )

    # 4. Permutation importance on held-out validation data.
    # This is intentionally validation-only, not test-set tuning.
    perm = permutation_importance(
        rf,
        Xv,
        y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )

    pi = pd.DataFrame({
        "feature": names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(
        "importance_mean",
        ascending=False,
    )

    pi.to_csv(
        audit_dir / "rf_permutation_importance.csv",
        index=False,
    )

    # 5. Single-feature AUC audit.
    # A feature with near-perfect AUC is a strong target-proxy warning.
    rows = []
    for j, name in enumerate(names):
        try:
            if np.unique(Xtr[:, j]).size < 2:
                a = np.nan
            else:
                a = roc_auc_score(y_train, Xtr[:, j])
                a = max(float(a), 1.0 - float(a))
        except Exception:
            a = np.nan

        rows.append({
            "feature": name,
            "single_feature_auc_abs": a,
        })

    single = pd.DataFrame(rows).sort_values(
        "single_feature_auc_abs",
        ascending=False,
    )
    single.to_csv(
        audit_dir / "rf_single_feature_auc.csv",
        index=False,
    )

    # 6. Train-vs-validation-vs-test distribution audit.
    rows = []
    for j, name in enumerate(names):
        tr = Xtr[:, j]
        va = Xv[:, j]
        te = Xt[:, j]

        # KS tests are descriptive here, not a hypothesis-test basis for
        # changing the model.
        try:
            ks_tv = ks_2samp(tr, va)
            ks_tt = ks_2samp(tr, te)
            ks_tv_stat, ks_tv_p = float(ks_tv.statistic), float(ks_tv.pvalue)
            ks_tt_stat, ks_tt_p = float(ks_tt.statistic), float(ks_tt.pvalue)
        except Exception:
            ks_tv_stat = ks_tv_p = ks_tt_stat = ks_tt_p = np.nan

        rows.append({
            "feature": name,
            "train_mean": float(np.mean(tr)),
            "validation_mean": float(np.mean(va)),
            "test_mean": float(np.mean(te)),
            "train_std": float(np.std(tr)),
            "validation_std": float(np.std(va)),
            "test_std": float(np.std(te)),
            "ks_train_validation": ks_tv_stat,
            "ks_p_train_validation": ks_tv_p,
            "ks_train_test": ks_tt_stat,
            "ks_p_train_test": ks_tt_p,
        })

    dist = pd.DataFrame(rows).sort_values(
        "ks_train_test",
        ascending=False,
    )
    dist.to_csv(
        audit_dir / "feature_distribution_audit.csv",
        index=False,
    )

    # 7. Correlation/proxy screen using training labels only.
    proxy_rows = []
    y_float = np.asarray(y_train, dtype=np.float64)
    for j, name in enumerate(names):
        x = np.asarray(Xtr[:, j], dtype=np.float64)
        if np.std(x) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x, y_float)[0, 1])
        proxy_rows.append({
            "feature": name,
            "pearson_target_correlation": corr,
            "absolute_correlation": abs(corr) if np.isfinite(corr) else np.nan,
        })

    proxy = pd.DataFrame(proxy_rows).sort_values(
        "absolute_correlation",
        ascending=False,
    )
    proxy.to_csv(
        audit_dir / "feature_target_correlation.csv",
        index=False,
    )

    # 8. Summary flags.
    top_single = single.iloc[0].to_dict() if len(single) else {}
    top_importance = fi.iloc[0].to_dict() if len(fi) else {}
    top_perm = pi.iloc[0].to_dict() if len(pi) else {}

    warnings = []

    if (
        np.isfinite(top_single.get("single_feature_auc_abs", np.nan))
        and top_single["single_feature_auc_abs"] >= 0.98
    ):
        warnings.append(
            "At least one individual feature has single-feature AUC >= 0.98; "
            "inspect it manually for target/proxy leakage."
        )

    if (
        np.isfinite(top_importance.get("importance", np.nan))
        and top_importance["importance"] >= 0.50
    ):
        warnings.append(
            "One RF feature contributes >=50% of impurity importance; "
            "inspect that feature for proxy leakage."
        )

    if (
        np.isfinite(top_perm.get("importance_mean", np.nan))
        and top_perm["importance_mean"] >= 0.50
    ):
        warnings.append(
            "One feature has unusually large permutation importance; "
            "inspect its construction and temporal availability."
        )

    summary = {
        "rf_test_accuracy": acc,
        "rf_test_precision": prec,
        "rf_test_recall": rec,
        "rf_test_f1": f1,
        "rf_test_roc_auc": auc,
        "rf_test_pr_auc": pr,
        "validation_threshold": float(threshold),
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "positive_train": int(np.sum(y_train)),
        "positive_validation": int(np.sum(y_val)),
        "positive_test": int(np.sum(y_test)),
        "top_single_feature": top_single,
        "top_impurity_feature": top_importance,
        "top_permutation_feature": top_perm,
        "warnings": warnings,
    }

    with open(audit_dir / "rf_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(
        audit_dir / "rf_audit_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("MULEHUNTER v13 — RF VALIDITY AUDIT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy : {acc:.6f}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Recall   : {rec:.6f}\n")
        f.write(f"F1       : {f1:.6f}\n")
        f.write(f"ROC-AUC  : {auc:.6f}\n")
        f.write(f"PR-AUC   : {pr:.6f}\n")
        f.write(f"Threshold: {threshold:.6f}\n\n")

        f.write("Top single-feature AUC:\n")
        f.write(str(top_single) + "\n\n")

        f.write("Top RF impurity importance:\n")
        f.write(str(top_importance) + "\n\n")

        f.write("Top permutation importance:\n")
        f.write(str(top_perm) + "\n\n")

        f.write("WARNINGS:\n")
        if warnings:
            for w in warnings:
                f.write("- " + w + "\n")
        else:
            f.write("- No automatic red flags triggered.\n")

    print("\nRF AUDIT")
    print("-" * 60)
    print(f"RF test ROC-AUC: {auc:.6f}")
    print(f"RF test PR-AUC : {pr:.6f}")
    print(f"RF test F1     : {f1:.6f}")
    print(
        "Top single-feature AUC:",
        f"{top_single.get('single_feature_auc_abs', np.nan):.6f}",
        top_single.get("feature", "N/A"),
    )

    if warnings:
        print("\nAUDIT WARNINGS:")
        for w in warnings:
            print(" -", w)
    else:
        print("\nNo automatic red flags triggered.")

    print(
        "Audit files:",
        audit_dir.resolve(),
    )

    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='archive.zip'); ap.add_argument('--output',default='results_v11'); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--chunk-size',type=int,default=250000); ap.add_argument('--max-train-edges',type=int,default=1000000); ap.add_argument('--max-samples',type=int,default=20000); a=ap.parse_args()
    seed_everything(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); archive=find_archive(a.data); start=time.perf_counter()
    print('='*72); print('MULEHUNTER v11'); print('CLEAN SAML-D ABLATION'); print('='*72)
    print('\n[1/7] Scanning dataset...'); info=scan_dataset(archive,a.chunk_size); print(f"Transactions: {info['rows']:,}"); print(f"Laundering transactions: {info['positives']:,}"); print(f"Date range: {info['min_time']} -> {info['max_time']}"); print(f"Train cutoff: {info['train_cutoff']}"); print(f"Validation cutoff: {info['val_cutoff']}")
    print('\n[2/7] Building train-only graph...'); accounts=collect_train_accounts(archive,info['train_cutoff'],a.chunk_size); amap={x:i for i,x in enumerate(accounts)}; node,gs,gd,observed=build_train_graph(archive,info['train_cutoff'],amap,a.chunk_size,a.max_train_edges); print(f"Graph nodes: {len(accounts):,}"); print(f"Training edges observed: {observed:,}"); print(f"Graph edges retained: {len(gs):,}")
    print('\n[3/7] Learning Markov and extracting samples...'); pmap=learn_payment_type_mapping(archive,info['train_cutoff'],a.chunk_size); q1,q2,tr=learn_markov_transition_model(archive,info['train_cutoff'],amap,a.chunk_size); print(f"Markov amount states: {q1:.6f}, {q2:.6f}"); train,val,test=extract_samples(archive,info['train_cutoff'],info['val_cutoff'],amap,a.chunk_size,a.max_samples,pmap,q1,q2,tr)
    Xtr,ytr,etr=train; Xv,yv,ev=val; Xt,yt,et=test; print(f"Train: {len(ytr):,} (laundering={int(ytr.sum()):,})"); print(f"Validation: {len(yv):,} (laundering={int(yv.sum()):,})"); print(f"Test: {len(yt):,} (laundering={int(yt.sum()):,})")
    # X from v10.1 has 13 transaction features + true chronological Markov score.
    if Xtr.shape[1] < 15: raise RuntimeError(f'Expected 15 columns from extractor (14 transaction features + Markov), got {Xtr.shape[1]}. Use matching v10.1 base.')
    print('\n[4/7] RF baselines + leakage-controlled stacking features...')

    # v13: audit the near-perfect RF before interpreting it as a publishable
    # result. The audit is based only on features available to the model.
    run_rf_audit(
        Xtr,
        ytr,
        Xv,
        yv,
        Xt,
        yt,
        TX_NAMES,
        out,
    )


    # The final feature in v10.1 is the chronological Markov score.
    mtr = Xtr[:, -1].astype(np.float32)
    mv = Xv[:, -1].astype(np.float32)
    mt = Xt[:, -1].astype(np.float32)

    txtr = Xtr[:, :-1]
    txv = Xv[:, :-1]
    txt = Xt[:, :-1]

    sc = StandardScaler()
    sc.fit(txtr)

    A = sc.transform(txtr).astype(np.float32)
    B = sc.transform(txv).astype(np.float32)
    C = sc.transform(txt).astype(np.float32)

    # Markov-only baseline.
    mtv = thr(yv, mv)
    mr, mp = score(
        'Markov Only',
        yt,
        mt,
        mtv,
    )

    # RF-only baseline.
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(A, ytr)

    rv = rf.predict_proba(B)[:, 1]
    rt = rf.predict_proba(C)[:, 1]

    rtth = thr(yv, rv)
    rr, rp = score(
        'RF Only',
        yt,
        rt,
        rtth,
    )

    # Markov + RF baseline.
    rfm = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=SEED,
        n_jobs=-1,
    )
    rfm.fit(
        np.c_[A, mtr],
        ytr,
    )

    rfv = rfm.predict_proba(
        np.c_[B, mv]
    )[:, 1]

    rft = rfm.predict_proba(
        np.c_[C, mt]
    )[:, 1]

    rfth = thr(yv, rfv)
    rfr, rfp = score(
        'Markov + RF',
        yt,
        rft,
        rfth,
    )

    # Leakage-controlled RF probabilities for stacking.
    print('\nGenerating out-of-fold RF probabilities for Hybrid...')
    rf_oof, rf_stack = fit_rf_oof(
        np.c_[A, mtr],
        ytr,
        folds=5,
    )

    rf_stack_val = rf_stack.predict_proba(
        np.c_[B, mv]
    )[:, 1].astype(np.float32)

    rf_stack_test = rf_stack.predict_proba(
        np.c_[C, mt]
    )[:, 1].astype(np.float32)

    # Sanity check: OOF predictions must not be perfect training predictions.
    print(
        "RF OOF mean probability:",
        f"{rf_oof.mean():.6f}"
    )

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print('\n[5/7] GraphSAGE ablations...')
    print('Device:', device)

    # GraphSAGE Only: graph structure only.
    zero_tx_tr = np.zeros(
        (len(Xtr), 1),
        dtype=np.float32,
    )
    zero_tx_v = np.zeros(
        (len(Xv), 1),
        dtype=np.float32,
    )
    zero_tx_t = np.zeros(
        (len(Xt), 1),
        dtype=np.float32,
    )

    sv, st, _, _, _ = fit_graph_custom(
        node, gs, gd,
        zero_tx_tr, ytr, etr,
        zero_tx_v, yv, ev,
        zero_tx_t, et,
        np.zeros((len(Xtr), 0), dtype=np.float32),
        np.zeros((len(Xv), 0), dtype=np.float32),
        np.zeros((len(Xt), 0), dtype=np.float32),
        a.epochs,
        device,
        label='GraphSAGE Only',
    )

    sth = thr(yv, sv)
    sr, sp = score(
        'GraphSAGE Only',
        yt,
        st,
        sth,
    )

    # GraphSAGE + MLP: graph embeddings + transaction features.
    gv, gt, _, _, _ = fit_graph(
        node, gs, gd,
        Xtr, ytr, etr,
        Xv, yv, ev,
        Xt, et,
        'graph_mlp',
        a.epochs,
        device,
    )

    gth = thr(yv, gv)
    gr, gp = score(
        'GraphSAGE + MLP',
        yt,
        gt,
        gth,
    )

    print('\n[6/7] Training stacked RF + GraphSAGE models...')

    # RF + GraphSAGE + MLP:
    # graph embeddings + transaction features + RF probability.
    rg_v, rg_t, rg_tl, rg_vl, _ = fit_graph_custom(
        node, gs, gd,
        txtr, ytr, etr,
        txv, yv, ev,
        txt, et,
        rf_oof.reshape(-1, 1),
        rf_stack_val.reshape(-1, 1),
        rf_stack_test.reshape(-1, 1),
        a.epochs,
        device,
        label='RF + GraphSAGE + MLP',
    )

    rgth = thr(yv, rg_v)
    rgr, rgp = score(
        'RF + GraphSAGE + MLP',
        yt,
        rg_t,
        rgth,
    )

    # Proposed Hybrid:
    # graph embeddings + transaction features + Markov + RF probability.
    hv, ht, tl, vl, z = fit_graph_custom(
        node, gs, gd,
        txtr, ytr, etr,
        txv, yv, ev,
        txt, et,
        np.c_[mtr, rf_oof],
        np.c_[mv, rf_stack_val],
        np.c_[mt, rf_stack_test],
        a.epochs,
        device,
        label='Hybrid Markov + RF + GraphSAGE',
    )

    hth = thr(yv, hv)
    hr, hp = score(
        'Hybrid Markov + RF + GraphSAGE',
        yt,
        ht,
        hth,
    )

    table = pd.DataFrame([
        mr,
        rr,
        rfr,
        sr,
        gr,
        rgr,
        hr,
    ])

    table.to_csv(
        out / 'metrics_comparison.csv',
        index=False,
    )

    print('\nMODEL COMPARISON')
    print(table.to_string(index=False))

    probs = {
        'Markov Only': mt,
        'RF Only': rt,
        'Markov + RF': rft,
        'GraphSAGE Only': st,
        'GraphSAGE + MLP': gt,
        'RF + GraphSAGE + MLP': rg_t,
        'Hybrid': ht,
    }

    plots(
        out,
        yt,
        probs,
        hp,
        hth,
        tl,
        vl,
    )

    pred = {
        'Markov Only': mp,
        'RF Only': rp,
        'Markov + RF': rfp,
        'GraphSAGE Only': sp,
        'GraphSAGE + MLP': gp,
        'RF + GraphSAGE + MLP': rgp,
        'Hybrid': hp,
    }

    th = {
        'Markov Only': mtv,
        'RF Only': rtth,
        'Markov + RF': rfth,
        'GraphSAGE Only': sth,
        'GraphSAGE + MLP': gth,
        'RF + GraphSAGE + MLP': rgth,
        'Hybrid': hth,
    }

    reps=[]
    for n,p in pred.items(): reps.append('='*60+f'\n{n}\nValidation threshold: {th[n]:.6f}\n\n'+classification_report(yt,p,digits=4,zero_division=0))
    (out/'classification_reports.txt').write_text('\n'.join(reps),encoding='utf-8')
    fi=pd.DataFrame({'feature':TX_NAMES,'importance':rf.feature_importances_}).sort_values('importance',ascending=False); fi.to_csv(out/'feature_importance.csv',index=False); plt.figure(figsize=(8,6)); top=fi.head(15).sort_values('importance'); plt.barh(top.feature,top.importance); plt.xlabel('Importance'); plt.title('RF-Only Feature Importance'); plt.tight_layout(); plt.savefig(out/'feature_importance.png',dpi=180); plt.close()
    profile={'version':'v13',
    'audit':'rf_audit/rf_audit_summary.txt','dataset':'SAML-D','target':'Is_laundering','transactions':info['rows'],'laundering_transactions':info['positives'],'train_cutoff':str(info['train_cutoff']),'validation_cutoff':str(info['val_cutoff']),'graph_nodes':len(accounts),'training_edges_observed':observed,'training_edges_retained':len(gs),'train_samples':len(ytr),'validation_samples':len(yv),'test_samples':len(yt),'test_positive_rate':float(yt.mean()),'device':str(device),'epochs':a.epochs,'runtime_seconds':time.perf_counter()-start,'protocol':'chronological 70/15/15; train-only graph; validation-only threshold; held-out test; RF stacking uses 5-fold OOF training probabilities',
    'hybrid_definition':'sender/receiver GraphSAGE embeddings + transaction features + Markov score + leakage-controlled RF probability',
    'rf_stack_definition':'RF probability is OOF for training rows and full-training RF probability for validation/test rows','models':['Markov Only','RF Only','Markov + RF','GraphSAGE Only','GraphSAGE + MLP','RF + GraphSAGE + MLP','Hybrid Markov + RF + GraphSAGE']}
    (out/'profiling.json').write_text(json.dumps(profile,indent=2),encoding='utf-8'); d=dashboard(out,table); print('\n'+'='*72); print('V12 COMPLETE'); print('='*72); print('Results:',out.resolve()); print('Dashboard:',d.resolve())

if __name__=='__main__': main()
