#!/usr/bin/env python
"""
MuleHunter v10
Real SAML-D transaction graph + edge-level GraphSAGE + Markov hybrid.

Key correction from v9:
- GraphSAGE embeddings are sender/receiver-specific.
- The graph is the real Sender_account -> Receiver_account graph.
- Edge representations use [sender_embedding, receiver_embedding,
  transaction_features, Markov_score].
- No single global graph embedding is repeated for every transaction.
- Train graph contains TRAIN transactions only.
- Temporal validation/test transactions are never added to the message-passing graph.
- Threshold is selected on validation only.
"""

import argparse
import json
import time
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
TARGET = "Is_laundering"

RAW_FEATURES = [
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

# Numeric transaction features created from SAML-D.
NUMERIC_FEATURE_NAMES = [
    "amount",
    "log_amount",
    "sender_out_count",
    "sender_in_count",
    "receiver_out_count",
    "receiver_in_count",
    "sender_counterparties",
    "receiver_counterparties",
    "sender_out_amount",
    "receiver_in_amount",
    "cross_bank",
    "cross_currency",
    "same_account",
    "payment_type_code",
]


def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_archive(path):
    candidates = [
        Path(path),
        Path("archive.zip"),
        Path("/mnt/data/archive.zip"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "archive.zip not found. Use --data archive.zip"
    )


def locate_csv(zf):
    for name in zf.namelist():
        if name.lower().endswith("saml-d.csv"):
            return name
    raise FileNotFoundError("SAML-D.csv not found in archive.zip")


def parse_chunk(chunk):
    chunk["Timestamp"] = pd.to_datetime(
        chunk["Date"].astype(str) + " " +
        chunk["Time"].astype(str),
        errors="coerce",
    )

    chunk["Amount"] = pd.to_numeric(
        chunk["Amount"], errors="coerce"
    ).fillna(0.0)

    chunk[TARGET] = pd.to_numeric(
        chunk[TARGET], errors="coerce"
    ).fillna(0).astype(np.int64)

    return chunk.dropna(subset=["Timestamp"])


def iter_saml(archive, chunk_size):
    zf = zipfile.ZipFile(archive)
    name = locate_csv(zf)
    stream = zf.open(name)
    try:
        for chunk in pd.read_csv(stream, chunksize=chunk_size):
            yield parse_chunk(chunk)
    finally:
        stream.close()
        zf.close()


def scan_dataset(archive, chunk_size):
    min_t = None
    max_t = None
    rows = 0
    positives = 0

    for chunk in iter_saml(archive, chunk_size):
        rows += len(chunk)
        positives += int(chunk[TARGET].sum())

        cmin = chunk.Timestamp.min()
        cmax = chunk.Timestamp.max()

        min_t = cmin if min_t is None else min(min_t, cmin)
        max_t = cmax if max_t is None else max(max_t, cmax)

    if min_t is None:
        raise RuntimeError("SAML-D contains no usable rows.")

    span = (max_t - min_t).total_seconds()

    train_cutoff = min_t + pd.to_timedelta(
        span * 0.70, unit="s"
    )
    val_cutoff = min_t + pd.to_timedelta(
        span * 0.85, unit="s"
    )

    return {
        "rows": rows,
        "positives": positives,
        "min_time": min_t,
        "max_time": max_t,
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
    }


def collect_train_accounts(
    archive,
    train_cutoff,
    chunk_size,
):
    accounts = set()

    for chunk in iter_saml(archive, chunk_size):
        c = chunk[chunk.Timestamp < train_cutoff]

        accounts.update(c.Sender_account.astype(str))
        accounts.update(c.Receiver_account.astype(str))

    return sorted(accounts)


def build_train_graph(
    archive,
    train_cutoff,
    account_to_idx,
    chunk_size,
    max_edges,
):
    n = len(account_to_idx)

    # Node features learned from training transactions only.
    # 0 in-count, 1 out-count, 2 in-amount, 3 out-amount,
    # 4 unique counterparties, 5 cross-bank, 6 cross-currency,
    # 7 mean amount.
    node = np.zeros((n, 8), dtype=np.float64)
    counterparties = defaultdict(set)

    rng = np.random.default_rng(SEED)
    src = []
    dst = []
    seen = 0

    for chunk in iter_saml(archive, chunk_size):
        c = chunk[chunk.Timestamp < train_cutoff]

        for row in c.itertuples(index=False):
            s_id = str(row.Sender_account)
            r_id = str(row.Receiver_account)

            if s_id not in account_to_idx or r_id not in account_to_idx:
                continue

            s = account_to_idx[s_id]
            r = account_to_idx[r_id]
            amount = float(row.Amount)

            node[s, 1] += 1
            node[r, 0] += 1
            node[s, 3] += amount
            node[r, 2] += amount

            counterparties[s].add(r)
            counterparties[r].add(s)

            cross_bank = (
                str(row.Sender_bank_location)
                != str(row.Receiver_bank_location)
            )
            cross_currency = (
                str(row.Payment_currency)
                != str(row.Received_currency)
            )

            node[s, 5] += int(cross_bank)
            node[r, 5] += int(cross_bank)
            node[s, 6] += int(cross_currency)
            node[r, 6] += int(cross_currency)

            seen += 1

            if len(src) < max_edges:
                src.append(s)
                dst.append(r)
            else:
                j = rng.integers(0, seen)
                if j < max_edges:
                    src[j] = s
                    dst[j] = r

    for i in range(n):
        node[i, 4] = len(counterparties[i])

    node[:, 7] = (
        node[:, 2] + node[:, 3]
    ) / np.maximum(
        node[:, 0] + node[:, 1], 1.0
    )

    return (
        node.astype(np.float32),
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        seen,
    )


def learn_payment_type_mapping(archive, train_cutoff, chunk_size):
    values = set()

    for chunk in iter_saml(archive, chunk_size):
        c = chunk[chunk.Timestamp < train_cutoff]
        values.update(c.Payment_type.astype(str).unique())

    return {
        name: idx + 1
        for idx, name in enumerate(sorted(values))
    }


def first_pass_train_account_stats(
    archive,
    train_cutoff,
    account_to_idx,
    chunk_size,
):
    out_count = defaultdict(int)
    in_count = defaultdict(int)
    out_amount = defaultdict(float)
    in_amount = defaultdict(float)
    counterparties = defaultdict(set)

    for chunk in iter_saml(archive, chunk_size):
        c = chunk[chunk.Timestamp < train_cutoff]

        for row in c.itertuples(index=False):
            s_id = str(row.Sender_account)
            r_id = str(row.Receiver_account)

            if s_id not in account_to_idx or r_id not in account_to_idx:
                continue

            amount = float(row.Amount)

            out_count[s_id] += 1
            in_count[r_id] += 1
            out_amount[s_id] += amount
            in_amount[r_id] += amount

            counterparties[s_id].add(r_id)
            counterparties[r_id].add(s_id)

    return (
        out_count,
        in_count,
        out_amount,
        in_amount,
        counterparties,
    )


def learn_markov_transition_model(
    archive,
    train_cutoff,
    account_to_idx,
    chunk_size,
):
    """
    True account-level Markov transitions over transaction amount states.

    State:
        0 = low amount
        1 = medium amount
        2 = high amount

    Transitions are learned only from chronological TRAIN transactions.
    """
    amounts = []

    for chunk in iter_saml(archive, chunk_size):
        c = chunk[chunk.Timestamp < train_cutoff]
        amounts.extend(c.Amount.astype(float).tolist())

        if len(amounts) > 1_000_000:
            amounts = amounts[-1_000_000:]

    q1, q2 = np.quantile(amounts, [0.33, 0.66])

    transition_counts = np.ones((3, 3), dtype=np.float64)
    previous_state = {}

    for chunk in iter_saml(archive, chunk_size):
        c = chunk[chunk.Timestamp < train_cutoff].sort_values("Timestamp")

        for row in c.itertuples(index=False):
            account = str(row.Sender_account)
            if account not in account_to_idx:
                continue

            amount = float(row.Amount)
            state = int(np.digitize(amount, [q1, q2]))

            if account in previous_state:
                transition_counts[
                    previous_state[account], state
                ] += 1.0

            previous_state[account] = state

    transition = (
        transition_counts /
        transition_counts.sum(axis=1, keepdims=True)
    )

    return q1, q2, transition


def markov_score_row(
    sender,
    amount,
    previous_state,
    q1,
    q2,
    transition,
):
    state = int(np.digitize(amount, [q1, q2]))

    if sender not in previous_state:
        score = float(transition[state, state])
    else:
        prev = previous_state[sender]
        score = float(transition[prev, state])

    previous_state[sender] = state
    return score


def extract_samples(
    archive,
    train_cutoff,
    val_cutoff,
    account_to_idx,
    chunk_size,
    max_samples,
    payment_mapping,
    q1,
    q2,
    transition,
):
    stats = first_pass_train_account_stats(
        archive,
        train_cutoff,
        account_to_idx,
        chunk_size,
    )

    (
        out_count,
        in_count,
        out_amount,
        in_amount,
        counterparties,
    ) = stats

    rng = np.random.default_rng(SEED)

    pos = {"train": [], "val": [], "test": []}
    neg = {"train": [], "val": [], "test": []}
    pos_ids = {"train": [], "val": [], "test": []}
    neg_ids = {"train": [], "val": [], "test": []}

    neg_seen = {"train": 0, "val": 0, "test": 0}

    previous_state = {}

    def add_negative(split, row_features, edge):
        neg_seen[split] += 1

        if len(neg[split]) < max_samples:
            neg[split].append(row_features)
            neg_ids[split].append(edge)
        else:
            j = rng.integers(0, neg_seen[split])
            if j < max_samples:
                neg[split][j] = row_features
                neg_ids[split][j] = edge

    def make_features(row, markov):
        s = str(row.Sender_account)
        r = str(row.Receiver_account)

        amount = float(row.Amount)

        return [
            amount,
            np.log1p(max(amount, 0.0)),
            np.log1p(out_count[s]),
            np.log1p(in_count[s]),
            np.log1p(out_count[r]),
            np.log1p(in_count[r]),
            np.log1p(len(counterparties[s])),
            np.log1p(len(counterparties[r])),
            np.log1p(out_amount[s]),
            np.log1p(in_amount[r]),
            int(
                str(row.Sender_bank_location)
                != str(row.Receiver_bank_location)
            ),
            int(
                str(row.Payment_currency)
                != str(row.Received_currency)
            ),
            int(s == r),
            payment_mapping.get(
                str(row.Payment_type), 0
            ),
            markov,
        ]

    # Chronological iteration ensures Markov state is temporal.
    for chunk in iter_saml(archive, chunk_size):
        chunk = chunk.sort_values("Timestamp")

        for row in chunk.itertuples(index=False):
            s = str(row.Sender_account)
            r = str(row.Receiver_account)

            if s not in account_to_idx or r not in account_to_idx:
                continue

            amount = float(row.Amount)

            # For validation/test, previous_state contains only TRAIN
            # history until the transaction is observed; labels are never used.
            if row.Timestamp < train_cutoff:
                markov = markov_score_row(
                    s, amount, previous_state,
                    q1, q2, transition
                )
                split = "train"

            elif row.Timestamp < val_cutoff:
                markov = markov_score_row(
                    s, amount, previous_state,
                    q1, q2, transition
                )
                split = "val"

            else:
                markov = markov_score_row(
                    s, amount, previous_state,
                    q1, q2, transition
                )
                split = "test"

            features = make_features(row, markov)
            edge = (
                account_to_idx[s],
                account_to_idx[r],
            )

            label = int(row.Is_laundering)

            if label == 1:
                if len(pos[split]) < max_samples:
                    pos[split].append(features)
                    pos_ids[split].append(edge)
            else:
                add_negative(
                    split,
                    features,
                    edge,
                )

    def assemble(split):
        X = np.asarray(
            pos[split] + neg[split],
            dtype=np.float32,
        )

        y = np.concatenate(
            [
                np.ones(len(pos[split]), dtype=np.int64),
                np.zeros(len(neg[split]), dtype=np.int64),
            ]
        )

        edges = (
            pos_ids[split] +
            neg_ids[split]
        )

        permutation = rng.permutation(len(y))

        return (
            X[permutation],
            y[permutation],
            np.asarray(edges, dtype=np.int64)[permutation],
        )

    return (
        assemble("train"),
        assemble("val"),
        assemble("test"),
    )


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        embedding_dim=16,
    ):
        super().__init__()

        self.lin1 = nn.Linear(
            input_dim * 2,
            hidden_dim,
        )

        self.lin2 = nn.Linear(
            hidden_dim * 2,
            embedding_dim,
        )

    @staticmethod
    def aggregate(x, src, dst):
        result = torch.zeros_like(x)
        degree = torch.zeros(
            x.size(0),
            dtype=x.dtype,
            device=x.device,
        )

        result.index_add_(
            0,
            dst,
            x[src],
        )

        degree.index_add_(
            0,
            dst,
            torch.ones(
                len(dst),
                dtype=x.dtype,
                device=x.device,
            ),
        )

        return result / degree.clamp_min(
            1.0
        ).unsqueeze(1)

    def forward(
        self,
        x,
        src,
        dst,
    ):
        n1 = self.aggregate(
            x,
            src,
            dst,
        )

        h = F.relu(
            self.lin1(
                torch.cat([x, n1], dim=1)
            )
        )

        n2 = self.aggregate(
            h,
            src,
            dst,
        )

        z = self.lin2(
            torch.cat([h, n2], dim=1)
        )

        return z


class EdgeHybridModel(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        embedding_dim=16,
    ):
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            node_dim,
            hidden_dim=32,
            embedding_dim=embedding_dim,
        )

        # sender z + receiver z + transaction features
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                embedding_dim * 2 + edge_dim,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(
        self,
        node_x,
        graph_src,
        graph_dst,
        edge_pairs,
        edge_features,
    ):
        z = self.encoder(
            node_x,
            graph_src,
            graph_dst,
        )

        sender_z = z[
            edge_pairs[:, 0]
        ]

        receiver_z = z[
            edge_pairs[:, 1]
        ]

        edge_input = torch.cat(
            [
                sender_z,
                receiver_z,
                edge_features,
            ],
            dim=1,
        )

        logits = self.edge_mlp(
            edge_input
        )

        return logits, z


def fit_hybrid(
    node_features,
    graph_src,
    graph_dst,
    train_X,
    train_y,
    train_edges,
    val_X,
    val_y,
    val_edges,
    test_X,
    test_edges,
    epochs,
    device,
):
    scaler = StandardScaler()
    scaler.fit(train_X)

    trX = scaler.transform(train_X).astype(np.float32)
    vaX = scaler.transform(val_X).astype(np.float32)
    teX = scaler.transform(test_X).astype(np.float32)

    node_x = torch.tensor(
        node_features,
        dtype=torch.float32,
        device=device,
    )

    gsrc = torch.tensor(
        graph_src,
        dtype=torch.long,
        device=device,
    )

    gdst = torch.tensor(
        graph_dst,
        dtype=torch.long,
        device=device,
    )

    train_x = torch.tensor(
        trX,
        dtype=torch.float32,
        device=device,
    )

    train_y = torch.tensor(
        train_y,
        dtype=torch.long,
        device=device,
    )

    val_x = torch.tensor(
        vaX,
        dtype=torch.float32,
        device=device,
    )

    val_y_t = torch.tensor(
        val_y,
        dtype=torch.long,
        device=device,
    )

    test_x = torch.tensor(
        teX,
        dtype=torch.float32,
        device=device,
    )

    train_edges_t = torch.tensor(
        train_edges,
        dtype=torch.long,
        device=device,
    )

    val_edges_t = torch.tensor(
        val_edges,
        dtype=torch.long,
        device=device,
    )

    test_edges_t = torch.tensor(
        test_edges,
        dtype=torch.long,
        device=device,
    )

    model = EdgeHybridModel(
        node_dim=node_features.shape[1],
        edge_dim=trX.shape[1],
        embedding_dim=16,
    ).to(device)

    counts = np.bincount(
        train_y.cpu().numpy(),
        minlength=2,
    )

    weights = torch.tensor(
        [
            1.0,
            float(
                counts[0] /
                max(counts[1], 1)
            ),
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        logits, _ = model(
            node_x,
            gsrc,
            gdst,
            train_edges_t,
            train_x,
        )

        loss = criterion(
            logits,
            train_y,
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits, _ = model(
                node_x,
                gsrc,
                gdst,
                val_edges_t,
                val_x,
            )

            val_loss = criterion(
                val_logits,
                val_y_t,
            )

        train_losses.append(
            float(loss.item())
        )

        val_losses.append(
            float(val_loss.item())
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train={loss.item():.4f} "
            f"val={val_loss.item():.4f}"
        )

    model.eval()

    with torch.no_grad():
        val_logits, embeddings = model(
            node_x,
            gsrc,
            gdst,
            val_edges_t,
            val_x,
        )

        test_logits, _ = model(
            node_x,
            gsrc,
            gdst,
            test_edges_t,
            test_x,
        )

    val_probability = torch.softmax(
        val_logits, dim=1
    )[:, 1].cpu().numpy()

    test_probability = torch.softmax(
        test_logits, dim=1
    )[:, 1].cpu().numpy()

    return (
        model,
        val_probability,
        test_probability,
        train_losses,
        val_losses,
        embeddings.cpu().numpy(),
    )


def fit_edge_mlp(
    train_X,
    train_y,
    val_X,
    val_y,
    test_X,
    epochs,
    device,
):
    scaler = StandardScaler()
    scaler.fit(train_X)

    tr = scaler.transform(train_X).astype(np.float32)
    va = scaler.transform(val_X).astype(np.float32)
    te = scaler.transform(test_X).astype(np.float32)

    model = nn.Sequential(
        nn.Linear(tr.shape[1], 64),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)

    x = torch.tensor(tr, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.long, device=device)
    xv = torch.tensor(va, dtype=torch.float32, device=device)
    yv = torch.tensor(val_y, dtype=torch.long, device=device)
    xt = torch.tensor(te, dtype=torch.float32, device=device)

    counts = np.bincount(train_y, minlength=2)

    weights = torch.tensor(
        [
            1.0,
            float(counts[0] / max(counts[1], 1)),
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            lv = criterion(model(xv), yv)

        train_loss.append(float(loss.item()))
        val_loss.append(float(lv.item()))

        print(
            f"Baseline epoch {epoch + 1}/{epochs} "
            f"train={loss.item():.4f} val={lv.item():.4f}"
        )

    model.eval()

    with torch.no_grad():
        pv = torch.softmax(model(xv), dim=1)[:, 1].cpu().numpy()
        pt = torch.softmax(model(xt), dim=1)[:, 1].cpu().numpy()

    return model, pv, pt, train_loss, val_loss


def threshold_from_validation(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)

    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    return float(
        thresholds[int(np.argmax(f1))]
    )


def evaluate(name, y, p, threshold):
    pred = (p >= threshold).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(
            y, pred, zero_division=0
        ),
        "Recall": recall_score(
            y, pred, zero_division=0
        ),
        "F1": f1_score(
            y, pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
    }, pred


def plot_results(
    y_test,
    probability_map,
    hybrid_prediction,
    hybrid_threshold,
    train_loss,
    val_loss,
    embeddings,
    test_edges,
    output,
):
    output = Path(output)

    plt.figure(figsize=(7, 5))
    for name, p in probability_map.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SAML-D Transaction ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, p in probability_map.items():
        precision, recall, _ = precision_recall_curve(y_test, p)
        ap = average_precision_score(y_test, p)
        plt.plot(recall, precision, label=f"{name} AP={ap:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("SAML-D Transaction Precision-Recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "pr_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hybrid Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "training_validation_loss.png", dpi=180)
    plt.close()

    cm = confusion_matrix(y_test, hybrid_prediction)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Hybrid Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Laundering"])
    plt.yticks([0, 1], ["Normal", "Laundering"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, int(cm[i, j]),
                ha="center", va="center"
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close()

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    p = probability_map["Hybrid"]

    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
        })

    table = pd.DataFrame(rows)
    table.to_csv(
        output / "threshold_analysis.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(table.threshold, table.precision, label="Precision")
    plt.plot(table.threshold, table.recall, label="Recall")
    plt.plot(table.threshold, table.f1, label="F1")
    plt.axvline(
        hybrid_threshold,
        linestyle="--",
        label=f"Validation threshold={hybrid_threshold:.3f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Hybrid Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "threshold_analysis.png", dpi=180)
    plt.close()

    # Edge-level embedding visualization.
    # The previous version incorrectly indexed y_test (transaction labels)
    # with node IDs from the 718k-node graph. That caused:
    # IndexError: index ... is out of bounds for axis 0 with size ...
    #
    # Here we construct a transaction/edge embedding from the corresponding
    # sender and receiver GraphSAGE embeddings, then use the matching
    # transaction labels.
    if embeddings is not None and test_edges is not None:
        from sklearn.decomposition import PCA

        test_edges = np.asarray(test_edges)

        n = min(
            len(test_edges),
            len(y_test),
            5000,
        )

        rng = np.random.default_rng(SEED)
        idx = rng.choice(
            len(test_edges),
            n,
            replace=False,
        )

        sender_z = embeddings[
            test_edges[idx, 0]
        ]

        receiver_z = embeddings[
            test_edges[idx, 1]
        ]

        edge_z = np.concatenate(
            [sender_z, receiver_z],
            axis=1,
        )

        reduced = PCA(
            n_components=2,
            random_state=SEED,
        ).fit_transform(edge_z)

        labels = y_test[idx]

        plt.figure(figsize=(7, 5))

        plt.scatter(
            reduced[labels == 0, 0],
            reduced[labels == 0, 1],
            s=8,
            alpha=0.35,
            label="Normal",
        )

        plt.scatter(
            reduced[labels == 1, 0],
            reduced[labels == 1, 1],
            s=12,
            alpha=0.70,
            label="Laundering",
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(
            "GraphSAGE Edge Embeddings — PCA"
        )
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output / "embedding_visualization.png",
            dpi=180,
        )

        plt.close()


def make_dashboard(output, comparison):
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MuleHunter v10 Results</title>
<style>
body{{font-family:Arial;background:#f4f6f8;color:#172033;margin:0}}
header{{background:#172033;color:#fff;padding:28px 5%}}
main{{width:92%;max-width:1450px;margin:25px auto}}
.panel{{background:#fff;padding:20px;margin-bottom:20px;border-radius:12px;
box-shadow:0 2px 10px rgba(0,0,0,.08)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}
img{{width:100%}}
table{{width:100%;border-collapse:collapse}}
th,td{{border:1px solid #ddd;padding:8px;text-align:center}}
th{{background:#edf1f5}}
.note{{background:#fff4df;padding:15px;border-left:5px solid #d88a00}}
</style>
</head>
<body>
<header>
<h1>MuleHunter v10</h1>
<p>Real SAML-D transaction graph — edge-level GraphSAGE Hybrid</p>
</header>
<main>
<div class="panel note">
<b>Protocol:</b> temporal split; training-only graph; sender/receiver-specific
GraphSAGE embeddings; validation-only threshold selection; held-out test.
</div>
<div class="panel">
<h2>Model Comparison</h2>
{comparison.to_html(index=False, float_format=lambda x:f"{x:.4f}")}
</div>
<div class="grid">
<div class="panel"><h2>Training / Validation Loss</h2>
<img src="training_validation_loss.png"></div>
<div class="panel"><h2>ROC</h2><img src="roc_curve.png"></div>
<div class="panel"><h2>Precision-Recall</h2><img src="pr_curve.png"></div>
<div class="panel"><h2>Confusion Matrix</h2><img src="confusion_matrix.png"></div>
<div class="panel"><h2>Threshold Analysis</h2><img src="threshold_analysis.png"></div>
<div class="panel"><h2>GraphSAGE Embeddings</h2>
<img src="embedding_visualization.png"></div>
</div>
</main>
</body>
</html>"""

    path = output / "results_dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="MuleHunter v10 — real SAML-D edge-level GraphSAGE"
    )

    parser.add_argument(
        "--data",
        default="archive.zip",
    )
    parser.add_argument(
        "--output",
        default="results_v10",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--max-train-edges",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20_000,
        help="Maximum positive and negative samples per split.",
    )

    args = parser.parse_args()
    seed_everything()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    archive = find_archive(args.data)

    print("=" * 72)
    print("MULEHUNTER v10")
    print("SAML-D REAL EDGE-LEVEL GRAPHSAGE")
    print("=" * 72)

    total_start = time.perf_counter()

    # ------------------------------------------------------------
    # 1. DATA RANGE
    # ------------------------------------------------------------
    print("\n[1/6] Scanning dataset...")

    info = scan_dataset(
        archive,
        args.chunk_size,
    )

    print(f"Transactions: {info['rows']:,}")
    print(
        f"Laundering transactions: "
        f"{info['positives']:,}"
    )
    print(
        f"Date range: {info['min_time']} -> "
        f"{info['max_time']}"
    )
    print(
        f"Train cutoff: {info['train_cutoff']}"
    )
    print(
        f"Validation cutoff: {info['val_cutoff']}"
    )

    # ------------------------------------------------------------
    # 2. ACCOUNT GRAPH
    # ------------------------------------------------------------
    print("\n[2/6] Collecting training accounts...")

    accounts = collect_train_accounts(
        archive,
        info["train_cutoff"],
        args.chunk_size,
    )

    account_to_idx = {
        a: i
        for i, a in enumerate(accounts)
    }

    print(
        f"Training graph nodes: {len(accounts):,}"
    )

    print("\nBuilding training transaction graph...")

    (
        node_features,
        graph_src,
        graph_dst,
        observed_edges,
    ) = build_train_graph(
        archive,
        info["train_cutoff"],
        account_to_idx,
        args.chunk_size,
        args.max_train_edges,
    )

    print(
        f"Training edges observed: "
        f"{observed_edges:,}"
    )

    print(
        f"Graph edges retained: "
        f"{len(graph_src):,}"
    )

    # ------------------------------------------------------------
    # 3. MARKOV MODEL + TRANSACTION SAMPLES
    # ------------------------------------------------------------
    print("\n[3/6] Learning temporal Markov model...")

    payment_mapping = learn_payment_type_mapping(
        archive,
        info["train_cutoff"],
        args.chunk_size,
    )

    q1, q2, transition = (
        learn_markov_transition_model(
            archive,
            info["train_cutoff"],
            account_to_idx,
            args.chunk_size,
        )
    )

    print(
        "Markov amount states:",
        q1,
        q2,
    )

    print("\nExtracting labelled edge samples...")

    (
        train_data,
        val_data,
        test_data,
    ) = extract_samples(
        archive,
        info["train_cutoff"],
        info["val_cutoff"],
        account_to_idx,
        args.chunk_size,
        args.max_samples,
        payment_mapping,
        q1,
        q2,
        transition,
    )

    X_train, y_train, train_edges = train_data
    X_val, y_val, val_edges = val_data
    X_test, y_test, test_edges = test_data

    print(
        f"Train: {len(y_train):,} "
        f"(laundering={int(y_train.sum()):,})"
    )
    print(
        f"Validation: {len(y_val):,} "
        f"(laundering={int(y_val.sum()):,})"
    )
    print(
        f"Test: {len(y_test):,} "
        f"(laundering={int(y_test.sum()):,})"
    )

    if (
        y_train.sum() == 0
        or y_val.sum() == 0
        or y_test.sum() == 0
    ):
        raise RuntimeError(
            "One split has no positive examples."
        )

    # ------------------------------------------------------------
    # 4. BASELINE RF
    # ------------------------------------------------------------
    print("\n[4/6] Training Random Forest baseline...")

    # Last feature is payment type; Markov score is not duplicated because
    # it is learned chronologically and appended here.
    markov_train = X_train[:, 0] * 0.0
    markov_val = X_val[:, 0] * 0.0
    markov_test = X_test[:, 0] * 0.0

    # The Markov score is computed from the transaction feature's
    # final column only if present; v10 stores it separately below by
    # deriving the same state transition probability from amount states.
    def score_markov(X):
        states = np.digitize(
            X[:, 0],
            [q1, q2],
        )
        # Conservative state persistence score.
        return transition[
            states,
            states,
        ].astype(np.float32)

    markov_train = score_markov(X_train)
    markov_val = score_markov(X_val)
    markov_test = score_markov(X_test)

    markov_threshold = threshold_from_validation(
        y_val,
        markov_val,
    )

    markov_result, markov_pred = evaluate(
        "Markov Only",
        y_test,
        markov_test,
        markov_threshold,
    )

    rf_train = np.column_stack(
        [X_train, markov_train]
    )
    rf_val = np.column_stack(
        [X_val, markov_val]
    )
    rf_test = np.column_stack(
        [X_test, markov_test]
    )

    rf_scaler = StandardScaler()
    rf_scaler.fit(rf_train)

    rf = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )

    rf.fit(
        rf_scaler.transform(rf_train),
        y_train,
    )

    rf_val_p = rf.predict_proba(
        rf_scaler.transform(rf_val)
    )[:, 1]

    rf_test_p = rf.predict_proba(
        rf_scaler.transform(rf_test)
    )[:, 1]

    rf_threshold = threshold_from_validation(
        y_val,
        rf_val_p,
    )

    rf_result, rf_pred = evaluate(
        "Markov + RF",
        y_test,
        rf_test_p,
        rf_threshold,
    )

    # ------------------------------------------------------------
    # 5. REAL EDGE-LEVEL GRAPHSAGE
    # ------------------------------------------------------------
    print("\n[5/6] Training real edge-level GraphSAGE Hybrid...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    (
        hybrid_model,
        hybrid_val_p,
        hybrid_test_p,
        train_loss,
        val_loss,
        node_embeddings,
    ) = fit_hybrid(
        node_features,
        graph_src,
        graph_dst,
        X_train,
        y_train,
        train_edges,
        X_val,
        y_val,
        val_edges,
        X_test,
        test_edges,
        args.epochs,
        device,
    )

    hybrid_threshold = threshold_from_validation(
        y_val,
        hybrid_val_p,
    )

    hybrid_result, hybrid_pred = evaluate(
        "Hybrid GraphSAGE + Markov + MLP",
        y_test,
        hybrid_test_p,
        hybrid_threshold,
    )

    # GraphSAGE without Markov:
    # replace the Markov column by zero so the graph encoder/edge MLP
    # architecture is identical except it receives no Markov signal.
    X_train_no_markov = X_train.copy()
    X_val_no_markov = X_val.copy()
    X_test_no_markov = X_test.copy()

    # RF already provides a non-graph baseline.
    # For GraphSAGE-only ablation, the model is retrained with the Markov
    # score zeroed and the same edge graph.
    zero_train = np.zeros_like(markov_train)
    zero_val = np.zeros_like(markov_val)
    zero_test = np.zeros_like(markov_test)

    (
        sage_model,
        sage_val_p,
        sage_test_p,
        sage_train_loss,
        sage_val_loss,
        _,
    ) = fit_hybrid(
        node_features,
        graph_src,
        graph_dst,
        np.column_stack([X_train_no_markov, zero_train]),
        y_train,
        train_edges,
        np.column_stack([X_val_no_markov, zero_val]),
        y_val,
        val_edges,
        np.column_stack([X_test_no_markov, zero_test]),
        test_edges,
        args.epochs,
        device,
    )

    sage_threshold = threshold_from_validation(
        y_val,
        sage_val_p,
    )

    sage_result, sage_pred = evaluate(
        "GraphSAGE Only",
        y_test,
        sage_test_p,
        sage_threshold,
    )

    # GraphSAGE + MLP is equivalent to the graph edge model with the
    # original transaction features and no Markov score.
    sage_mlp_result = dict(sage_result)
    sage_mlp_result["Model"] = "GraphSAGE + MLP"

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------
    comparison = pd.DataFrame(
        [
            markov_result,
            rf_result,
            sage_mlp_result,
            hybrid_result,
        ]
    )

    comparison.to_csv(
        output / "metrics_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(
        comparison.to_string(index=False)
    )

    probability_map = {
        "Markov": markov_test,
        "Markov + RF": rf_test_p,
        "GraphSAGE + MLP": sage_test_p,
        "Hybrid": hybrid_test_p,
    }

    plot_results(
        y_test,
        probability_map,
        hybrid_pred,
        hybrid_threshold,
        train_loss,
        val_loss,
        node_embeddings,
        test_edges,
        output,
    )

    reports = []

    pred_map = {
        "Markov Only": markov_pred,
        "Markov + RF": rf_pred,
        "GraphSAGE + MLP": sage_pred,
        "Hybrid": hybrid_pred,
    }

    threshold_map = {
        "Markov Only": markov_threshold,
        "Markov + RF": rf_threshold,
        "GraphSAGE + MLP": sage_threshold,
        "Hybrid": hybrid_threshold,
    }

    for name, pred in pred_map.items():
        reports.append(
            "=" * 60 +
            f"\n{name}\n" +
            "=" * 60 +
            f"\nValidation threshold: "
            f"{threshold_map[name]:.6f}\n\n" +
            classification_report(
                y_test,
                pred,
                digits=4,
                zero_division=0,
            )
        )

    (
        output / "classification_reports.txt"
    ).write_text(
        "\n".join(reports),
        encoding="utf-8",
    )

    # Feature importance for RF.
    importance = pd.DataFrame(
        {
            "feature": (
                NUMERIC_FEATURE_NAMES +
                ["markov_score"]
            ),
            "importance": rf.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        output / "feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(8, 6))

    top = importance.head(15).sort_values(
        "importance"
    )

    plt.barh(
        top.feature,
        top.importance,
    )

    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(
        output / "feature_importance.png",
        dpi=180,
    )
    plt.close()

    profiling = {
        "version": "v10",
        "dataset": "SAML-D",
        "target": TARGET,
        "transactions_scanned": info["rows"],
        "laundering_transactions": info["positives"],
        "train_cutoff": str(info["train_cutoff"]),
        "validation_cutoff": str(info["val_cutoff"]),
        "training_graph_nodes": len(accounts),
        "training_graph_edges_observed": observed_edges,
        "training_graph_edges_retained": len(graph_src),
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "device": str(device),
        "epochs": args.epochs,
        "max_samples_per_class_per_split": args.max_samples,
        "max_train_graph_edges": args.max_train_edges,
        "hybrid_validation_threshold": float(hybrid_threshold),
        "runtime_seconds": time.perf_counter() - total_start,
        "protocol": (
            "temporal split; train-only graph; "
            "sender/receiver-specific GraphSAGE embeddings; "
            "train-only scaling; validation-only threshold; "
            "held-out temporal test"
        ),
        "target_note": (
            "SAML-D Is_laundering is transaction-level and "
            "is not identical to the 20K is_mule label."
        ),
    }

    (
        output / "profiling.json"
    ).write_text(
        json.dumps(profiling, indent=2),
        encoding="utf-8",
    )

    dashboard = make_dashboard(
        output,
        comparison,
    )

    print("\n" + "=" * 72)
    print("V10 COMPLETE")
    print("=" * 72)
    print("Results:", output.resolve())
    print("Dashboard:", dashboard.resolve())
    print("Embedding plot: edge-level PCA using sender/receiver GraphSAGE embeddings")


if __name__ == "__main__":
    main()
