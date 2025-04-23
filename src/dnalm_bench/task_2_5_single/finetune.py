import os
from abc import ABCMeta, abstractmethod
import hashlib
import shutil
import importlib
import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    AutoModel,
)
from tqdm import tqdm
import polars as pl
import h5py
import pyfaidx
import pyBigWig
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

from ..finetune import HFClassifierModel, LoRAModule
from ..utils import onehot_to_chars, one_hot_encode, NoModule, log1mexp


class ChromatinEndToEndDataset(Dataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "elem_start": pl.UInt32,
        "elem_end": pl.UInt32,
    }

    def __init__(
        self,
        genome_fa,
        bigwig,
        elements_tsv,
        chroms,
        crop,
        downsample_ratio=None,
        cache_dir=None,
        return_idx_orig=False,
    ):
        super().__init__()

        self.crop = crop
        self.return_idx_orig = return_idx_orig

        self.elements_df_all = self._load_elements(elements_tsv, chroms)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

            bw_path_abs = os.path.abspath(bigwig)
            bw_path_hash = hashlib.sha256(bw_path_abs.encode("utf-8")).hexdigest()
            bw_cache_path = os.path.join(cache_dir, bw_path_hash + ".bw")
            self._copy_if_not_exists(bigwig, bw_cache_path)
            bigwig = bw_cache_path

            fa_path_abs = os.path.abspath(genome_fa)
            fa_idx_path_abs = fa_path_abs + ".fai"
            fa_path_hash = hashlib.sha256(fa_path_abs.encode("utf-8")).hexdigest()
            fa_cache_path = os.path.join(cache_dir, fa_path_hash + ".fa")
            fa_idx_cache_path = fa_cache_path + ".fai"
            self._copy_if_not_exists(genome_fa, fa_cache_path)
            genome_fa = fa_cache_path
            try:
                self._copy_if_not_exists(fa_idx_path_abs, fa_idx_cache_path)
            except FileNotFoundError:
                pass

        self.genome_fa = genome_fa
        fa = pyfaidx.Fasta(self.genome_fa)  # Build index if needed
        fa.close()

        self.bw = bigwig

        self.downsample_ratio = downsample_ratio
        if downsample_ratio is None:
            self.elements_df = self.elements_df_all

    @classmethod
    def _load_elements(cls, elements_file, chroms):
        df = pl.scan_csv(
            elements_file, separator="\t", quote_char=None, dtypes=cls._elements_dtypes
        ).with_row_index()

        if chroms is not None:
            df = df.filter(pl.col("chr").is_in(chroms))

        df = df.collect()

        return df

    @staticmethod
    def _copy_if_not_exists(src, dst):
        try:
            with open(dst, "xb") as f, open(src, "rb") as sf:
                shutil.copyfileobj(sf, f)
        except FileExistsError:
            pass

    def set_epoch(self, epoch):
        if self.downsample_ratio is None:
            return

        offset = epoch % self.downsample_ratio
        self.elements_df = self.elements_df_all.take_every(
            n=self.downsample_ratio, offset=offset
        )

    def __len__(self):
        return self.elements_df.height

    def __getitem__(self, idx):
        idx_orig, chrom, start, end, elem_start, elem_end, _, _ = self.elements_df.row(
            idx
        )

        seq = np.zeros((end - start, 4), dtype=np.int8)

        fa = pyfaidx.Fasta(self.genome_fa, one_based_attributes=False)

        sequence_data = fa[chrom][max(0, start) : end]
        sequence = sequence_data.seq.upper()
        start_adj = sequence_data.start
        end_adj = sequence_data.end

        a = start_adj - start
        b = end_adj - start
        seq[a:b, :] = one_hot_encode(sequence)

        fa.close()

        out_start = start + self.crop
        out_end = end - self.crop
        out_start_adj = max(out_start, start_adj)
        out_end_adj = min(out_end, end_adj)

        c = out_start_adj - out_start
        d = out_end_adj - out_start

        signal = np.zeros(out_end - out_start, dtype=np.float32)

        bw = pyBigWig.open(self.bw)
        track = bw.values(chrom, out_start_adj, out_end_adj, numpy=True)
        signal[c:d] = np.nan_to_num(track)
        bw.close()

        if self.return_idx_orig:
            return (
                torch.from_numpy(seq),
                torch.from_numpy(signal),
                torch.tensor(idx_orig),
            )

        return torch.from_numpy(seq), torch.from_numpy(signal)


class PeaksEndToEndDataset(Dataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "label": pl.Utf8,
    }

    def __init__(
        self,
        genome_fa,
        elements_tsv,
        chroms,
        classes,
        cache_dir=None,
        return_idx_orig=False,
    ):
        super().__init__()

        self.classes = classes
        self.elements_df = self._load_elements(elements_tsv, chroms)

        self.return_idx_orig = return_idx_orig

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

            fa_path_abs = os.path.abspath(genome_fa)
            fa_idx_path_abs = fa_path_abs + ".fai"
            fa_path_hash = hashlib.sha256(fa_path_abs.encode("utf-8")).hexdigest()
            fa_cache_path = os.path.join(cache_dir, fa_path_hash + ".fa")
            fa_idx_cache_path = fa_cache_path + ".fai"
            self._copy_if_not_exists(genome_fa, fa_cache_path)
            genome_fa = fa_cache_path
            try:
                self._copy_if_not_exists(fa_idx_path_abs, fa_idx_cache_path)
            except FileNotFoundError:
                pass

        self.genome_fa = genome_fa
        fa = pyfaidx.Fasta(self.genome_fa)  # Build index if needed
        fa.close()

    @classmethod
    def _load_elements(cls, elements_file, chroms):
        df = pl.scan_csv(
            elements_file, separator="\t", quote_char=None, dtypes=cls._elements_dtypes
        ).with_row_index()

        if chroms is not None:
            df = df.filter(pl.col("chr").is_in(chroms))

        df = df.collect()

        return df

    @staticmethod
    def _copy_if_not_exists(src, dst):
        try:
            with open(dst, "xb") as f, open(src, "rb") as sf:
                shutil.copyfileobj(sf, f)
        except FileExistsError:
            pass

    def __len__(self):
        return self.elements_df.height

    def __getitem__(self, idx):
        idx_orig, chrom, start, end, _, _, _, label = self.elements_df.row(idx)

        seq = np.zeros((end - start, 4), dtype=np.int8)

        fa = pyfaidx.Fasta(self.genome_fa, one_based_attributes=False)

        sequence_data = fa[chrom][max(0, start) : end]
        sequence = sequence_data.seq.upper()
        start_adj = sequence_data.start
        end_adj = sequence_data.end

        a = start_adj - start
        b = end_adj - start

        seq[a:b, :] = one_hot_encode(sequence)

        fa.close()

        label_ind = self.classes[label]

        if self.return_idx_orig:
            return (
                torch.from_numpy(seq),
                torch.tensor(label_ind),
                torch.tensor(idx_orig),
            )

        return torch.from_numpy(seq), torch.tensor(label_ind)


def log1pMSELoss(log_predicted_counts, true_counts):
    log_true = torch.log(true_counts + 1)
    return torch.mean(torch.square(log_true - log_predicted_counts), dim=-1)


def pearson_correlation(a, b):
    a = a - torch.mean(a)
    b = b - torch.mean(b)

    var_a = torch.sum(a**2)
    var_b = torch.sum(b**2)
    cov = torch.sum(a * b)

    r = cov / torch.sqrt(var_a * var_b)
    r = torch.nan_to_num(r)

    return r.item()


def counts_pearson(log_preds, targets):
    log_targets = torch.log(targets + 1)

    r = pearson_correlation(log_preds, log_targets)

    return r


def counts_spearman(log_preds, targets):
    log_targets = torch.log(targets + 1)

    preds_rank = log_preds.argsort().argsort().float()
    targets_rank = log_targets.argsort().argsort().float()

    r = pearson_correlation(preds_rank, targets_rank)

    return r


def train_finetuned_chromatin_model(
    train_pos_dataset,
    train_neg_dataset,
    val_pos_dataset,
    val_neg_dataset,
    model,
    num_epochs,
    out_dir,
    batch_size,
    lr,
    wd,
    accumulate,
    num_workers,
    prefetch_factor,
    device,
    progress_bar=False,
    resume_from=None,
    seed=0,
):

    val_pos_dataloader = DataLoader(
        val_pos_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    val_neg_dataloader = DataLoader(
        val_neg_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = [
        "epoch",
        "val_loss",
        "val_pearson_all",
        "val_spearman_all",
        "val_pearson_peaks",
        "val_spearman_peaks",
    ]

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if resume_from is not None:
        # start_epoch = int(resume_from.split("_")[-1].split(".")[0]) + 1
        resume_checkpoint_path = os.path.join(out_dir, f"checkpoint_{resume_from}.pt")
        optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{resume_from}.pt")
        start_epoch = resume_from + 1
        checkpoint_resume = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint_resume, strict=False)
        try:
            optimizer_resume = torch.load(optimizer_checkpoint_path)
            optimizer.load_state_dict(optimizer_resume)
        except FileNotFoundError:
            warnings.warn(
                f"Optimizer checkpoint not found at {optimizer_checkpoint_path}"
            )
    else:
        start_epoch = 0

    with open(log_file, "a") as f:
        if resume_from is None:
            f.write("\t".join(log_cols) + "\n")
            f.flush()

        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_pos_dataset.set_epoch(epoch)
            train_neg_dataset.set_epoch(epoch)
            train_dataset = ConcatDataset([train_pos_dataset, train_neg_dataset])
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
            )

            optimizer.zero_grad()
            for i, (seq, track) in enumerate(
                tqdm(
                    train_dataloader,
                    disable=(not progress_bar),
                    desc="train",
                    ncols=120,
                )
            ):
                # seq = seq.to(device)
                track = track.to(device)
                true_counts = track.sum(dim=1)

                fallback = False
                try:
                    log1p_counts = model(seq).squeeze(1)
                    loss = log1pMSELoss(log1p_counts, true_counts) / accumulate
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    fallback = True
                    warnings.warn(
                        f"Batch {i} does not fit in memory, falling back to single sequence processing"
                    )

                if fallback:
                    for j in range(seq.shape[0]):
                        try:
                            seq_j = seq[j : j + 1]
                            true_counts_j = true_counts[j : j + 1]

                            log1p_counts_j = model(seq_j).squeeze(1)
                            loss_j = log1pMSELoss(log1p_counts_j, true_counts_j) / (
                                accumulate * seq.shape[0]
                            )
                            loss_j.backward()

                        except torch.cuda.OutOfMemoryError:
                            warnings.warn(
                                f"Failed to process sequence {i*j} due to OOM"
                            )

                if (i + 1) % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()

            val_loss = 0
            val_counts_pred = []
            val_counts_true = []
            model.eval()
            with torch.no_grad():
                for i, (seq, track) in enumerate(
                    tqdm(
                        val_pos_dataloader,
                        disable=(not progress_bar),
                        desc="val_pos",
                        ncols=120,
                    )
                ):
                    # seq = seq.to(device)
                    track = track.to(device)
                    true_counts = track.sum(dim=1)

                    log1p_counts = model(seq).squeeze(1)
                    loss = log1pMSELoss(log1p_counts, true_counts)

                    val_loss += loss.item()
                    val_counts_pred.append(log1p_counts)
                    val_counts_true.append(true_counts)

                val_counts_pred_peaks = torch.cat(val_counts_pred, dim=0)
                val_counts_true_peaks = torch.cat(val_counts_true, dim=0)

                val_pearson_peaks = counts_pearson(
                    val_counts_pred_peaks, val_counts_true_peaks
                )
                val_spearman_peaks = counts_spearman(
                    val_counts_pred_peaks, val_counts_true_peaks
                )

                for i, (seq, track) in enumerate(
                    tqdm(
                        val_neg_dataloader,
                        disable=(not progress_bar),
                        desc="val_neg",
                        ncols=120,
                    )
                ):
                    # seq = seq.to(device)
                    track = track.to(device)
                    true_counts = track.sum(dim=1)

                    log1p_counts = model(seq).squeeze(1)
                    loss = log1pMSELoss(log1p_counts, true_counts)

                    val_loss += loss.item()
                    val_counts_pred.append(log1p_counts)
                    val_counts_true.append(true_counts)

                val_loss /= len(val_pos_dataloader) + len(val_neg_dataloader)
                val_counts_pred = torch.cat(val_counts_pred, dim=0)
                val_counts_true = torch.cat(val_counts_true, dim=0)

                val_pearson_all = counts_pearson(val_counts_pred, val_counts_true)
                val_spearman_all = counts_spearman(val_counts_pred, val_counts_true)

            print(
                f"Epoch {epoch}: val_loss={val_loss}, val_pearson_all={val_pearson_all}, val_spearman_all={val_spearman_all}, val_pearson_peaks={val_pearson_peaks}, val_spearman_peaks={val_spearman_peaks}"
            )
            f.write(
                f"{epoch}\t{val_loss}\t{val_pearson_all}\t{val_spearman_all}\t{val_pearson_peaks}\t{val_spearman_peaks}\n"
            )
            f.flush()

            checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{epoch}.pt")
            torch.save(optimizer.state_dict(), optimizer_checkpoint_path)


def evaluate_finetuned_chromatin_model(
    pos_dataset,
    idr_dataset,
    neg_dataset,
    model,
    batch_size,
    out_path,
    num_workers,
    prefetch_factor,
    device,
    progress_bar=False,
    seed=0,
):
    # val_loss = 0
    # val_counts_pred = []
    # val_counts_true = []
    torch.manual_seed(seed)

    model.to(device)

    model.eval()

    with torch.no_grad():
        test_loss_pos = 0
        test_counts_pred_pos = []
        test_counts_true_pos = []
        test_pos_dataloader = DataLoader(
            pos_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
        for i, (seq, track) in enumerate(
            tqdm(
                test_pos_dataloader,
                disable=(not progress_bar),
                desc="test_pos",
                ncols=120,
            )
        ):
            track = track.to(device)
            true_counts = track.sum(dim=1)

            log1p_counts = model(seq).squeeze(1)
            loss = log1pMSELoss(log1p_counts, true_counts)

            test_loss_pos += loss.item()
            test_counts_pred_pos.append(log1p_counts)
            test_counts_true_pos.append(true_counts)

        test_counts_pred_pos = torch.cat(test_counts_pred_pos, dim=0)
        test_counts_true_pos = torch.cat(test_counts_true_pos, dim=0)
        test_pearson_pos = counts_pearson(test_counts_pred_pos, test_counts_true_pos)
        test_spearman_pos = counts_spearman(test_counts_pred_pos, test_counts_true_pos)
        test_loss_pos /= len(test_pos_dataloader)

        test_loss_idr = 0
        test_counts_pred_idr = []
        test_counts_true_idr = []
        test_idr_dataloader = DataLoader(
            idr_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
        for i, (seq, track) in enumerate(
            tqdm(
                test_idr_dataloader,
                disable=(not progress_bar),
                desc="test_idr",
                ncols=120,
            )
        ):
            track = track.to(device)
            true_counts = track.sum(dim=1)

            log1p_counts = model(seq).squeeze(1)
            loss = log1pMSELoss(log1p_counts, true_counts)

            test_loss_idr += loss.item()
            test_counts_pred_idr.append(log1p_counts)
            test_counts_true_idr.append(true_counts)

        test_counts_pred_idr = torch.cat(test_counts_pred_idr, dim=0)
        test_counts_true_idr = torch.cat(test_counts_true_idr, dim=0)
        test_pearson_idr = counts_pearson(test_counts_pred_idr, test_counts_true_idr)
        test_spearman_idr = counts_spearman(test_counts_pred_idr, test_counts_true_idr)
        test_loss_idr /= len(test_idr_dataloader)

        test_loss_neg = 0
        test_counts_pred_neg = []
        test_counts_true_neg = []
        test_neg_dataloader = DataLoader(
            neg_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
        for i, (seq, track) in enumerate(
            tqdm(
                test_neg_dataloader,
                disable=(not progress_bar),
                desc="test_neg",
                ncols=120,
            )
        ):
            track = track.to(device)
            true_counts = track.sum(dim=1)

            log1p_counts = model(seq).squeeze(1)
            loss = log1pMSELoss(log1p_counts, true_counts)

            test_loss_neg += loss.item()
            test_counts_pred_neg.append(log1p_counts)
            test_counts_true_neg.append(true_counts)

        test_counts_pred_neg = torch.cat(test_counts_pred_neg, dim=0)
        test_counts_true_neg = torch.cat(test_counts_true_neg, dim=0)
        test_pearson_neg = counts_pearson(test_counts_pred_neg, test_counts_true_neg)
        test_spearman_neg = counts_spearman(test_counts_pred_neg, test_counts_true_neg)
        test_loss_neg /= len(test_neg_dataloader)

        test_loss_all = (test_loss_pos + test_loss_neg) / 2
        test_counts_pred_all = torch.cat(
            [test_counts_pred_pos, test_counts_pred_neg], dim=0
        )
        test_counts_true_all = torch.cat(
            [test_counts_true_pos, test_counts_true_neg], dim=0
        )
        test_pearson_all = counts_pearson(test_counts_pred_all, test_counts_true_all)
        test_spearman_all = counts_spearman(test_counts_pred_all, test_counts_true_all)

        test_counts_pred_cls = torch.cat(
            [test_counts_pred_idr, test_counts_pred_neg], dim=0
        )
        test_labels = torch.cat(
            [
                torch.ones_like(test_counts_pred_idr),
                torch.zeros_like(test_counts_pred_neg),
            ],
            dim=0,
        )
        test_auroc = roc_auc_score(
            test_labels.numpy(force=True), test_counts_pred_cls.numpy(force=True)
        )
        test_auprc = average_precision_score(
            test_labels.numpy(force=True), test_counts_pred_cls.numpy(force=True)
        )

        metrics = {
            "test_loss_pos": test_loss_pos,
            "test_loss_idr": test_loss_idr,
            "test_loss_neg": test_loss_neg,
            "test_loss_all": test_loss_all,
            "test_pearson_pos": test_pearson_pos,
            "test_pearson_idr": test_pearson_idr,
            "test_pearson_neg": test_pearson_neg,
            "test_pearson_all": test_pearson_all,
            "test_spearman_pos": test_spearman_pos,
            "test_spearman_idr": test_spearman_idr,
            "test_spearman_neg": test_spearman_neg,
            "test_spearman_all": test_spearman_all,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
        }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def train_finetuned_peak_classifier(
    train_dataset,
    val_dataset,
    model,
    num_epochs,
    out_dir,
    batch_size,
    lr,
    wd,
    accumulate,
    num_workers,
    prefetch_factor,
    device,
    progress_bar=False,
    resume_from=None,
    seed=0,
):

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = ["epoch", "val_loss", "val_acc"]

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if resume_from is not None:
        resume_checkpoint_path = os.path.join(out_dir, f"checkpoint_{resume_from}.pt")
        optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{resume_from}.pt")
        start_epoch = resume_from + 1
        checkpoint_resume = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint_resume, strict=False)
        try:
            optimizer_resume = torch.load(optimizer_checkpoint_path)
            optimizer.load_state_dict(optimizer_resume)
        except FileNotFoundError:
            warnings.warn(
                f"Optimizer checkpoint not found at {optimizer_checkpoint_path}"
            )
    else:
        start_epoch = 0

    criterion = torch.nn.CrossEntropyLoss()

    with open(log_file, "a") as f:
        if resume_from is None:
            f.write("\t".join(log_cols) + "\n")
            f.flush()

        for epoch in range(start_epoch, num_epochs):
            model.train()

            optimizer.zero_grad()
            for i, (seq, labels) in enumerate(
                tqdm(
                    train_dataloader,
                    disable=(not progress_bar),
                    desc="train",
                    ncols=120,
                )
            ):
                # seq = seq.to(device)
                labels = labels.to(device)

                fallback = False
                try:
                    pred = model(seq).squeeze(1)
                    loss = criterion(pred, labels) / accumulate
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    fallback = True
                    warnings.warn(
                        f"Batch {i} does not fit in memory, falling back to single sequence processing"
                    )

                if fallback:
                    for j in range(seq.shape[0]):
                        try:
                            seq_j = seq[j : j + 1]
                            labels_j = labels[j : j + 1]

                            pred_j = model(seq_j).squeeze(1)
                            loss_j = criterion(pred_j, labels_j) / (
                                accumulate * seq.shape[0]
                            )
                            loss_j.backward()

                        except torch.cuda.OutOfMemoryError:
                            warnings.warn(
                                f"Failed to process sequence {i*j} due to OOM"
                            )

                if (i + 1) % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()

            val_loss = 0
            val_acc = 0
            model.eval()
            with torch.no_grad():
                for i, (seq, labels) in enumerate(
                    tqdm(
                        val_dataloader,
                        disable=(not progress_bar),
                        desc="val",
                        ncols=120,
                    )
                ):
                    labels = labels.to(device)

                    pred = model(seq).squeeze(1)
                    loss = criterion(pred, labels)

                    val_loss += loss.item()
                    val_acc += (pred.argmax(dim=1) == labels).sum().item()

            val_loss /= len(val_dataloader.dataset)
            val_acc /= len(val_dataloader.dataset)

            print(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc}")
            f.write(f"{epoch}\t{val_loss}\t{val_acc}\n")
            f.flush()

            checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{epoch}.pt")
            torch.save(optimizer.state_dict(), optimizer_checkpoint_path)


def eval_finetuned_peak_classifier(
    test_dataset,
    model,
    out_path,
    batch_size,
    num_workers,
    prefetch_factor,
    device,
    progress_bar=False,
    seed=0,
):

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    torch.manual_seed(seed)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0
    pred_log_probs = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, (seq, labels_batch) in enumerate(
            tqdm(test_dataloader, disable=(not progress_bar), desc="test", ncols=120)
        ):
            labels_batch = labels_batch.to(device)

            pred = model(seq)
            loss = criterion(pred, labels_batch)

            test_loss += loss.item()
            pred_log_probs.append(F.log_softmax(pred, dim=1))
            labels.append(labels_batch)

    test_loss /= len(test_dataloader.dataset)

    pred_log_probs = torch.cat(pred_log_probs, dim=0)
    log_probs_others = log1mexp(pred_log_probs)
    pred_log_odds = torch.nan_to_num(pred_log_probs - log_probs_others)
    labels = torch.cat(labels, dim=0)
    test_acc = (pred_log_probs.argmax(dim=1) == labels).sum().item() / len(
        test_dataloader.dataset
    )

    metrics = {"test_loss": test_loss, "test_acc": test_acc}

    for class_name, class_idx in test_dataloader.dataset.classes.items():
        class_log_odds = pred_log_odds[:, class_idx]
        class_preds = (class_log_odds >= 0).float()
        class_labels = (labels == class_idx).float()

        class_log_odds = class_log_odds.numpy(force=True)
        class_preds = class_preds.numpy(force=True)
        class_labels = class_labels.numpy(force=True)

        class_auroc = roc_auc_score(class_labels, class_log_odds)
        class_auprc = average_precision_score(class_labels, class_log_odds)
        class_mcc = matthews_corrcoef(class_labels, class_preds)
        class_acc = (class_preds == class_labels).sum().item() / len(class_labels)

        metrics[f"class_{class_name}_auroc"] = class_auroc
        metrics[f"class_{class_name}_auprc"] = class_auprc
        metrics[f"class_{class_name}_mcc"] = class_mcc
        metrics[f"class_{class_name}_acc"] = class_acc

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


class DNABERT2LoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"zhihan1996/{model_name}"
        with NoModule("triton"):
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            config = BertConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, config=config
            )
            model.bert.embeddings = LoRAModule(
                model.bert.embeddings, lora_rank, lora_alpha, lora_dropout
            )
            model.bert.encoder = LoRAModule(
                model.bert.encoder, lora_rank, lora_alpha, lora_dropout
            )

        super().__init__(tokenizer, model)


class MistralDNALoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, num_labels=num_labels
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model = LoRAModule(model.model, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)


class GENALMLoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        gena_module_name = model_base.__class__.__module__
        cls = getattr(
            importlib.import_module(gena_module_name), "BertForSequenceClassification"
        )
        model = cls.from_pretrained(model_name, num_labels=num_labels)

        model.bert.embeddings = LoRAModule(
            model.bert.embeddings, lora_rank, lora_alpha, lora_dropout
        )
        model.bert.encoder = LoRAModule(
            model.bert.encoder, lora_rank, lora_alpha, lora_dropout
        )

        super().__init__(tokenizer, model)


class NucleotideTransformerLoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, num_labels=num_labels
        )
        model.esm = LoRAModule(model.esm, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None


class HyenaDNALoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, num_labels=num_labels
        )
        model.hyena = LoRAModule(model.hyena, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None

    def forward(self, seqs):
        tokens, _ = self._tokenize(seqs)

        torch_outs = self.model(tokens)
        logits = torch_outs.logits

        return logits


class CaduceusLoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"kuleshov-group/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, num_labels=num_labels
        )
        model.caduceus = LoRAModule(model.caduceus, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None

    def forward(self, seqs):
        tokens, _ = self._tokenize(seqs)

        torch_outs = self.model(tokens)
        logits = torch_outs.logits

        return logits


class LargeCNNClassifier(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        n_filters,
        n_residual_convs,
        output_channels,
        seq_len,
        pos_channels=1,
        first_kernel_size=21,
        residual_kernel_size=3,
    ):
        super().__init__()
        self.n_residual_convs = n_residual_convs
        self.iconv = torch.nn.Conv1d(
            input_channels, n_filters, kernel_size=first_kernel_size
        )
        self.irelu = torch.nn.ReLU()

        self.pos_emb = torch.nn.Parameter(torch.zeros(seq_len, pos_channels))
        self.pos_proj = torch.nn.Linear(pos_channels, n_filters)

        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=residual_kernel_size,
                    dilation=2**i,
                )
                for i in range(n_residual_convs)
            ]
        )
        self.rrelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(n_residual_convs)]
        )
        self.output_layer = torch.nn.Linear(n_filters, output_channels)

        device_indicator = torch.empty(0)
        self.register_buffer("device_indicator", device_indicator)

    @property
    def device(self):
        return self.device_indicator.device

    def forward(self, x):
        x = x.to(self.device).float()

        x = x.swapaxes(1, 2)
        x = self.iconv(x)
        x = x.swapaxes(1, 2)
        p = self.pos_proj(self.pos_emb)
        x = self.irelu(x + p)
        x = x.swapaxes(1, 2)

        # x = self.irelu(self.iconv(x))

        for i in range(self.n_residual_convs):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            crop_amount = (x.shape[-1] - x_conv.shape[-1]) // 2
            x_cropped = x[:, :, crop_amount:-crop_amount]
            x = torch.add(x_cropped, x_conv)

        x = torch.mean(x, dim=-1)

        final_out = self.output_layer(x)

        return final_out
