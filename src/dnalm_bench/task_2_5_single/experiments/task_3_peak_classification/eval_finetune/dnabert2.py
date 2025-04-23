import os
import sys

import numpy as np
import pandas as pd
import torch

from ....finetune import (
    DNABERT2LoRAModel,
    PeaksEndToEndDataset,
    eval_finetuned_peak_classifier,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    eval_mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    model_name = "DNABERT-2-117M"

    genome_fa = os.path.join(
        work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    elements_tsv = os.path.join(
        work_dir,
        "task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv",
    )

    batch_size = 128
    num_workers = 4
    prefetch_factor = 2
    seed = 0
    device = "cuda"

    chroms_train = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY",
    ]

    chroms_val = ["chr6", "chr21"]

    chroms_test = ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"]

    modes = {"train": chroms_train, "val": chroms_val, "test": chroms_test}

    emb_channels = 768

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    accumulate = 1

    lr = 1e-4
    wd = 0.01
    num_epochs = 10

    out_dir = os.path.join(
        work_dir,
        f"task_3_peak_classification/supervised_model_outputs/fine_tuned/{model_name}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    model_dir = os.path.join(
        work_dir,
        f"task_3_peak_classification/supervised_models/fine_tuned/{model_name}",
    )
    train_log = f"{model_dir}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")

    classes = {"GM12878": 0, "H1ESC": 1, "HEPG2": 2, "IMR90": 3, "K562": 4}

    test_dataset = PeaksEndToEndDataset(
        genome_fa, elements_tsv, modes[eval_mode], classes, cache_dir=cache_dir
    )

    model = DNABERT2LoRAModel(
        model_name, lora_rank, lora_alpha, lora_dropout, len(classes)
    )
    checkpoint_resume = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_resume, strict=False)

    metrics = eval_finetuned_peak_classifier(
        test_dataset,
        model,
        out_path,
        batch_size,
        num_workers,
        prefetch_factor,
        device,
        progress_bar=True,
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")
