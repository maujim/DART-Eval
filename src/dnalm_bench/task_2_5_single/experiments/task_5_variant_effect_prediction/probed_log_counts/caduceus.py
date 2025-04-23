import os
import sys

import numpy as np
import pandas as pd
import polars as pl

from ....components import VariantDataset
from ....evaluators import CaduceusProbingVariantEvaluator
from ....training import CNNEmbeddingsPredictorBase

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

    batch_size = 256
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms = None

    variants_bed = sys.argv[1]
    output_prefix = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = "GM12878"

    model_folder = os.path.join(
        root_output_dir,
        f"task_4_chromatin_activity/supervised_models/probed/{model_name}/{cell_line}/v1",
    )
    train_log = f"{model_folder}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])

    model_path = f"{model_folder}/checkpoint_{checkpoint_num}.pt"

    out_dir = os.path.join(
        root_output_dir, f"task_5_variant_effect_prediction/outputs/probed/{model_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    input_channels = 512
    hidden_channels = 32
    kernel_size = 3

    print(output_prefix)
    out_path = os.path.join(out_dir, f"{output_prefix}" + ".tsv")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    model = CNNEmbeddingsPredictorBase(input_channels, hidden_channels, kernel_size)
    evaluator = CaduceusProbingVariantEvaluator(
        model, model_path, model_name, batch_size, num_workers, device
    )
    counts_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, counts_df], how="horizontal")
    scored_df.write_csv(out_path, separator="\t")
