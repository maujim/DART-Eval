import os
import sys

from torch.utils.data import DataLoader

from ....training import (
    PeaksEmbeddingsDataset,
    CNNEmbeddingsPredictor,
    train_peak_classifier,
)

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

    peaks_h5 = os.path.join(
        root_output_dir, f"task_3_peak_classification/embeddings/{model_name}.h5"
    )
    elements_tsv = os.path.join(
        root_output_dir,
        "task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv",
    )

    batch_size = 256
    num_workers = 0
    prefetch_factor = None
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

    input_channels = 512
    hidden_channels = 32
    kernel_size = 3

    crop = 557

    lr = 2e-3
    num_epochs = 150

    out_dir = os.path.join(
        root_output_dir,
        f"task_3_peak_classification/supervised_models/probed/{model_name}",
    )
    os.makedirs(out_dir, exist_ok=True)

    classes = {"GM12878": 0, "H1ESC": 1, "HEPG2": 2, "IMR90": 3, "K562": 4}

    train_dataset = PeaksEmbeddingsDataset(
        peaks_h5, elements_tsv, chroms_train, classes
    )
    val_dataset = PeaksEmbeddingsDataset(peaks_h5, elements_tsv, chroms_val, classes)

    model = CNNEmbeddingsPredictor(
        input_channels, hidden_channels, kernel_size, out_channels=len(classes)
    )
    train_peak_classifier(
        train_dataset,
        val_dataset,
        model,
        num_epochs,
        out_dir,
        batch_size,
        lr,
        num_workers,
        prefetch_factor,
        device,
        progress_bar=True,
        resume_from=resume_checkpoint,
    )
