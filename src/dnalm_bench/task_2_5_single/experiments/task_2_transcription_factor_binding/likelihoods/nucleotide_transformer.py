import os

from ....components import FootprintingDataset
from ....evaluators import NTEvaluator

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "nucleotide-transformer-v2-500m-multi-species"
    seq_table = os.path.join(
        root_output_dir,
        "task_2_footprinting/processed_data/footprint_dataset_350_v1.txt",
    )
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"

    out_dir = os.path.join(root_output_dir, "task_2_footprinting/outputs/likelihoods/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        root_output_dir, f"task_2_footprinting/outputs/likelihoods/{model_name}.tsv"
    )

    dataset = FootprintingDataset(seq_table, seed)
    evaluator = NTEvaluator(model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)
