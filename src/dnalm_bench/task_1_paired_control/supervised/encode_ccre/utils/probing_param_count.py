

from ...training import (
    CNNSlicedEmbeddingsClassifier,
)

if __name__ == "__main__":

    input_channels = 256
    hidden_channels = 32
    kernel_size = 8

    model = CNNSlicedEmbeddingsClassifier(input_channels, hidden_channels, kernel_size)

    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
