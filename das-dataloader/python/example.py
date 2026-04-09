import sys
import logging
# Importing necessary modules for data loading and transformation
from data_loader import DASDataLoader, fft

logging.basicConfig(level=logging.INFO)


def main():
    # Dictionary defining decimation factors for different labels
    decim_dict = {
        # The 'regular' label will be decimated by a factor of 50
        'regular': 50,
    }

    # Initializing the DASDataLoader with dataset parameters
    parser = DASDataLoader(
        '../data',  # Path to the dataset directory
        2048,  # Sample length
        transform=fft,  # Applying FFT as a preprocessing step
        fsize=8192,  # Window size for sliding window segmentation
        # Step size for the sliding window (overlap of 75% with fsize=8192)
        shift=2048,
        # Dictionary specifying the decimation factor for each label
        decimate=decim_dict,
    )

    # Parsing the dataset into features (x) and labels (y)
    x, y = parser.parse_dataset()

    # Output parsed dataset details
    print(x, y)
    print(f'The dataset contains {len(x)} elements')

    return 0


if __name__ == "__main__":
    sys.exit(main())
