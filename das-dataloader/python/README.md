# DAS Dataset Loader

The `data_loader.py` file contains the `DASDataLoader` class designed for parsing `HDF5` data files from the OptaSense ODH interrogator. This class helps in loading, processing, and preparing the data for further analysis or model training.

## Description

The `DASDataLoader` class takes the following main arguments:

- **data_dir**: Path to the main dataset directory containing subdirectories. Each subdirectory name corresponds to a label name.
- **sample_len**: Length of the sample window.
- **transform**: A preprocessing method applied to each signal window from the dataset.
- **fsize**: Size of the window in samples.
- **shift**: Overlap between windows in samples.
- **drop_noise**: Flag to indicate whether to drop noisy samples.
- **decimate**: Dictionary specifying the decimation factor for each label.

Each label directory contains data measurement files ending with `*.h5`, which include raw data, and bitmap files ending with `*.npy`, which specify which raw data windows correspond to the label.

Due to the potential size of the dataset, it might be too large for a regular PC to handle. Therefore, the `decimate` parameter allows you to specify how many samples to decimate from each dataset label, reducing the overall data size.

## Usage Example

Here is an example of how to use the `DASDataLoader`:

```python
import numpy as np
from data_loader import DASDataLoader

def fft(x):
    x = np.fft.rfft(x)[:, 1:]
    x = np.abs(x) + 1
    x = np.log10(x)
    return x

decim_dict = {
    'regular': 50,
}

parser = DASDataLoader(
    data_dir='data',
    sample_len=2048,
    transform=fft,
    fsize=8192,
    shift=2048,
    decimate=decim_dict,
)

x, y = parser.parse_dataset()

print(f'The dataset contains {len(x)} elements')
```

## Dependencies

Ensure you have the following packages installed:

- `scikit-learn`
- `h5py`
- `numpy`

You can install these dependencies using pip:

```bash

pip install scikit-learn h5py numpy
```

## Key Features

- **Multi-threaded Processing:** Utilizes multiprocessing to handle large datasets efficiently.
- **FFT Transformation:** Converts time-domain signals to frequency-domain using FFT.
- **Noise Filtering:** Optionally filters out noisy data samples.
- **Label Encoding:** Supports both label binarization and class weight computation.
- **Window Generation:** Extracts overlapping windows from the raw data for model training.

## Logging

The script uses the logging module to provide detailed information about the data loading and preprocessing steps. Logs include information about the dataset structure, number of samples, and class distribution.

## License

This project is licensed under the GNU General Public License v3.0.
