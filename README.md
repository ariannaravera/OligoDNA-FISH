# OligoDNA-FISH

## Overview
This script performs 3D nuclei and fluorescent spot segmentation using Cellpose and OpenCV-based image processing techniques. It generates masks for cells and loci, visualizes results, and extracts spatial information about loci inside cells.

## Features
- **Cell Segmentation:** Uses Cellpose to generate cell masks from 3D images.
- **Spot Segmentation:** Applies image thresholding and filtering to detect loci.
- **Visualization:** Combines segmented channels into an RGB image for inspection.
- **Locus Analysis:** Extracts and analyzes spatial distribution of loci within segmented cells.

## Dependencies
- Python 3.12.3
- OpenCV (`cv2`)
- NumPy
- tifffile
- scikit-image (`skimage`)
- Cellpose
- Warnings module (to suppress unnecessary output)

Install dependencies using:
```bash
pip install opencv-python numpy tifffile scikit-image cellpose
```

## Usage
Define all the required variables as explained in the script itself:
```python
if __name__ == '__main__':
    ######################################################################################
    ## REQUIRED VARIABLES ##
    ...
    input_path = 'data-folder-path'
    output_path = 'result-folder-path'
    ...
    cellpose_model_path = 'retrained-cellpose-model-path'
```
- `input_path`: Your images directory
- `output_path`: Directory to save results
- `cellpose_model_path`: Path to a custom Cellpose model (optional)



## Notes
- To adjust spot segmentation threshold, modify the `cv2.threshold` value in `mask_spots()`.
- Adjust Cellpose model parameters to optimize segmentation results.

## Author
Arianna Ravera
