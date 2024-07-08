# FITS Image Reconstruction from Tiles Script

## Table of Contents

- [FITS Image Reconstruction from Tiles Script](#fits-image-reconstruction-from-tiles-script)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Directory Structure](#directory-structure)
    - [Running the Script](#running-the-script)
    - [Script Output](#script-output)
  - [Script Details](#script-details)
    - [Main Functions](#main-functions)
  - [Conclusion](#conclusion)

## Overview

This project provides a Python script designed to reconstruct a full image from its tiles using metadata stored in a JSON file. This script is particularly useful for reassembling images that have been previously split into smaller tiles for detailed analysis or storage efficiency. The script reads the metadata to determine the correct placement of each tile and reconstructs the original image.

## Features

1. **Image Reconstruction from Tiles**:
   - Reads metadata from a JSON file to reconstruct the full image.
   - Uses OpenCV to read and combine tile images into a single image.
   - Saves the reconstructed image in the specified output directory.

2. **Custom Logging**:
   - Provides informative logging throughout the process, including errors and progress updates.
   - Utilizes color-coded console output for enhanced readability.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `opencv-python-headless`
  - `tqdm`
  - `colorama`

You can install the required packages using the following command:

```bash
pip install numpy opencv-python-headless tqdm colorama
```

## Usage

### Directory Structure

Ensure you have the following directory structure:

```
project_root/
│
├── output_tiles/
│   └── (place your metadata JSON files and tile images here)
│
├── reconstructed_images/
│   └── (reconstructed images will be saved here)
│
└── imageReconstructor.py
```

### Running the Script

1. **Place Metadata Files and Tile Images**:

   - Place your metadata JSON files and corresponding tile images in the `output_tiles` directory.

2. **Execute the Script**:

   - Run the script from the project root directory.

   ```bash
   python imageReconstructor.py
   ```

3. **Script Execution**:
   - The script will process all metadata JSON files in the `output_tiles` directory, reconstruct the images, and save them in the `reconstructed_images` directory.

### Script Output

1. **Reconstructed Images**:

   - Reconstructed images will be saved in the `reconstructed_images` directory with filenames prefixed by `reconstructed_`.

2. **Logging**:
   - The script will log the progress and any errors encountered during the process in the console.

## Script Details

### Main Functions

- **reconstruct_image_from_tiles(metadata_file, output_dir)**:
  - Reconstructs an image from its tiles using metadata from the JSON file.
  - Saves the reconstructed image in the specified output directory.

- **main()**:
  - Orchestrates the entire process: reading metadata files, reconstructing images from tiles, and saving the reconstructed images.

## Conclusion

This script is a useful tool for astronomers and researchers who need to reconstruct full images from tiled datasets. By leveraging detailed metadata, the script ensures accurate reassembly of the original images. The custom logging provides clear and informative feedback throughout the process, making it easier to identify and address any issues that may arise.

For further assistance or inquiries, please refer to the contact information provided in the project's repository.
