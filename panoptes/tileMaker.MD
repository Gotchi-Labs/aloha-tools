# FITS Image Tiling and Duplicate Detection Script

## Table of Contents

- [FITS Image Tiling and Duplicate Detection Script](#fits-image-tiling-and-duplicate-detection-script)
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
  - [Scientific Approach](#scientific-approach)
    - [Data Management and Reduction](#data-management-and-reduction)
    - [Integrity Preservation](#integrity-preservation)
    - [Duplicate Detection](#duplicate-detection)
  - [Using AI for Astronomical Data](#using-ai-for-astronomical-data)
    - [Training AI with Metadata](#training-ai-with-metadata)
    - [Star JSON Format for AI](#star-json-format-for-ai)
    - [Future Work](#future-work)
  - [Example Output](#example-output)
    - [Tile Image](#tile-image)
    - [Metadata JSON](#metadata-json)
  - [Conclusion](#conclusion)

## Overview

This project provides a Python script designed to process FITS (Flexible Image Transport System) files, commonly used in astronomy, into smaller image tiles for detailed analysis. The script also includes functionality to detect duplicate tiles across multiple FITS files based on their content. This process aims to improve the manageability of large astronomical datasets and facilitate the detection of phenomena such as exoplanet transits.

## Features

1. **Tiling of FITS Images**:

   - Splits large FITS images into smaller, manageable 500x500 pixel tiles.
   - Each tile is saved as a separate image file.
   - Metadata for each tile is generated and saved in a JSON file, including a unique hash to identify the tile content.

2. **Duplicate Detection**:
   - Scans all generated metadata files to identify and log duplicate tiles based on their hash values.
   - Ensures precise detection of exoplanet transits by preserving all data integrity.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `astropy`
  - `opencv-python-headless`
  - `tqdm`
  - `colorama`

You can install the required packages using the following command:

```bash
pip install numpy astropy opencv-python-headless tqdm colorama
```

## Usage

### Directory Structure

Ensure you have the following directory structure:

```
project_root/
│
├── fits_images/
│   └── (place your .fits and .fit files here)
│
├── output_tiles/
│   └── (generated tiles and metadata will be saved here)
│
└── tileMaker.py
```

### Running the Script

1. **Place FITS Files**:

   - Place your `.fits` and `.fit` files in the `fits_images` directory.

2. **Execute the Script**:

   - Run the script from the project root directory.

   ```bash
   python tileMaker.py
   ```

3. **Script Execution**:
   - The script will process all FITS files in the `fits_images` directory, generate tiles, and save them in the `output_tiles` directory along with metadata JSON files.
   - After processing, the script will check for duplicate tiles across all generated metadata files and log the results.

### Script Output

1. **Tile Images**:

   - Generated tile images will be saved in subdirectories within `output_tiles`, named after the original FITS files.

2. **Metadata Files**:

   - Metadata for each FITS file will be saved as a JSON file in the corresponding subdirectory within `output_tiles`.

3. **Duplicate Detection Log**:
   - The script will log any duplicate tiles found based on their hash values in the console.

## Script Details

### Main Functions

- **calculate_hash(image)**:
  - Calculates the SHA-256 hash of the given image data.
- **fits_to_tiles(fits_file, output_dir, tile_size=500)**:
  - Processes the FITS file into 500x500 pixel tiles.
  - Saves each tile as an image and generates metadata, including the tile's hash.
- **find_duplicates(metadata_dir)**:
  - Scans metadata JSON files in the specified directory.
  - Identifies and logs duplicate tiles based on their hash values.
- **main()**:
  - Orchestrates the entire process: reading FITS files, processing them into tiles, generating metadata, and checking for duplicates.

## Scientific Approach

### Data Management and Reduction

The tiling process divides large FITS files into smaller segments, which can be more easily analyzed and managed. By converting large FITS images into 500x500 pixel tiles, the data becomes more manageable for processing and storage. This segmentation can also aid in focusing analysis on specific regions of the sky, potentially reducing the file sizes and computational load required for processing.

### Integrity Preservation

Each tile is assigned a unique hash value using the SHA-256 algorithm. This ensures that the content of each tile can be uniquely identified and verified. The inclusion of hash values in the metadata allows for the detection of duplicate tiles, which is crucial for maintaining data integrity, especially in astronomical research where detecting subtle changes in light (such as those caused by exoplanet transits) is essential.

### Duplicate Detection

The duplicate detection mechanism scans all metadata JSON files to find tiles with identical hash values. This helps in identifying redundant data, which can be crucial for optimizing storage and ensuring the accuracy of scientific analysis. By logging duplicate tiles, researchers can further investigate and verify the integrity of their datasets.

## Using AI for Astronomical Data

### Training AI with Metadata

The detailed metadata generated by the script, combined with high-resolution images, can theoretically be used to train AI models for various astronomical tasks. These tasks could include:

- **Anomaly Detection**: AI can be trained to detect anomalies in the data, such as unexpected changes in star brightness that might indicate the presence of exoplanets. By identifying these anomalies, researchers can focus their efforts on specific regions of interest in the sky.
- **Star Tracking**: The AI can track the positions, velocities, and other characteristics of stars over time, allowing for the creation of dynamic star maps. This capability would enable researchers to study the movement and evolution of stars with high precision.

### Star JSON Format for AI

To facilitate AI training and star tracking, we are proposing a detailed JSON format similar to the following:

```json
{
  "tile_id": "unique_tile_identifier",
  "timestamp": "2024-07-07T00:00:00Z",
  "stars": [
    {
      "id": "star_1",
      "name": "Sirius",
      "position": { "x": 100, "y": 150 },
      "brightness": 0.85,
      "color_index": 0.65,
      "velocity": { "vx": 0.001, "vy": -0.002 },
      "light_curve": [
        { "timestamp": "2024-07-07T00:00:00Z", "brightness": 0.85 },
        { "timestamp": "2024-07-07T00:00:01.5Z", "brightness": 0.84 }
        // Additional light curve data points
      ],
      "type": "main_sequence",
      "radius": 1.0
    }
    // Additional stars
  ]
}
```

This JSON format provides comprehensive information about each star, including its ID, name, position, brightness, color index, velocity, and light curve data. This information is crucial for accurately recreating star maps and analyzing stellar behavior over time. By including such detailed metadata, we aim to:

- **Accurately Recreate Star Maps**: The more detailed the information, the better the representation of the starmap can be. This allows for a high-fidelity reconstruction of the night sky from the JSON data.
- **Enhance Interactivity**: With comprehensive metadata, the starmap can be made interactive, allowing users to click on individual stars to retrieve detailed information about them. This interactivity can aid in educational purposes and in-depth analysis by researchers.
- **Improve Representation**: Detailed metadata ensures that various attributes of the stars, such as their motion and brightness variations, are captured accurately. This leads to more realistic and informative visualizations.

### Future Work

While the AI component is currently theoretical, we envision using the generated metadata to train AI models that can further enhance our understanding of the cosmos. By leveraging machine learning techniques, we aim to develop tools that can automatically analyze vast amounts of astronomical data, identify patterns, and predict stellar phenomena.

## Example Output

### Tile Image

```
output_tiles/
├── example_fits_file/
│   ├── example_fits_file_tile_0_0.png
│   ├── example_fits_file_tile_0_1.png
│   ├── ...
│   └── example_fits_file_metadata.json
```

### Metadata JSON

```json
{
  "fits_file": "example_fits_file.fits",
  "tile_size": 500,
  "tiles": [
    {
      "tile_id": "example_fits_file_tile_0_0",
      "filename": "example_fits_file_tile_0_0.png",
      "hash": "abcd1234...",
      "position": {
        "x_start": 0,
        "y_start": 0,
        "x_end": 500,
        "y_end": 500
      }
    },
    ...
  ]
}
```

## Conclusion

This script is a powerful tool for astronomers and researchers working with large FITS files. By breaking down the images into smaller tiles and providing a robust duplicate detection mechanism, the script ensures precise analysis of astronomical data, which is crucial for detecting exoplanets and other celestial phenomena. The scientific approach adopted in this script ensures data integrity and efficient management of large datasets, making it suitable for peer-reviewed research and large-scale astronomical studies.

For further assistance or inquiries, please refer to the contact information provided in the project's repository.
