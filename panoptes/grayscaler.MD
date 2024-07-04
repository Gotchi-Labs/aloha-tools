# Exoplanet Detection Image Processing Script

## Overview

This script is designed to process FITS (Flexible Image Transport System) files of astronomical images to enhance the visibility of potential exoplanet transits. By applying various image processing techniques, the script converts FITS files to grayscale PNG images and generates metadata for each processed image. Additionally, it ensures data integrity through the creation of a Merkle tree based on the image file hashes. This documentation provides a comprehensive explanation of the script's functionalities, the scientific rationale behind each processing step, and how to configure and use the script.

## Table of Contents

1. [Script Features](#script-features)
2. [Configuration](#configuration)
3. [Image Processing Techniques](#image-processing-techniques)
4. [Merkle Tree for Data Integrity](#merkle-tree-for-data-integrity)
5. [Usage](#usage)
6. [Scientific Justification](#scientific-justification)

## Script Features

- **FITS File Processing**: Converts FITS files to grayscale PNG images.
- **Image Processing Techniques**: Applies logarithmic scaling, intensity clipping, and CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image features.
- **Metadata Generation**: Creates a JSON metadata file for each processed image, including processing details.
- **Data Integrity**: Utilizes a Merkle tree to ensure the integrity of processed image files.
- **Colored Logging**: Provides detailed logging with color-coded messages for different stages of the process.

## Configuration

The script uses a configuration dictionary to manage various parameters for image processing. The configuration can be adjusted to suit different datasets or processing requirements.

```python
config = {
    'lower_percentile': 5,       # Lower percentile for intensity clipping
    'upper_percentile': 95,      # Upper percentile for intensity clipping
    'clahe_clip_limit': 0.01,    # CLAHE clip limit
    'log_scale': 1000,           # Scaling factor for logarithmic adjustment
    'num_images': 0,             # Number of images to process; 0 means process all images
    'apply_log_scale': True,     # Apply logarithmic scaling
    'apply_clipping': True,      # Apply intensity clipping
    'apply_clahe': True          # Apply CLAHE
}
```

## Image Processing Techniques

### Logarithmic Scaling

**Purpose**: Enhances features in regions with varying brightness levels, making subtle changes in light intensity more detectable.

**Process**: The script applies a logarithmic transformation to the pixel values of the image data, which compresses the dynamic range and amplifies low-intensity regions.

**Implementation**:
```python
if config['apply_log_scale']:
    image_data = np.log1p(image_data * config['log_scale'])
    logging.info(Fore.GREEN + "Logarithmic scaling applied.")
```

### Intensity Clipping

**Purpose**: Normalizes the image data by removing outliers, improving contrast to better identify variations in light intensity.

**Process**: The script clips the pixel values to the specified lower and upper percentiles, ensuring that the majority of pixel values fall within a manageable range.

**Implementation**:
```python
if config['apply_clipping']:
    vmin, vmax = np.percentile(image_data, [config['lower_percentile'], config['upper_percentile']])
    image_data = np.clip(image_data, vmin, vmax)
    logging.info(Fore.GREEN + "Intensity clipping applied.")
```

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Purpose**: Enhances local contrast, making small variations in light intensity more noticeable, which is crucial for detecting potential exoplanet transits.

**Process**: The script applies CLAHE to the normalized image data, which improves the visibility of local details by equalizing the histogram within small regions of the image.

**Implementation**:
```python
if config['apply_clahe']:
    processed_image = exposure.equalize_adapthist(processed_image, clip_limit=config['clahe_clip_limit'])
    logging.info(Fore.GREEN + "CLAHE applied.")
```

## Merkle Tree for Data Integrity

**Purpose**: Ensures the integrity and authenticity of the processed image files by generating a root hash that represents the entire dataset.

**Process**: The script generates a SHA-256 hash for each processed image file. These hashes are then combined in pairs to form a Merkle tree, with the final root hash representing the dataset.

**Implementation**:
```python
def create_merkle_root(hashes):
    if len(hashes) == 1:
        return hashes[0]
    new_hashes = []
    for i in range(0, len(hashes), 2):
        combined_hash = hash_file_name(hashes[i] + (hashes[i + 1] if i + 1 < len(hashes) else ''))
        new_hashes.append(combined_hash)
    return create_merkle_root(new_hashes)
```

## Usage

### Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `astropy`, `matplotlib`, `scikit-image`, `colorama`

### Running the Script

1. **Set up the configuration**: Adjust the `config` dictionary as needed.
2. **Place FITS files**: Ensure your FITS files are located in the `images` directory.
3. **Execute the script**: Run the script using the following command:
   ```bash
   python <script_name>.py
   ```

### Example Output

- **Grayscale Images**: Saved in the `processed_images` directory with a unique folder for each image.
- **Metadata JSON**: A metadata file is generated for each image, containing processing details and settings.
- **Merkle Root**: The root hash of the Merkle tree, representing the integrity of the dataset, is logged.

## Scientific Justification

### Why These Techniques Are Useful for Finding Exoplanets

1. **Logarithmic Scaling**: This technique enhances the visibility of subtle changes in brightness, which is crucial when trying to detect dips in light caused by exoplanets passing in front of their host stars.

2. **Intensity Clipping**: By normalizing the intensity values, this method improves the overall contrast of the image, making it easier to identify variations in brightness indicative of potential exoplanet transits.

3. **CLAHE**: This technique enhances local contrast and brings out small variations in light intensity, such as the slight dimming of a star during a transit event. It is particularly effective in highlighting features that might be lost in the global brightness of the image.

### Applications in Exoplanet Detection

- **Transit Method**: Detecting the slight dimming of a star as an exoplanet passes in front of it requires high contrast and visibility of subtle changes in brightness. The image processing techniques used in this script improve the ability to observe these minute variations.
- **Direct Imaging**: Enhancing the contrast and visibility of small variations in brightness against the bright background of a star helps in detecting potential transits more effectively.

The script is intended to significantly enhances the quality and usability of astronomical images for exoplanet detection. The combination of logarithmic scaling, intensity clipping, and CLAHE provides a powerful toolkit for astronomers seeking to uncover the hidden details in their observational data. We will keep working with known exoplanet data to find the best settings if they are available to us in these libraries. 