import os
import numpy as np
from astropy.io import fits
import cv2
import json
import hashlib
import logging
from datetime import datetime
from time import time
from colorama import Fore, Style, init
from tqdm import tqdm

# Initialize colorama for colored console output
init(autoreset=True)

# ASCII Art for "PANOPTES"
PANOPTES_ASCII = """
  _____   _____   _   _  _____   _____  _______  ______  ______
 |_____] |_____| | \  | |     | |_____]    |    |______ |_____ 
 |       |     | |  \_| |_____| |          |    |______ ______|
"""

# Custom logging configuration
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    grey = Style.DIM + Fore.WHITE
    green = Style.BRIGHT + Fore.GREEN
    blue = Style.BRIGHT + Fore.BLUE
    red = Style.BRIGHT + Fore.RED

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.grey)
        formatter = logging.Formatter(log_fmt + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL)
        return formatter.format(record)

    FORMATS = {
        logging.DEBUG: blue,
        logging.INFO: grey,
        logging.WARNING: blue,
        logging.ERROR: red,
        logging.CRITICAL: red
    }

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.handlers[0].setFormatter(CustomFormatter())  # Update default handler

def calculate_hash(image):
    """
    Calculates the SHA-256 hash of an image.

    Parameters:
    image (numpy.ndarray): Image data.

    Returns:
    str: SHA-256 hash of the image.
    """
    hash_obj = hashlib.sha256(image.tobytes())
    return hash_obj.hexdigest()

def fits_to_tiles(fits_file, output_dir, tile_size=500):
    """
    Processes a FITS file into 500x500 pixel tiles, saves each tile as a separate image,
    and generates a JSON file with metadata for each tile.

    Parameters:
    fits_file (str): Path to the input FITS file.
    output_dir (str): Directory where the output tiles will be saved.
    tile_size (int): Size of the tiles (default is 500x500 pixels).

    Returns:
    None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            # Get image data from the primary HDU
            image_data = hdul[0].data
            if image_data is None:
                logger.error(f"No image data found in FITS file: {fits_file}")
                return
    except Exception as e:
        logger.error(f"Error opening FITS file: {e}")
        return

    # Ensure the image is in 2D format
    if len(image_data.shape) != 2:
        logger.error("Error: FITS file does not contain 2D image data")
        return

    # Get the dimensions of the image
    height, width = image_data.shape

    # Calculate the number of tiles
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # Initialize metadata dictionary
    metadata = {
        "fits_file": os.path.basename(fits_file),
        "tile_size": tile_size,
        "tiles": []
    }

    # Initialize time tracking
    start_time = time()

    # Generate and save the tiles
    total_tiles = num_tiles_x * num_tiles_y
    for i in tqdm(range(num_tiles_y), desc="Processing tiles", unit="tile"):
        for j in range(num_tiles_x):
            # Extract the tile
            tile = image_data[i * tile_size:(i + 1) * tile_size,
                              j * tile_size:(j + 1) * tile_size]

            # Normalize the tile data to the range [0, 65535] for 16-bit
            tile_min = np.min(tile)
            tile_max = np.max(tile)
            if tile_max == tile_min:
                logger.warning(f"Tile ({i}, {j}) has uniform value")
                normalized_tile = np.zeros((tile_size, tile_size), dtype=np.uint16)
            else:
                normalized_tile = ((tile - tile_min) / (tile_max - tile_min) * 65535).astype(np.uint16)

            # Calculate hash for the tile
            tile_hash = calculate_hash(normalized_tile)

            # Create unique tile ID and filename
            tile_id = f"{os.path.basename(fits_file)}_tile_{i}_{j}"
            tile_filename = os.path.join(output_dir, f"{tile_id}.png")
            
            # Save the tile image
            try:
                cv2.imwrite(tile_filename, normalized_tile)
            except Exception as e:
                logger.error(f"Error saving tile image {tile_filename}: {e}")

            # Add metadata for the tile
            tile_metadata = {
                "tile_id": tile_id,
                "filename": os.path.basename(tile_filename),
                "hash": tile_hash,
                "position": {
                    "x_start": j * tile_size,
                    "y_start": i * tile_size,
                    "x_end": (j + 1) * tile_size,
                    "y_end": (i + 1) * tile_size
                }
            }
            metadata["tiles"].append(tile_metadata)

    # Calculate total elapsed time
    elapsed_time = time() - start_time
    avg_time_per_tile = elapsed_time / total_tiles
    logger.info(f"Completed processing {total_tiles} tiles in {elapsed_time:.2f}s (avg {avg_time_per_tile:.4f}s per tile)")

    # Save metadata to JSON file
    json_file = os.path.join(output_dir, f'{os.path.basename(fits_file)}_metadata.json')
    try:
        with open(json_file, 'w') as json_f:
            json.dump(metadata, json_f, indent=4)
        logger.info(f"Metadata saved to {json_file}")
    except Exception as e:
        logger.error(f"Error saving metadata to JSON file: {e}")

def find_duplicates(metadata_dir):
    """
    Finds and logs duplicate tiles by their hash from metadata JSON files.

    Parameters:
    metadata_dir (str): Path to the directory containing metadata JSON files.

    Returns:
    None
    """
    hash_dict = {}
    duplicates = []

    for subdir, _, files in os.walk(metadata_dir):
        for file in files:
            if file.lower().endswith('_metadata.json'):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r') as json_file:
                        metadata = json.load(json_file)
                        for tile in metadata.get("tiles", []):
                            tile_hash = tile["hash"]
                            tile_id = tile["tile_id"]
                            if tile_hash in hash_dict:
                                duplicates.append((tile_id, tile["filename"], hash_dict[tile_hash]["tile_id"], hash_dict[tile_hash]["filename"]))
                            else:
                                hash_dict[tile_hash] = {"tile_id": tile_id, "filename": tile["filename"]}
                except Exception as e:
                    logger.error(f"Error reading metadata file {file_path}: {e}")

    if duplicates:
        logger.info(Fore.RED + f"Found {len(duplicates)} duplicate tiles:")
        for duplicate in duplicates:
            logger.info(Fore.RED + f"Duplicate found: {duplicate[0]} ({duplicate[1]}) and {duplicate[2]} ({duplicate[3]})")
    else:
        logger.info(Fore.GREEN + "No duplicate tiles found.")

def main():
    print(PANOPTES_ASCII)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_directory = os.path.join(script_directory, 'fits_images')
    output_directory = os.path.join(script_directory, 'output_tiles')

    # Find all .fits and .fit files
    fits_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.lower().endswith(('.fits', '.fit'))]

    if not fits_files:
        logger.info(Fore.RED + "No FITS files found in the directory. Exiting.")
        return

    logger.info(Fore.BLUE + "Initiating FITS image tiling process...")
    logger.info(Fore.BLUE + f"Found {len(fits_files)} FITS files in directory.")

    for fits_file in fits_files:
        logger.info(Fore.BLUE + f"Processing {fits_file}...")

        file_base_name = os.path.splitext(os.path.basename(fits_file))[0]
        file_output_dir = os.path.join(output_directory, file_base_name)

        fits_to_tiles(fits_file, file_output_dir)

    logger.info(Fore.BLUE + "Checking for duplicate tiles...")
    find_duplicates(output_directory)

if __name__ == '__main__':
    main()
