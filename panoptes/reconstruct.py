import os
import numpy as np
import cv2
import json
import logging
from tqdm import tqdm
from colorama import Fore, Style, init

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

def reconstruct_image_from_tiles(metadata_file, output_dir):
    """
    Reconstructs an image from its tiles using metadata from the JSON file.

    Parameters:
    metadata_file (str): Path to the metadata JSON file.
    output_dir (str): Directory where the reconstructed images will be saved.

    Returns:
    None
    """
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_file}: {e}")
        return

    tile_size = metadata["tile_size"]
    tiles = metadata["tiles"]

    # Determine the dimensions of the full image
    x_coords = [tile["position"]["x_end"] for tile in tiles]
    y_coords = [tile["position"]["y_end"] for tile in tiles]
    width = max(x_coords)
    height = max(y_coords)

    # Create an empty array for the full image
    full_image = np.zeros((height, width), dtype=np.uint16)

    # Place each tile in its correct position
    for tile in tqdm(tiles, desc="Reconstructing image", unit="tile"):
        tile_filename = os.path.join(os.path.dirname(metadata_file), tile["filename"])
        x_start = tile["position"]["x_start"]
        y_start = tile["position"]["y_start"]
        x_end = tile["position"]["x_end"]
        y_end = tile["position"]["y_end"]

        try:
            tile_image = cv2.imread(tile_filename, cv2.IMREAD_UNCHANGED)
            if tile_image is None:
                raise ValueError(f"Tile image {tile_filename} could not be read.")
            full_image[y_start:y_end, x_start:x_end] = tile_image
        except Exception as e:
            logger.error(f"Error reading tile image {tile_filename}: {e}")

    # Save the reconstructed image
    fits_basename = os.path.basename(metadata["fits_file"])
    output_filename = os.path.join(output_dir, f"reconstructed_{fits_basename}.png")
    try:
        cv2.imwrite(output_filename, full_image)
        logger.info(f"Reconstructed image saved to {output_filename}")
    except Exception as e:
        logger.error(f"Error saving reconstructed image {output_filename}: {e}")

def main():
    print(PANOPTES_ASCII)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    metadata_directory = os.path.join(script_directory, 'output_tiles')
    output_directory = os.path.join(script_directory, 'reconstructed_images')

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Find all metadata JSON files
    metadata_files = [os.path.join(subdir, file)
                      for subdir, _, files in os.walk(metadata_directory)
                      for file in files if file.lower().endswith('_metadata.json')]

    if not metadata_files:
        logger.info(Fore.RED + "No metadata JSON files found in the directory. Exiting.")
        return

    logger.info(Fore.BLUE + f"Found {len(metadata_files)} metadata JSON files in directory.")

    for metadata_file in metadata_files:
        logger.info(Fore.BLUE + f"Processing {metadata_file}...")
        reconstruct_image_from_tiles(metadata_file, output_directory)

if __name__ == '__main__':
    main()
