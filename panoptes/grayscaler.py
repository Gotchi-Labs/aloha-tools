import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage import exposure  # For CLAHE
import logging
from colorama import Fore, Style, init
from datetime import datetime
import json
import hashlib

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

config = {
    'lower_percentile': 15,       # Lower percentile for intensity clipping; pixels below this percentile will be clipped.
    'upper_percentile': 90,       # Upper percentile for intensity clipping; pixels above this percentile will be clipped.
    'clahe_clip_limit': 0.05,     # Clip limit for CLAHE; higher values give more contrast.
    'log_scale': 100,             # Scaling factor for logarithmic adjustment; adjusts the compression of dynamic range.
    'num_images': 1,              # Number of images to process; 0 means process all images in the directory.
    'apply_log_scale': True,      # Whether to apply logarithmic scaling to the image data.
    'apply_clipping': True,       # Whether to apply intensity clipping to the image data.
    'apply_clahe': True           # Whether to apply CLAHE to the image data.
}


def apply_image_processing(image_data, config):
    logging.debug("Starting image processing...")
    if config['apply_log_scale']:
        image_data = np.log1p(image_data * config['log_scale'])
        logging.info(Fore.GREEN + "Logarithmic scaling applied.")

    if config['apply_clipping']:
        vmin, vmax = np.percentile(image_data, [config['lower_percentile'], config['upper_percentile']])
        image_data = np.clip(image_data, vmin, vmax)
        logging.info(Fore.GREEN + "Intensity clipping applied.")

    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
    processed_image = image_data.astype(np.uint8)

    if config['apply_clahe']:
        processed_image = exposure.equalize_adapthist(processed_image, clip_limit=config['clahe_clip_limit'])
        logging.info(Fore.GREEN + "CLAHE applied.")

    return processed_image

def save_metadata_json(output_directory, metadata):
    metadata_file_path = os.path.join(output_directory, "metadata.json")
    with open(metadata_file_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    logging.info(Fore.GREEN + "Metadata JSON file saved.")

def hash_file_name(content):
    hasher = hashlib.sha256()
    hasher.update(content.encode())
    return hasher.hexdigest()

def create_merkle_root(hashes):
    if len(hashes) == 1:
        return hashes[0]
    new_hashes = []
    for i in range(0, len(hashes), 2):
        combined_hash = hash_file_name(hashes[i] + (hashes[i + 1] if i + 1 < len(hashes) else ''))
        new_hashes.append(combined_hash)
    return create_merkle_root(new_hashes)

def process_fits_to_grayscale(fits_files, output_directory, config):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    logging.info(Fore.BLUE + "Output directory created at: " + output_directory)

    num_images = config['num_images'] if config['num_images'] > 0 else len(fits_files)
    logging.info(Fore.BLUE + f"Configured to process {num_images} images.")

    hashes = []

    for i, fits_file in enumerate(fits_files[:num_images]):
        logging.debug(Fore.BLUE + f"Opening FITS file {i+1}/{num_images}: {fits_file}")
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data
            if image_data is not None:
                processed_data = apply_image_processing(image_data, config)
                image_hash = hash_file_name(processed_data.tobytes().decode('ISO-8859-1'))
                image_folder = os.path.join(output_directory, datetime.now().strftime('%Y%m%d_%H%M%S_') + image_hash)
                os.makedirs(image_folder, exist_ok=True)
                image_path = os.path.join(image_folder, image_hash + ".png")
                plt.imsave(image_path, processed_data, cmap='gray')
                logging.info(Fore.GREEN + f"Grayscale image saved: {image_path}")

                metadata = {
                    "processed_file": os.path.basename(fits_file),
                    "image_path": os.path.basename(image_path),
                    "processing_details": {
                        "time": datetime.now().isoformat(),
                        "settings": config
                    }
                }
                save_metadata_json(image_folder, metadata)
                hashes.append(image_hash)
            else:
                logging.error(Fore.RED + f"No data found in FITS file {fits_file}.")

    merkle_root = create_merkle_root(hashes)
    logging.info(Fore.GREEN + f"Merkle Root: {merkle_root}")

def main():
    print(PANOPTES_ASCII)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_directory = os.path.join(script_directory, 'images')
    output_directory = os.path.join(script_directory, 'processed_images')
    
    fits_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.FIT')]
    logging.info(Fore.BLUE + "Initiating exoplanet detection preprocessing sequence...")
    logging.info(Fore.BLUE + f"Found {len(fits_files)} FITS files in directory. Preparing for detailed analysis.")
    process_fits_to_grayscale(fits_files, output_directory, config)

if __name__ == '__main__':
    main()
