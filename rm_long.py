import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import glob
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image(image_path):
    """
    Process a single image: check its dimensions and delete if width > 1400
    Returns tuple of (path, width, height, deleted)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return image_path, None, None, False
        
        height, width = img.shape[:2]
        
        if width > 1400:
            img = None
            os.remove(image_path)
            logger.info(f"Deleted {image_path} (width: {width}, height: {height})")
            return image_path, width, height, True
        
        return image_path, width, height, False
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return image_path, None, None, False

def main():
    # Set your directory path here
    input_dir = os.path.expanduser("~/classifier_data/nogoal_crop_refined2") # Change this to your directory path
    max_workers = min(32, os.cpu_count() * 2)
    
    if not os.path.exists(input_dir):
        logger.error(f"Directory does not exist: {input_dir}")
        return
    
    # Get list of image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(ext))
        image_files.extend(Path(input_dir).glob(ext.upper()))
    
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Statistics tracking
    stats = {
        'total': len(image_files),
        'processed': 0,
        'deleted': 0,
        'errors': 0,
        'width_stats': []
    }
    
    # Process images using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, img_path) for img_path in image_files]
        
        for future in tqdm(futures, total=len(image_files), desc="Processing images"):
            try:
                path, width, height, deleted = future.result()
                stats['processed'] += 1
                
                if deleted:
                    stats['deleted'] += 1
                elif width is not None:
                    stats['width_stats'].append(width)
                else:
                    stats['errors'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}")
                stats['errors'] += 1
    
    # Print final statistics
    logger.info("\nProcessing complete!")
    logger.info(f"Total images processed: {stats['processed']}")
    logger.info(f"Images deleted: {stats['deleted']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    
    if stats['width_stats']:
        avg_width = sum(stats['width_stats']) / len(stats['width_stats'])
        max_width = max(stats['width_stats'])
        min_width = min(stats['width_stats'])
        logger.info(f"Average width of remaining images: {avg_width:.2f}")
        logger.info(f"Min width: {min_width}")
        logger.info(f"Max width: {max_width}")

if __name__ == "__main__":
    main()