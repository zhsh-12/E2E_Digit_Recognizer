import re
from common.config import image_extensions

def get_image_files(img_path):
    """Return a list of image file Path objects from the input_imgs directory"""
    return [f for f in img_path.iterdir() if f.suffix.lower() in image_extensions]

def detect_label_mode(image_files) -> bool:
    """
    Auto-detect whether image filenames contain [number] formatted true labels.
    
    Returns:
        True: Has true labels (labeled mode)
        False: No true labels (unlabeled mode)
    """
    sample_size = min(len(image_files), 5)  # Check the first 5 images
    for f in image_files[:sample_size]:
        if re.search(r'\[(\d+)\]', f.name):
            return True
    return False

def extract_true_label(filename: str) -> int:
    """Extract the true label from filename, e.g. img_40[2].png -> 2"""
    match = re.search(r'\[(\d+)\]', filename)
    if not match:
        raise ValueError(f"No label pattern [number] found in filename: {filename}")
    return int(match.group(1))






