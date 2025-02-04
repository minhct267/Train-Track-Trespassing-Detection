from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
YOUTUBE = 'YouTube'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, YOUTUBE, VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'img_3.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'img_3.png'

# ML Model config
MODEL_DIR = ROOT / 'models'
SEGMENTATION_MODEL = MODEL_DIR / 'model3.pt'
