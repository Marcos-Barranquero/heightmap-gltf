from PIL import Image
import numpy as np


# heights must be a 2D numpy array of integers
def create_heightmap(heights: np.ndarray):
  # Fail if negative heights are found
  if np.any(heights < 0):
    raise ValueError("Negative heights found!")
  # Fail if heights are not integers
  if not np.issubdtype(heights.dtype, np.integer):
    raise TypeError("Heights must be integers!")
  # Fail if heights are not 2D
  if len(heights.shape) != 2:
    raise ValueError("Heights must be 2D!")

  # Get the maximum height
  max_height = np.max(heights)
  # Normalize heights to a 0-65535 range
  normalized_heights = (np.array(heights) / max_height) * 65535
  # Create a grayscale image from the normalized heights
  heightmap = Image.fromarray(normalized_heights.astype(np.uint16), 'I;16')
  return heightmap

def image_to_heights(image):
  parsed =  np.array(image, dtype=np.float32)
  if len(parsed.shape) != 2:
    raise ValueError("Image must be 2D!")
  return parsed
