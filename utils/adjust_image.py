import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

def adjust_image(img, shadows=0, highlights=0, brilliance=0, exposure=0, 
                 contrast=0, brightness=0, black_point=0, sharpness=0, noise_reduction=0):
    # Check if img is a numpy array (OpenCV image)
    if isinstance(img, np.ndarray):
        # Convert to PIL Image
        if len(img.shape) == 2:  # Grayscale
            img = Image.fromarray(img, mode='L')
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported image format")
    elif isinstance(img, str):
        # If it's a string (file path), open with PIL
        img = Image.open(img)
    elif not isinstance(img, Image.Image):
        raise ValueError("Input must be a NumPy array, file path, or PIL Image object")
    
    # Store original mode
    original_mode = img.mode
    
    # Convert image to numpy array
    img_array = np.array(img).astype(float)
    
    # Adjust shadows and highlights
    img_array = np.where(img_array < 128, img_array * (1 + shadows/100), img_array)
    img_array = np.where(img_array >= 128, 255 - (255 - img_array) * (1 - highlights/100), img_array)
    
    # Adjust brilliance (overall luminance)
    img_array *= (1 + brilliance/100)
    
    # Adjust exposure
    # img_array *= 2 ** exposure
    
    # Clip values to valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    img = Image.fromarray(img_array, mode=original_mode)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1 + contrast/100)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1 + brightness/100)
    
    # Adjust black point
    img_array = np.array(img).astype(float)
    img_array = np.clip(img_array - black_point, 0, 255)
    img_array = ((img_array / (255 - black_point)) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode=original_mode)
    
    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1 + sharpness/100)
    
    # Apply noise reduction
    if noise_reduction > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=noise_reduction/10))
    
    # Convert back to numpy array for return
    img_array = np.array(img)
    
    # Convert RGB to BGR if it's a color image
    if original_mode == 'RGB':
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array