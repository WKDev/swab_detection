import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def nothing(x):
    pass

def process_image(min_area, min_distance, threshold):
    # Otsu's thresholding
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise by filtering using contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(thresh, [c], 0, (0,0,0), -1)

    # Compute Euclidean distance and find peaks
    distance_map = ndimage.distance_transform_edt(thresh)
    local_max = peak_local_max(distance_map,min_distance=min_distance, labels=thresh)

    # Perform connected component analysis
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    
    # Ensure markers and thresh have the same shape
    markers = markers.astype(np.int32)
    
    # Apply Watershed
    labels = watershed(-distance_map, markers, mask=thresh)

    # Create the result image
    result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # Iterate through unique labels
    for label in np.unique(labels):
        if label == 0:
            continue

        # Create a mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(result, [c], -1, (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)), 2)

    return result

# Load image and convert to grayscale
image = cv2.imread('images/mb_001_A_500.jpg')
if image is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create window and trackbars
cv2.namedWindow('Segmented Image')
cv2.createTrackbar('Min Area', 'Segmented Image', 1000, 5000, nothing)
cv2.createTrackbar('Min Distance', 'Segmented Image', 20, 100, nothing)
cv2.createTrackbar('Threshold', 'Segmented Image', 0, 255, nothing)

while True:
    min_area = cv2.getTrackbarPos('Min Area', 'Segmented Image')
    min_distance = cv2.getTrackbarPos('Min Distance', 'Segmented Image')
    threshold = cv2.getTrackbarPos('Threshold', 'Segmented Image')
    
    result = process_image(min_area, min_distance, threshold)
    
    cv2.imshow('Segmented Image', result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()