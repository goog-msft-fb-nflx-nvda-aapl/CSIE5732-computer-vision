# Enter your code here
import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

# Load the image
image_path = "scanned-form.jpg"
image = cv2.imread(image_path)

# Resize the image to a reasonable width for processing
ratio = image.shape[0] / 500.0
original_image = image.copy()
image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to 3-channel
gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Use GrabCut for binarization and background removal
mask = np.zeros(gray.shape, dtype=np.uint8)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)
rect = (50, 50, gray.shape[1] - 50, gray.shape[0] - 50)
cv2.grabCut(gray_3channel, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to create a binary mask for the foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Find contours in the binary mask
contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the document)
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour to a rectangle
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Perform perspective transformation
warped = four_point_transform(original_image, approx.reshape(4, 2) * ratio)

# Resize the rectified image to a width of 500 pixels
warped_resized = cv2.resize(warped, (500, int(500 * warped.shape[0] / warped.shape[1])))

# Display the original and rectified images
cv2.imshow("Original Image", original_image)
cv2.imshow("Rectified Image", warped_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
