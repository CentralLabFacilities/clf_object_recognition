import os
import sys
import cv2
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    ),
)

from clf_object_recognition_yolox import yolo26_recognizer

MODEL = "/home/tbeckmann/catkin_ws/src/clf_object_recognition/yolo26n-seg.pt"
IMAGE = "/home/tbeckmann/catkin_ws/src/clf_object_recognition/bus.jpg"

rec = yolo26_recognizer.Recognizer(MODEL)

img = cv2.imread(IMAGE)
orig = img.copy()

h, w = img.shape[:2]

cls, scores, boxes, masks = rec.inference(img, confthreshold=0.25)

print("Classes:", cls)
print("Scores:", scores)
print("Boxes:", boxes)

if masks is not None:
    print("Masks shape:", masks.shape)
    print("Masks type: ", masks.dtype)
# -----------------------------
# Draw segmentation masks
# -----------------------------
if masks is not None:
    for i, mask in enumerate(masks):
        # Convert to binary mask
        mask_u8 = (mask > 0.5).astype(np.uint8)

        print(
            f"Mask {i}: shape={mask_u8.shape}, "
            f"pixels={mask_u8.sum()}"
        )

        # Resize mask to original image size
        mask_u8 = cv2.resize(
            mask_u8,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        # Random color per instance
        color = np.random.randint(0, 255, 3, dtype=np.uint8)

        colored_mask = np.zeros_like(orig, dtype=np.uint8)
        colored_mask[mask_u8 > 0] = color

        # Blend with original image
        img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)

# -----------------------------
# Show result
# -----------------------------
cv2.imshow("YOLO Segmentation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: save output
cv2.imwrite("yolo26_seg_result_bus.jpg", img)
print("Saved: yolo26_seg_result.jpg")