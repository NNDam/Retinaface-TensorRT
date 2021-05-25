import os
import cv2
import time
import logging
import argparse
from utils.getter import get_model
from utils.image import ImageData



parser = argparse.ArgumentParser(description='Port Face Detector from Mxnet to TensorRT-Runtime.')
parser.add_argument('--model_name', type=str, default='retinaface_r50_v1',
                    help='Model architecture: retinaface_r50_v1, retinaface_mnet025_v2, mnet_cov2')
parser.add_argument('--it', type=int, default=10,
                    help='Number of iterations for calculate FPS')
parser.add_argument('--dir', type=str, default="weights",
                    help='Working Directory')
parser.add_argument('--im_width', type=int, default=640,
                    help='Input image width')
parser.add_argument('--im_height', type=int, default=640,
                    help='Input image height')
parser.add_argument('--fp16', action='store_true',
                    help='Use FP16 for faster inference')
args = parser.parse_args()



'''
ATTENTION!!! This script is for testing purposes only. Work in progress.

This demo script is intended for testing face detection modules.
Following steps will be executed upon launch:

1. Download official MXNet Retinaface model
2. Convert model to ONNX, applying fixes to MXNet model and resulting ONNX
3. Reshape ONNX inputs/outputs to specified image W, H
4. Build TensorRT engine from ONNX
5. Run inference with TensorRT

If Centerface model is selected, it will be downloaded and than steps 3-4 applied.

'''

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

iters = args.it
model_name = args.model_name
backend = 'trt'
im_size = [args.im_width, args.im_height]
root_dir = args.dir
detector = get_model(model_name, backend, im_size=im_size, root_dir=root_dir, force_fp16=args.fp16, download_model = True)
detector.prepare(nms=0.4)

# Since dynamic shapes are not yet implemented, images during inference
# must be resized to exact shape which was used for TRT building.
# To ensure this we read expected input shape from engine itself.
input_shape = detector.input_shape[2:][::-1]
logging.info(f"Images will be resized to WxH : {input_shape}")

tt0 = time.time()
image = cv2.imread('test_images/lumia.jpg', cv2.IMREAD_COLOR)
image = ImageData(image, input_shape)
image.resize_image(mode='pad')
tt1 = time.time()

print(f"Preparing image took: {tt1 - tt0}")

# Since we are using Numba JIT compilation, first run will cause
# compilation on NMS module
logging.info('Warming up NMS jit...')
detections = detector.detect(image.transformed_image, threshold=0.1)
logging.info('Warming up complete!')



det_count = 0

t0 = time.time()
for i in range(iters):
    tf0 = time.time()
    detections = detector.detect(image.transformed_image, threshold=0.6)
    tf1 = time.time()
    det_count = len(detections[0])
    logging.debug(f"Full detection  took: {tf1 - tf0}")

t1 = time.time()
print(f'Took {t1 - t0} s. ({iters / (t1 - t0)} im/sec)')
print(f"Total faces detected: {det_count}")

mask_thresh = 0.2

for det in detections[0]:

    if det[4] > 0.6:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Testing mnet_cov2 model.
    if detector.masks == True:
        mask_score = det[5]
        logging.info(f'Mask score: {mask_score}. Face score: {det[4]}')
        if mask_score >= mask_thresh:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

    pt1 = tuple(map(int, det[0:2]))
    pt2 = tuple(map(int, det[2:4]))
    cv2.rectangle(image.transformed_image, pt1, pt2, color, 1)
print('Save test image to {}.jpg'.format(model_name))
cv2.imwrite(f'{model_name}.jpg', image.transformed_image)
