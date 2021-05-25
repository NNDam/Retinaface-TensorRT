import os
import cv2
import torch
import onnx
import time
import argparse
import onnxruntime
import numpy as np
from utils.image import ImageData
from detectors.retinaface import *
from converters.insight2onnx import convert_insight_model

class ONNX_Retinaface(object):
    def __init__(self, model_path, input_shape = (640, 640), batch_size = 4):
        print('[INFO] Create ONNX runtime')
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_shape = input_shape
        self.rac = 'net3'
        self.masks = False
        self.batch_size = batch_size
        for size in self.input_shape:
            if size % 32 != 0:
                raise ValueError("Current support only size which is multiple of 32 for compabilities")

    def prepare(self, nms: float = 0.4, **kwargs):
        self.nms_threshold = nms
        self.landmark_std = 1.0
        _ratio = (1.,)
        fmc = 3
        if self.rac == 'net3':
            _ratio = (1.,)
        elif self.rac == 'net3l':
            _ratio = (1.,)
            self.landmark_std = 0.2
        else:
            assert False, 'rac setting error %s' % self.rac

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        self.use_landmarks = True
        self.fpn_keys = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        
        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        # Create anchor
        print('Rebuild anchor')
        self.anchor_plane_cache = {}
        for _idx, s in enumerate(self._feat_stride_fpn):
            stride = int(s)
            width = int(self.input_shape[0]/stride)
            height = int(self.input_shape[1]/stride)
            K = width * height
            A = self._num_anchors['stride%s' % s]
            key = (height, width, stride)
            anchors_fpn = self._anchors_fpn['stride%s' % s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            self.anchor_plane_cache[key] = np.tile(anchors.reshape((K * A, 4)), (self.batch_size, 1, 1))
        self.warm_up()

    def warm_up(self):
        '''
            Warm up NMS jit
        '''
        print('Warming up NMS jit...')
        tik = time.time()
        image = cv2.imread('test_images/lumia.jpg', cv2.IMREAD_COLOR)
        image = ImageData(image, self.input_shape)
        image.resize_image(mode='pad')
        im = cv2.cvtColor(image.transformed_image, cv2.COLOR_BGR2RGB)
        im = np.transpose(im, (2, 0, 1))
        input_blob = np.tile(im, (self.batch_size, 1, 1, 1)).astype(np.float32)
        _ = self.detect(input_blob, threshold=0.1)
        tok = time.time()
        print('Warming up complete, time cost = {}'.format(tok - tik))

    def detect(self, batch_img, threshold: float = 0.6):
        t0 = time.time()
        batch_size = len(batch_img)
        assert batch_size == self.batch_size, "Model define with batch_size = {}, your input: {}".format(self.batch_size, batch_size) 
        ort_inputs = {self.ort_session.get_inputs()[0].name: batch_img}
        net_out = self.ort_session.run(None, ort_inputs)
        # print('Len net_out: {}'.format(net_out[0].shape))
        t1 = time.time()
        print('Infer: ', t1-t0)
        # print(net_out[0].shape)
        t0 = time.time()
        result = self.postprocess(net_out, threshold, batch_size = batch_size)
        t1 = time.time()
        print('Post: ', t1-t0)
        return result

    def postprocess(self, net_out, threshold, batch_size):
        '''
            Post process for batch-inference
        '''
        proposals_list_batch = {i : [] for i in range(batch_size)}
        scores_list_batch = {i : [] for i in range(batch_size)}
        landmarks_list_batch = {i : [] for i in range(batch_size)}
        t0 = time.time()
        # Foreach FPN layer
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            if self.masks:
                idx = _idx * 4

            A = self._num_anchors['stride%s' % s]
            
            scores_batch = net_out[idx]
            scores_batch = scores_batch[:, A:, :, :]
            idx += 1
            bbox_deltas_batch = net_out[idx]
            height, width = bbox_deltas_batch.shape[2], bbox_deltas_batch.shape[3]

            # K = height * width
            key = (height, width, stride)
            anchors_batch = self.anchor_plane_cache[key]

            scores_batch = clip_pad(scores_batch, (height, width))
            scores_batch = scores_batch.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 1))

            bbox_deltas_batch = clip_pad(bbox_deltas_batch, (height, width))
            bbox_deltas_batch = bbox_deltas_batch.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas_batch.shape[3] // A
            bbox_deltas_batch = bbox_deltas_batch.reshape((batch_size, -1, bbox_pred_len))
            proposals_batch = bbox_pred_batch(anchors_batch, bbox_deltas_batch)
            
            
            # Get proposal
            scores_batch = scores_batch.reshape((batch_size, -1))
            order_batch = np.argwhere(scores_batch >= threshold)
            # print(_key, order_batch)
            # Get landmark
            if self.use_landmarks:
                idx += 1
                landmark_deltas_batch = net_out[idx]
                landmark_deltas_batch = clip_pad(landmark_deltas_batch, (height, width))
                landmark_pred_len = landmark_deltas_batch.shape[1] // A
                landmark_deltas_batch = landmark_deltas_batch.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 5, landmark_pred_len // 5))
                landmark_deltas_batch *= self.landmark_std
                landmarks = landmark_pred_batch(anchors_batch, landmark_deltas_batch)

            # Foreach image
            for ib in range(batch_size): 
                order = [id[1] for id in order_batch if id[0] == ib]
                proposals_list_batch[ib].append(proposals_batch[ib, order])
                scores_list_batch[ib].append(scores_batch[ib, order].reshape((-1, 1)))
                if self.use_landmarks:
                    landmarks_list_batch[ib].append(landmarks[ib, order])
                
        # Foreach image
        list_det = []
        list_landmarks = []
        for ib in range(batch_size):
            proposals_list = proposals_list_batch[ib]
            scores_list = scores_list_batch[ib]
            landmarks_list = landmarks_list_batch[ib]

            proposals = np.vstack(proposals_list)
            landmarks = None
            if proposals.shape[0] == 0:
                if self.use_landmarks:
                    landmarks = np.zeros((0, 5, 2))
                list_det.append(np.zeros((0, 5)))
                list_landmarks.append(landmarks)
                continue

            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]

            if self.use_landmarks:
                landmarks = np.vstack(landmarks_list)
                landmarks = landmarks[order].astype(np.float32, copy=False)

            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = nms(pre_det, thresh=self.nms_threshold)
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
            t1 = time.time()
            list_det.append(det)
            list_landmarks.append(landmarks)
        return list_det, list_landmarks

def read_image(im_path):
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ImageData(image, (640, 640))
    image.resize_image(mode='pad')
    return image

parser = argparse.ArgumentParser(description='Port Face Detector from Mxnet to ONNX-Runtime.')
parser.add_argument('--model_symbol', type=str,
                    help='Path to model symbol')
parser.add_argument('--model_params', type=str,
                    help='Path to model params')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size')
parser.add_argument('--im_width', type=int, default=640,
                    help='Input image width')
parser.add_argument('--im_height', type=int, default=640,
                    help='Input image height')
parser.add_argument('--tmp_folder', type=str, default="tmp",
                    help='Path to save visualize image')
parser.add_argument('--onnx_path', type=str,
                    help='Path to output ONNX model')
args = parser.parse_args()
if not os.path.exists(args.tmp_folder):
    os.makedirs(args.tmp_folder)
batch_size = args.batch_size
symbol = args.model_symbol
params = args.model_params
ONNX_FILE_PATH = args.onnx_path
im_width = args.im_width
im_height = args.im_height
print('[INFO] Convert model')
convert_insight_model(symbol, params, ONNX_FILE_PATH, input_shape=(batch_size, 3, im_width, im_height))
print('[INFO] Test model')
ort_session = onnxruntime.InferenceSession(ONNX_FILE_PATH)
onnx_face = ONNX_Retinaface(ONNX_FILE_PATH, batch_size = batch_size)
onnx_face.prepare()

print('- Preprocess input')
tt0 = time.time()
img_path = 'test_images/lumia.jpg'
image = read_image(img_path)
inp = np.array([image.transformed_image for i in range(batch_size)], dtype = np.float32)
inp = np.transpose(inp, (0, 3, 1, 2))
tt1 = time.time()
print(f"Preparing image took: {tt1 - tt0}")

print('- Perform inference')
tik = time.time()
list_det, list_landmarks = onnx_face.detect(inp)
tok = time.time() - tik

print('- Visualize')
# Visualize
for i in range(1): # Foreach image
    vis_im = cv2.cvtColor(image.transformed_image.copy(), cv2.COLOR_RGB2BGR)
    for det in list_det[i]: 
        if det[4] > 0.6:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        pt1 = tuple(map(int, det[0:2]))
        pt2 = tuple(map(int, det[2:4]))
        cv2.rectangle(vis_im, pt1, pt2, color, 1)
    cv2.imwrite(os.path.join(args.tmp_folder, 'res_{}.jpg'.format(i)), vis_im)
print('[INFO] Port to ONXX completed! Please read document if u want to port to TensorRT')