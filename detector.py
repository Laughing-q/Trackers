import pycuda.driver as cuda
import tensorrt as trt
import time
import sys
sys.path.append('./yolov5')
from utils.torch_utils import select_device, load_classifier, time_synchronized, time_synchronized
from utils.plots import plot_one_box
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy,\
    merge_dict, polygon_ROIarea, nms_numpy
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
from models.yolo import Model
import random
import os
import cv2
import numpy as np
import torch
import yaml

random.seed(1)


class Yolov5:
    def __init__(self, weight_path, device, img_hw=(384, 640)):
        self.weights = weight_path
        self.device = select_device(device)
        self.half = True
        # path aware
        # self.model = attempt_load(self.weights, map_location=self.device)
        # pt -> pth, path agnostic
        self.model = torch.load(self.weights, map_location=self.device)['model']
        with open(weight_path.replace('.pt', '.yaml'), 'w') as f:
            yaml.safe_dump(self.model.yaml, f, sort_keys=False)
        torch.save(self.model.float().state_dict(), weight_path.replace('.pt', '.pth'))
        self.model.float().fuse().eval()

        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]
        self.show = False
        self.img_hw = img_hw
        self.pause = False

    def preprocess(self, image, auto=True):  # (h, w)
        if type(image) == str and os.path.isfile(image):
            img0 = cv2.imread(image)
        else:
            img0 = image
        # img, _, _ = letterbox(img0, new_shape=new_shape)
        img, _, _ = letterbox(img0, new_shape=self.img_hw, auto=auto)
        # cv2.imshow('x', img)
        # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0

    def dynamic_detect(self, image, img0s, areas=None, classes=None, conf_threshold=0.6, iou_threshold=0.4):
        output = {}
        if classes is not None:
            for c in classes:
                output[self.names[int(c)]] = 0
        else:
            for n in self.names:
                output[n] = 0
        img = torch.from_numpy(image).to(self.device)
        # print('xxxxxxxxxxxxxx:', time.time() - ttx)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print(img.shape)
        torch.cuda.synchronize()
        pred = self.model(img)[0] 
        pred = non_max_suppression(
            pred, conf_threshold, iou_threshold, classes=classes, agnostic=False)

        torch.cuda.synchronize()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0s[i].shape).round()
                # if areas is not None and len(areas[i]):
                #     _, warn = polygon_ROIarea(
                #         det[:, :4], areas[i], img0s[i])
                #     det = det[warn]
                #     pred[i] = det
                for di, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    output[self.names[int(cls)]] += 1
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    # label = '%s' % (self.names[int(cls)])
                    if self.show:
                        plot_one_box(xyxy, img0s[i], label=label,
                                     color=self.colors[int(cls)], 
                                     line_thickness=2)

        if self.show:
            for i in range(len(img0s)):
                cv2.namedWindow(f'p{i}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'p{i}', img0s[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(' ') else False
            if key == ord('q') or key == ord('e') or key == 27:
                exit()
        return pred, output


class YOLOV5TRT:
    def __init__(self, engine_file_path, LoadLibrary, cfx, stream, names, num_classes, img_hw=(384, 640), device=0):
        # Create a Context on this device,
        self.cfx = cfx
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        LoadLibrary.Regist(num_classes, img_hw[1], img_hw[0])

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        # print(engine)
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        # self.bindings = bindings

        self.img_hw = img_hw
        self.batch_size = engine.max_batch_size
        self.alloc_output(batch_size=self.batch_size)
        # self.alloc_input(batch_size=engine.max_batch_size)

        self.names = names
        self.show = False
        self.pause = False
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.names) if not isinstance(self.names, int) else self.names)]
        self.draw = None

    def alloc_input(self, batch_size=0, hw=(384, 640)):
        batch_size = self.engine.max_batch_size if batch_size == 0 else batch_size
        size = hw[0] * hw[1] * 3 * batch_size
        dtype = np.float32
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        self.host_inputs.append(host_mem)
        self.cuda_inputs.append(cuda_mem)

    def alloc_output(self, batch_size=0, size=6001):
        batch_size = self.engine.max_batch_size if batch_size == 0 else batch_size
        size = size * batch_size
        dtype = np.float32
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        self.host_outputs.append(host_mem)
        self.cuda_outputs.append(cuda_mem)

    def copy_data(self, input_image):
        # input_image = input_image / 255.
        np.copyto(self.host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        return self.cuda_inputs

    def preprocess(self, image, new_shape=(384, 640)):  # (h, w)
        if type(image) == str and os.path.isfile(image):
            img0 = cv2.imread(image)
        else:
            img0 = image
        # img, _, _ = letterbox(img0, new_shape=new_shape)
        img, _, _ = letterbox(img0, new_shape=self.img_hw)

        # cv2.imshow('x', img)
        # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0

    def multi_detect(self, cuda_input, image_raw, areas, classes=None, conf_threshold=0.6, iou_threshold=0.4):
        """
        image for inference
        draw_image for visualization
        return:
        pred:[N, 6], box, conf, cls
        output_count:dict{}, {"class1":number1, "class2":nmber2.....}
        """
        output_count = {}
        if classes is not None:
            for c in classes:
                output_count[self.names[int(c)]] = 0
        else:
            for n in self.names:
                output_count[n] = 0
        """
        support batch-size = 1 only for now
        """
        t1 = time.time()
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        # host_inputs = self.host_inputs
        cuda_inputs = cuda_input
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        # Run inference.
        context.execute_async(batch_size=self.batch_size,
                              bindings=[cuda_inputs[0], cuda_outputs[0]], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        preds = self.post_process(
            output, self.batch_size, classes, conf_threshold, iou_threshold)
        t2 = time.time()
        print('inference+nms time:', t2 - t1)

        for i, pred in enumerate(preds):
            if pred is None:
                continue
            pred[:, :4] = scale_coords(
                self.img_hw, pred[:, :4], image_raw[i].shape).round()

            if areas is not None and len(areas[i]):
                _, warn = polygon_ROIarea(
                    pred[:, :4], areas[i], image_raw[i])
                pred = pred[warn]

            for *xyxy, conf, cls in pred:
                output_count[self.names[int(cls)] if not isinstance(
                    self.names, int) else int(cls)] += 1
                label = '%s %.2f%%' % (self.names[int(cls)], conf*100)
                plot_one_box(xyxy, image_raw[i], label=label,
                             color=self.colors[int(cls)], line_thickness=2)
        if self.show:
            for i in range(self.batch_size):
                cv2.namedWindow(f'p{i}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'p{i}', image_raw[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(' ') else False
            if key == ord('q') or key == ord('e') or key == 27:
                exit()
        return preds, output_count, t2 - t1

    def dynamic_detect(self, images, draw_images, areas, classes=None, conf_threshold=0.6, iou_threshold=0.4):
        """
        Multiple call detect(batch-size=1) for multi batch-size
        images for inference
        draw_images for visualization
        return:
        pred:list[], for each [N, 6], box, conf, cls
        outputs:dict{}, {"class1":number1, "class2":nmber2.....}
        """
        preds = []
        outputs = {}
        for i, (img, dimg) in enumerate(zip(images, draw_images)):
           pred, output_dict, _ = self.multi_detect(img, [dimg], classes=classes, areas=[areas[i]],
                                                    conf_threshold=conf_threshold, iou_threshold=iou_threshold)
           preds.append(pred[0])
           merge_dict(outputs, output_dict)  # 融合字典，字典对应键值相加，如无键值则添加新的键值对
        return preds, outputs, 0

    def destory(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def post_process(self, output, batch_size,  classes=None, conf_threshold=0.6, iou_threshold=0.4):
        '''
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        '''
        # Get the num of boxes detected
        # num = int(output[0])
        preds = np.split(output, batch_size)  # list
        # preds = [output]  # list
        # Reshape to a two dimentional ndarray
        # pred = np.reshape(output[1:], (-1, 6))[:num, :]
        preds = [np.reshape(pred[1:], (-1, 6))[:int(pred[0]), :]
                 for pred in preds]
        out_preds = []
        for pred in preds:
            if classes:
                pred = pred[(pred[:, 5:6] == classes).any(1)]
            si = pred[:, 4] > conf_threshold
            pred = pred[si]
            # boxes = boxes[si, :]
            # scores = scores[si]
            # classid = classid[si]
            pred[:, :4] = xywh2xyxy(pred[:, :4])
            # Do nms
            indices = nms_numpy(pred[:, :4], pred[:, 4],
                                pred[:, 5], threshold=iou_threshold)
            keep_pred = torch.from_numpy(pred[indices, :])
            # keep_pred = pred[indices, :]
            out_preds.append(keep_pred if len(keep_pred) else None)
        # return pred[indices, :]
        return out_preds


if __name__ == '__main__':
    detector = Yolov5(weight_path='/d/projects/IFLYTEK_competition/yolov5/runs/train/exp3/weights/best.pt', device='0', img_hw=(640, 640))

    detector.show = True
    detector.pause = True
    # for video
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     img, img_raw = detector.preprocess(frame, auto=True)
    #     preds, _ = detector.dynamic_detect(img, [img_raw])
    image = cv2.imread('/d/competition/IFLYTEK/object_detection/科大讯飞股份有限公司_X光安检图像识别2021挑战赛初赛第一阶段数据集/images/train10001.jpg')
    img, img_raw = detector.preprocess(image, auto=True)
    preds, _ = detector.dynamic_detect(img, [img_raw])
