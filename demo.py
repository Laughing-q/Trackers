from detector import Yolov5
import cv2
from trackers.deep_sort.parser import get_config
from trackers.deep_sort.deep_sort import DeepSort
from yolov5.utils.plots import plot_one_box
from tracker import ObjectTracker


if __name__ == '__main__':
    detector = Yolov5(weight_path='./yolov5/weights/yolov5s.pt', device='0', img_hw=(640, 640))

    detector.show = False
    detector.pause = False

    # tracker = ObjectTracker('deepsort')
    tracker = ObjectTracker('sort')
    # for video
    pause = True
    cap = cv2.VideoCapture('/e/1.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _ = detector.dynamic_detect(img, [img_raw], classes=[0])
        box = preds[0][:, :4].cpu()
        cls = preds[0][:, 5].cpu()
        # tracks, temp_ids = tracker.update(bbox_xyxy=box, 
        #                                   cls=cls,
        #                                   ori_img=img_raw,)
        tracks = tracker.update(dets=box, cls=cls)

        for i, track in enumerate(tracks):
            # 行人检测框
            box = track[:4]
            id = track[4]
            plot_one_box(box, img_raw, label=None, color=(
                20, 20, 255), line_thickness=2)
            # cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), (20, 20, 255))
            text = "{}".format(id)
            cv2.putText(img_raw, text, (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('p', img_raw)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(' ') else False
        if key == ord('q') or key == ord('e') or key == 27:
            break
