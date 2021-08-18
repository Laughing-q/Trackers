from detector import Yolov5
import cv2
from trackers.deep_sort.parser import get_config
from trackers.deep_sort.deep_sort import DeepSort
from yolov5.utils.plots import plot_one_box
from trackers import ObjectTracker
from counter import OneLine


if __name__ == '__main__':
    detector = Yolov5(weight_path='./yolov5/weights/yolov5s.pt', device='0', img_hw=(640, 640))

    Track = True
    detector.show = False if Track else True
    detector.pause = False

    num_lines = 1
    lineStat = []

    # tracker = ObjectTracker('deepsort')
    tracker = ObjectTracker('sort')
    # for video
    pause = True
    cap = cv2.VideoCapture('/d/projects/tensorrtx-2.0n/yolov5/test.mp4')
    frame_num = 0
    cv2.namedWindow('p', cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num == 1:
            frame_ = frame.copy()
            for j in range(num_lines):
                roi1 = []
                for i in range(2):
                    roi1.append(
                        cv2.selectROI(windowName="roi",
                                      img=frame_,
                                      showCrosshair=False,
                                      fromCenter=False))
                xr2, yr2, wr2, hr2 = roi1[0][0], roi1[0][1], roi1[1][
                    0], roi1[1][1]
                line = [(xr2, yr2), (wr2, hr2)]
                # line = [(xr2 - xr1, yr2 - yr1), (wr2 - xr1, hr2 - yr1)]
                lineStat.append(line)
                # pt.entryExitStat[j].line = line
                cv2.line(frame_, line[0], line[1], (255, 0, 100), 2)
            cv2.destroyWindow('roi')
            Counter = OneLine(lineStat=lineStat, pixel=10)

        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _ = detector.dynamic_detect(img, [img_raw], classes=[0])
        if not Track:
            continue
        box = preds[0][:, :4].cpu()
        cls = preds[0][:, 5].cpu()
        # tracks, temp_ids = tracker.update(bbox_xyxy=box, 
        #                                   cls=cls,
        #                                   ori_img=img_raw,)
        tracks = tracker.update(dets=box, cls=cls)

        for i, track in enumerate(tracks):
            # 行人检测框
            box = [int(b) for b in track[:4]]
            id = track[4]
            pt = ((box[0] + box[2]) // 2, box[3])  # 取脚点
            Counter.add_obj(id, pt)

            plot_one_box(box, img_raw, label=None, color=(
                20, 20, 255), line_thickness=2)
            # cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), (20, 20, 255))
            text = "{}".format(id)
            cv2.putText(img_raw, text, (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            Counter.plot_tail(id, img_raw)
            Counter.parse_obj(id, pt, img_raw)

        img_raw = Counter.plot_line(img_raw)
        img_raw = Counter.show_count(img_raw)

        cv2.imshow('p', img_raw)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(' ') else False
        if key == ord('q') or key == ord('e') or key == 27:
            break
