from detector import Yolov5
from yolov5.utils.plots import plot_one_box
from trackers.counter import OneLine
import cv2
from trackers.tracker import ObjectTracker
import os


if __name__ == "__main__":
    detector = Yolov5(
        weight_path="/home/laughing/Person_Head_manager.pt", device="0", img_hw=(640, 640)
    )

    Track = True
    detector.show = False if Track else True
    detector.pause = False

    num_lines = 1
    lineStat = []

    save = True  # 是否保存视频
    save_dir = "./output"  # 保存视频路径
    os.makedirs(save_dir, exist_ok=True)

    # type = "sort"
    type = 'deepsort'
    # type = 'bytetrack'
    tracker = ObjectTracker(type=type)
    conf_thresh = 0.2 if type == "bytetrack" else 0.4

    # for video
    pause = True
    test_video = "/d/九江/1216/balihu_keliu.mp4"
    cap = cv2.VideoCapture(test_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    fourcc = "mp4v"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_writer = (
        cv2.VideoWriter(
            os.path.join(save_dir, test_video.split(os.sep)[-1]),
            cv2.VideoWriter_fourcc(*fourcc),
            fps if fps <= 30 else 25,
            (w, h),
        )
        if save
        else None
    )

    frame_num = 0
    if Track:
        cv2.namedWindow("p", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num == 1:
            frame_ = frame.copy()
            for j in range(num_lines):
                rois = cv2.selectROIs("roi", frame, False, False)  # xywh
                # print(rois)
                assert len(rois) == 2

                x1, y1 = rois[0][:2]
                x2, y2 = rois[1][:2]
                line = [(x1, y1), (x2, y2)]
                lineStat.append(line)
                cv2.line(frame_, line[0], line[1], (255, 0, 100), 2)
            cv2.destroyWindow("roi")
            Counter = OneLine(lineStat=lineStat, pixel=10)

        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _ = detector.dynamic_detect(
            img, [img_raw], classes=None, conf_threshold=conf_thresh
        )
        if not Track:
            continue
        box = preds[0][:, :4].cpu()
        conf = preds[0][:, 4].cpu()
        cls = preds[0][:, 5].cpu()
        tracks = tracker.update(
            bboxes=box,
            scores=conf,
            cls=cls,
            ori_img=img_raw,
        )

        for i, track in enumerate(tracks):
            # 行人检测框
            box = [int(b) for b in track[:4]]
            id = track[4]
            pt = ((box[0] + box[2]) // 2, box[3])  # 取脚点
            # Counter.add_obj(id, pt)
            Counter.count(id, pt, frame_num=frame_num, frame=img_raw)
            Counter.plot_tail(id, frame=img_raw)

            plot_one_box(
                box, img_raw, label=None, color=(20, 20, 255), line_thickness=2
            )
            text = "{}".format(id)
            cv2.putText(
                img_raw,
                text,
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        Counter.plot_result(img_raw)
        # img_raw = Counter.plot_result_PIL(img_raw)

        Counter.clear_old_points(current_frame=frame_num)

        if vid_writer is not None:
            vid_writer.write(frame)

        cv2.imshow("p", img_raw)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(" ") else False
        if key == ord("q") or key == ord("e") or key == 27:
            break
