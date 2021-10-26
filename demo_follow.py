from detector import Yolov5
import cv2
from trackers.deep_sort.parser import get_config
from trackers.deep_sort.deep_sort import DeepSort
from yolov5.utils.plots import plot_one_box
from trackers import ObjectTracker
from counter import TrailParser, Dist
import os
import os.path as osp
import numpy as np


if __name__ == "__main__":
    detector = Yolov5(
        weight_path="./yolov5/weights/yolov5m.pt", device="0", img_hw=(640, 640)
    )

    Track = True
    detector.show = False if Track else True
    detector.pause = False

    save = True  # 是否保存视频
    save_dir = "./output"  # 保存视频路径
    os.makedirs(save_dir, exist_ok=True)

    type = "sort"
    # type = 'deepsort'
    # type = 'bytetrack'
    tracker = ObjectTracker(type=type)
    conf_thresh = 0.2 if type == "bytetrack" else 0.4
    trail = TrailParser()

    # for video
    pause = True
    test_video = "./test.mp4"
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

        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _ = detector.dynamic_detect(
            img, [img_raw], classes=[0], conf_threshold=conf_thresh
        )

        if not Track:
            continue
        box = preds[0][:, :4].cpu()
        conf = preds[0][:, 4].cpu()
        cls = preds[0][:, 5].cpu()

        tracks = tracker.update(bboxes=box, scores=conf, cls=cls, ori_img=img_raw)

        # 计算每个跟踪对象的距离,默认使用中心点
        dist = Dist(bboxes=tracks[:, :4], ids=tracks[:, 4])
        dist.plot(frame, dist_thres=150)

        for i, track in enumerate(tracks):
            box = [int(b) for b in track[:4]]
            id = track[4]
            plot_one_box(
                box, img_raw, label=f"{id}", color=(255, 20, 20), line_thickness=2
            )

            # pt = ((box[0] + box[2]) // 2, box[3])  # 取脚点
            pt = ((box[0] + box[2]) // 2, (box[3] + box[1]) // 2)  # 取中心点
            trail.add_point(id, pt, frame_num)
            trail.plot(id, frame)

            # TODO
            # 计算所有轨迹与该id轨迹的拟合程度
            fits, tids = trail.linesfit(id=id)

            # 轨迹相似的id
            tids = tids[fits <= 0.6]
            # 距离相近的id
            # 取右上角的距离即可
            dids = dist.ids[dist.dist[i] <= 150][i + 1 :]
            # TODO
            for ii, did in enumerate(dids):
                if did not in tids:
                    continue
                pt = ((box[0] + box[2]) // 2, box[3] - 15 * ii)  # 取脚点
                cv2.putText(
                    img_raw,
                    f"{did}",
                    pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        trail.clear_old_points(current_frame=frame_num)

        if vid_writer is not None:
            vid_writer.write(frame)

        cv2.imshow("p", img_raw)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(" ") else False
        if key == ord("q") or key == ord("e") or key == 27:
            break
