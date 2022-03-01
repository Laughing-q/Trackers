from collections import defaultdict
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from .utils import (
    point2LineDistance,
    intersect,
    vertical_line,
    vLineAngle,
    getLineCenter,
)


class OneLine:
    """
    Args:
        lineStat (List[List[(x1, y1), (x2, y2)] * num_lines]): line.
    """

    def __init__(self, lineStat, pixel=10, count_once=False, logger=None):
        self.trailObject = TrailParser()
        self.entryCnt = 0
        self.exitCnt = 0
        self.pixel = pixel
        self.lineStat = lineStat
        self.num_lines = len(lineStat)
        self.logger = logger
        self.countedObject = (
            [defaultdict(list) for _ in range(self.num_lines)] if count_once else None
        )

    def count(self, id, pt, frame_num, frame=None):
        # print(len(self.trailObject.frameObject))
        self.trailObject.add_point(id, pt, frame_num)
        hisPoints = self.trailObject.frameObject[id]
        # if not (
        #     len(hisPoints) > 3 and getPointLen(hisPoints[0], hisPoints[-1]) > self.pixel
        # ):
        #     return
        # p1, p2 = hisPoints[-1], hisPoints[-4]
        if not (len(hisPoints) >= 2):
            return
        p1, p2 = hisPoints[-1], hisPoints[-2]
        # for visual test
        cv2.line(frame, p1, p2, (255, 0, 0), 5)
        for i, line in enumerate(self.lineStat):
            # sd = point2LineDistance(p1, line)
            # ed = point2LineDistance(p2, line)
            # if sd * ed < 0 and intersect(
            #     p1, p2, line[0], line[1]
            # ):
            if intersect(p1, p2, line[0], line[1]):
                ang = vLineAngle([p1, p2], line)
                if ang < 180 and self._count_check(i, id, "in"):
                    self.entryCnt += 1
                    self._count_sign(i, id, "in", frame_num)
                elif ang >= 180 and self._count_check(i, id, "out"):
                    self.exitCnt += 1
                    self._count_sign(i, id, "out", frame_num)

    def _count_check(self, i, id, sign):
        return not (
            self.countedObject is not None and sign in self.countedObject[i][id]
        )

    def _count_sign(self, i, id, sign, frame_num):
        if self.countedObject is not None:
            self.countedObject[i][id].append(sign)
        if self.logger is not None:
            self.logger.info(f"{frame_num}-{sign}")

    def clear_old_points(self, current_frame, interval):
        self.trailObject.clear_old_points(current_frame, interval)

    def plot_tail(self, id, frame):
        # plot trail
        self.trailObject.plot(id, frame)

    def plot_result(self, frame):
        # plot count
        cv2.putText(
            frame,
            f"in:{self.entryCnt}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"out:{self.exitCnt}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )
        # plot line
        for i in range(len(self.lineStat)):
            cv2.line(
                frame, self.lineStat[i][0], self.lineStat[i][1], (100, 230, 255), 2
            )
            p1_1, p1_2, out = vertical_line(self.lineStat[i][0], self.lineStat[i][1])
            if out is None:
                continue
            sign = "out" if out else "in"
            cv2.putText(
                frame,
                sign,
                tuple(p1_2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )
            cv2.arrowedLine(
                frame, tuple(p1_1), tuple(p1_2), color=(0, 200, 255), thickness=2
            )

    def plot_result_PIL(self, frame):
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/arphic/ukai.ttc", size=30)
        draw.text((50, 100), f"进入:{self.entryCnt}", fill=(0, 255, 0), font=font)
        draw.text((50, 150), f"出去:{self.exitCnt}", fill=(0, 255, 0), font=font)
        for i in range(len(self.lineStat)):
            p1_1, p1_2, out = vertical_line(self.lineStat[i][0], self.lineStat[i][1])
            if out is None:
                continue
            sign = "出" if out else "进"
            draw.text(tuple(p1_2), sign, fill=(0, 0, 255), font=font)
        frame = np.array(im)
        for i in range(len(self.lineStat)):
            cv2.line(
                frame, self.lineStat[i][0], self.lineStat[i][1], (100, 230, 255), 2
            )
            cv2.arrowedLine(
                frame, tuple(p1_1), tuple(p1_2), color=(0, 200, 255), thickness=2
            )
        return frame

    def __repr__(self) -> str:
        return f"inCnt:{self.entryCnt}, outCnt:{self.exitCnt}"


class TwoLine(OneLine):
    """
    Args:
        lineStat (List[List[(x1, y1), (x2, y2), (x3, y3), (x4, y4)] * num_lines]): line.
        pixel (int): 判定点之间的距离要求.
        count_once (bool): 是否同一个id只计数一次(每条线分开算的).
    """

    def __init__(self, lineStat, pixel=10, count_once=False, logger=None):
        super(TwoLine, self).__init__(lineStat, pixel, count_once, logger)
        self.lineObject = [defaultdict(list) for _ in range(self.num_lines)]

    def count(self, id, pt, frame_num, frame=None):
        self.trailObject.add_point(id, pt, frame_num)
        hisPoints = self.trailObject.frameObject[id]
        # if not (
        #     len(hisPoints) > 3 and getPointLen(hisPoints[0], hisPoints[-1]) > self.pixel
        # ):
        #     return
        # p1, p2 = hisPoints[-1], hisPoints[-4]
        if not (len(hisPoints) >= 2):
            return
        p1, p2 = hisPoints[-1], hisPoints[-2]
        # for visual test
        cv2.line(frame, p1, p2, (255, 0, 0), 5)
        for i, line in enumerate(self.lineStat):
            # TODO
            if intersect(p1, p2, line[0], line[1]):
                self.lineObject[i][id].append(1)
            if intersect(p1, p2, line[2], line[3]):
                self.lineObject[i][id].append(2)
            if len(self.lineObject[i][id]) >= 2:
                if (
                    self.lineObject[i][id][-2] < self.lineObject[i][id][-1]
                ) and self._count_check(i, id, "out"):
                    self.exitCnt += 1
                    self._count_sign(i, id, "out", frame_num)
                if (
                    self.lineObject[i][id][-2] > self.lineObject[i][id][-1]
                ) and self._count_check(i, id, "in"):
                    self.entryCnt += 1
                    self._count_sign(i, id, "in", frame_num)
                # 计数之后清空
                if self.lineObject[i][id][-2] != self.lineObject[i][id][-1]:
                    del self.lineObject[i][id]

    def clear_old_points(self, current_frame, interval):
        remove_ids = self.trailObject.clear_old_points(current_frame, interval)
        for id in remove_ids:
            for i in range(self.num_lines):
                if id in self.lineObject[i].keys():
                    del self.lineObject[i][id]
                if (
                    self.countedObject is not None
                    and id in self.countedObject[i].keys()
                ):
                    del self.countedObject[i][id]

    def plot_result(self, frame):
        # plot count
        cv2.putText(
            frame,
            f"in:{self.entryCnt}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"out:{self.exitCnt}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )
        # plot line
        for line in self.lineStat:
            cv2.line(frame, line[0], line[1], (100, 230, 255), 2)
            cv2.line(frame, line[2], line[3], (100, 230, 255), 2)
            c1 = getLineCenter(line[0], line[1])
            c2 = getLineCenter(line[2], line[3])
            cc = getLineCenter(c1, c2)
            cv2.putText(
                frame,
                "out",
                cc,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )
            cv2.arrowedLine(frame, c1, c2, color=(0, 200, 255), thickness=2)

    def plot_result_PIL(self, frame):
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/arphic/ukai.ttc", size=30)
        draw.text((50, 100), f"进入:{self.entryCnt}", fill=(0, 255, 0), font=font)
        draw.text((50, 150), f"出去:{self.exitCnt}", fill=(0, 255, 0), font=font)

        c1s, c2s = [], []
        for line in self.lineStat:
            c1 = getLineCenter(line[0], line[1])
            c2 = getLineCenter(line[2], line[3])
            c1s.append(c1)
            c2s.append(c2)
            cc = getLineCenter(c1, c2)
            draw.text(cc, f"出", fill=(0, 0, 255), font=font)
        frame = np.array(im)
        for i, line in enumerate(self.lineStat):
            cv2.line(frame, line[0], line[1], (100, 230, 255), 2)
            cv2.line(frame, line[2], line[3], (100, 230, 255), 2)
            cv2.arrowedLine(frame, c1s[i], c2s[i], color=(0, 200, 255), thickness=2)
        return frame


class TrailParser:
    """trace the trail and analyse"""

    def __init__(self):
        self.frameObject = defaultdict(list)
        self.frameTime = {}

    def add_point(self, id, pt, current_frame=None):
        if len(self.frameObject[id]) > 40:
            del self.frameObject[id][0]
        self.frameObject[id].append(pt)

        if current_frame is None:
            return
        self.frameTime[id] = current_frame

    def clear_old_points(self, current_frame, interval):
        """clear abandoned ids
        Args:
            current_frame: The current frame of the video.
            interval: The interval of removed id's frames.
        """
        frame_nums = np.array(list(self.frameTime.values()))
        ids = np.array(list(self.frameTime.keys()))

        dist_frame = current_frame - frame_nums
        remove_ids = ids[dist_frame > interval]
        for id in remove_ids:
            del self.frameObject[id]
            del self.frameTime[id]
        return remove_ids

    def plot(self, id, frame, nums=20, color=(0, 0, 255)):
        """plot"""
        num_points = len(self.frameObject[id])
        plotPoints = self.frameObject[id][-min(nums, num_points) :]
        for i, pt in enumerate(plotPoints):
            cv2.line(frame, pt, plotPoints[max(i - 1, 0)], color, 2)

    def linesfit(self, id=None):
        """fit the trails"""
        # trails = np.array(self.frameObject.values())
        # print(trails)
        if len(self.frameObject) == 0 or len(self.frameObject[id]) == 0:
            return

        num = len(self.frameObject)
        ids = np.array(list(self.frameObject.keys()))
        current_obj = self.frameObject[id]
        fits = np.ones(num) * np.inf

        for i, (id_i, pts_i) in enumerate(self.frameObject.items()):
            if id_i == id or len(current_obj) != len(pts_i):
                continue
            fits[i] = self._linefit(current_obj, pts_i)
        return fits, ids

    def _linefit(self, line1, line2):
        """
        Args:
            line1: (n, 2).
            line2: (n, 2).
        """
        line1 = np.array(line1)
        line2 = np.array(line2)

        max1 = np.max(line1, axis=0)
        min1 = np.min(line1, axis=0)

        max2 = np.max(line2, axis=0)
        min2 = np.min(line2, axis=0)

        nline1 = (line1 - min1) / max1
        nline2 = (line2 - min2) / max2

        dist = np.sqrt((np.square(nline1 - nline2)).sum())
        return dist


class Dist:
    """calculate the distance between the tracks, chose the center point by default"""

    def __init__(self, bboxes, ids):
        # The `bboxes` and `ids` should correspond each other
        assert len(bboxes) == len(ids)
        bboxes = bboxes if bboxes.ndim == 2 else bboxes[None, :]
        self.bboxes = bboxes
        self.ids = ids

        self._x_center = (self.bboxes[:, 0] + self.bboxes[:, 2]) // 2
        self._y_center = (self.bboxes[:, 1] + self.bboxes[:, 3]) // 2

        # (n, 2)
        self.cxy = np.stack((self._x_center, self._y_center), axis=1)

        self.dist = self._calculate()
        self.bbox_num = len(self.dist)

    def _calculate(self):
        """
        The distance between bboxes.
        Returns:
            The distance of each bboxes, (n, n).
        """
        # (n, n)
        return np.sqrt((np.square(self.cxy[:, None, :] - self.cxy[None, :, :])).sum(-1))

    def plot(self, frame, dist_thres=100):
        for i in range(self.bbox_num):
            for j in range(self.bbox_num):
                # 忽略矩阵左下对角
                if j <= i or self.dist[i][j] > dist_thres:
                    # if j <= i:
                    continue

                p1 = [int(x) for x in self.cxy[i]]
                p2 = [int(x) for x in self.cxy[j]]
                cv2.line(frame, p1, p2, (0, 0, 255), 2)


if __name__ == "__main__":
    a = [1, 2, 3, 4]
    b = a
    b.append(0)
    print(a)

    a = np.ones((2, 2)) * np.inf
    print(a == np.inf)
    print(a.min() == np.inf)

    a = np.array([1, 2, 3])
    b = np.array([1, 2, 1, 2, 3])
    print(np.unique(b, return_inverse=True))
