from easydict import EasyDict as edict
import cv2
import math

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def vLineAngle(v1, v2):
    '''
    计算两条直线的夹角(0-360)
    '''
    dx1 = v1[1][0] - v1[0][0]
    dy1 = v1[1][1] - v1[0][1]
    dx2 = v2[1][0] - v2[0][0]
    dy2 = v2[1][1] - v2[0][1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = float(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = float(angle2 * 180 / math.pi)
    # print(angle2)
    included_angle = angle2 - angle1
    if included_angle < 0:
        included_angle += 360
    return included_angle

def getPointLen(p1, p2):
    '''
    计算2点间距离
    '''
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return math.sqrt((x**2) + (y**2))

def point2LineDistance(point, line, abs=False):
    '''
    计算点到线的距离
    abs: 取绝对值
    '''
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0][0]
    line_s_y = line[0][1]
    line_e_x = line[1][0]
    line_e_y = line[1][1]
    # 若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x) if abs else point_x - line_s_x
    # 若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y) if abs else point_y - line_s_y
    # 斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    # 截距
    b = line_s_y - k * line_s_x
    # 带入公式得到距离dis
    t = math.fabs(k * point_x - point_y +
                  b) if abs else k * point_x - point_y + b
    dis = t / math.pow(k * k + 1, 0.5)
    return dis


class OneLine:
    def __init__(self):
        self.frameObject = {}
        self.entryCnt = 0
        self.exirCnt = 0

    def add_obj(self, id, pt):
        self.frameObject[id] = edict({
            'trackpoints': [pt],
        })

    def parse_obj(self, id, pt, lineStat):
        hisPoints = self.frameObject[id].trackpoints
        if len(hisPoints) > 40:
            del hisPoints[0]
        hisPoints.append(pt)
        if not (len(hisPoints) > 3 and getPointLen(hisPoints[0], hisPoints[-1]) > 10):
            return 
        p1, p2 = hisPoints[-1], hisPoints[-4]
        for i in range(len(lineStat)):
            sd = point2LineDistance(p1, lineStat[i])
            ed = point2LineDistance(p2, lineStat[i])
            if sd * ed < 0 and intersect(
                    p1, p2, lineStat[i][0],
                    lineStat[i][1]):
                ang = vLineAngle([p1, p2], lineStat[i])
                if ang < 180:
                    self.entryCnt += 1
                else:
                    self.exitCnt += 1
                hisPoints.clear()
            elif sd * ed > 0 and abs(sd) > 80 and abs(
                    ed) > 80 and len(
                        hisPoints) > 20:  # 离线比较远时清除旧数据
                for j in range(0, 20):
                    del hisPoints[0]


    def plot_line(self, id, frame):
        num_points = len(self.frameObject[id].trackpoints)
        plotPoints = self.frameObject[id].trackpoints[-min(20, num_points):]
        for i, pt in enumerate(plotPoints):
            cv2.line(frame, pt, plotPoints[max(i - 1, 0)], (0, 0, 255), 2)


class TwoLine:
    def __init__(self):
        pass


if __name__ == "__main__":
    a = [1, 2, 3, 4]
    b = a
    b.append(0)
    print(a)

