import cv2
import numpy as np
rows = 4
cols = 2
zb = [[0 for _ in range(cols)] for _ in range(rows)]# 用于存储检测到的点
frame=cv2.imread("bug.jpg")
frame_1=frame
blue_lower = np.array([90, 50, 50])  # 深蓝色
blue_upper = np.array([130, 255, 255])  # 浅蓝色

# 转换为HSV颜色空间
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 创建蓝色掩膜
mask = cv2.inRange(hsv, blue_lower, blue_upper)
# 根据二值化图像框选图像轮廓
blue_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 0
detected_count=0
for contour in blue_contours:
    if detected_count >= rows:  # 如果已经存储了4个点，则停止检测
        break
    # 计算一个简单的边界框
    x, y, w, h = cv2.boundingRect(contour)
    # 计算每个封闭的contour的周长，乘0.03赋值给epsilon作为拟合的精度参数
    epsilon = 0.03 * cv2.arcLength(contour, True)
    # 对contour做多边形逼近，epsilon定义了原始轮廓和逼近多边形之间的最大距离，
    # epsilon越小逼近的多边形就越接近原始的轮廓
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # 求多边形边数
    lens = len(approx)
    # 求多边形面积
    area = cv2.contourArea(contour)
    if lens == 6 and 1300< area < 3000:
        # 画出边界框
        img_1 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        zb[detected_count] = [x, y]
        detected_count += 1
    """符合条件的x，y存入数组"""
cv2.imshow("image", frame)
#cv2.waitKey(0)
print(zb)
#cv2.imwrite("binary.png", img_1)
h = 0
n = 0
for h in range(3):
    for n in range(0, 2):
        if np.sum(zb[n]) > np.sum(zb[n + 1]):
            zb[n], zb[n + 1] = zb[n + 1], zb[n]
'''(a,b)=zb[0]
(c,d)=zb[1]
(e,f)=zb[2]'''
if detected_count > 0:
    # 只对实际检测到的点排序
    sorted_points = sorted(zb[:detected_count], key=lambda p: (p[1], p[0]))

    # 更新zb列表
    for i in range(detected_count):
        zb[i] = sorted_points[i]
objectPoints = np.array([[0, 0, 70],
                         [125, 0, 70],
                         [0, 0, 0],
                         [125, 0, 0]], dtype=np.float32)
imagePoints = np.array(zb, dtype=np.float32).reshape(-1, 1, 2)
cameraMatrix = np.array([[969.3421, 641.2600, 0],#641.2600
                         [0, 970.6431, 361.0710],#361.0710
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.array([0.0779, -0.0843, 0, 0], dtype=np.float32).reshape(1,4)
# 使用solvePnP求解相机姿态
_, rvec, tvec = cv2.solvePnP(
    objectPoints,
    imagePoints,
    cameraMatrix,
    distCoeffs
)
# 使用solvePnP求解相机姿态
_, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
axis_points = np.array([
    [0, 0, 0],  # 原点
    [100, 0, 0],  # X轴
    [0, 100, 0],  # Y轴
    [0, 0, 100]  # Z轴
], dtype=np.float32)
projected_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, cameraMatrix, distCoeffs)
projected_axis_points = np.int32(projected_axis_points)
for i in range(len(projected_axis_points)):
    cv2.line(frame_1, tuple(projected_axis_points[0].ravel()), tuple(projected_axis_points[i].ravel()),
             (0, 255, 0), 2)
#cv2.imshow('frame_window1', frame_1)
#cv2.imshow('frame_window2', img_out)
#cv2.imshow('frame_window3', gray)
#cv2.imshow('frame_window4', binary)
cv2.waitKey(0)
