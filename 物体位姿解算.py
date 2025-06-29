import cv2
import numpy as np
# 初始化zb列表 - 使用4x2的嵌套列表存储四个点的坐标
rows = 4
cols = 2
zb = [[0 for _ in range(cols)] for _ in range(rows)]
output_video = "output_video.mp4"
cv2.namedWindow('mp4_out', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mp4_out', 1200, 800)
video = cv2.VideoCapture("yellow2.mp4")
if not video.isOpened():
    print("无法打开视频文件")
    exit()
# 获取视频信息
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
frame_count = 0
# 黄色HSV范围（根据实际情况调整）
yellow_lower = np.array([23, 100, 100])
yellow_upper = np.array([30, 255, 255])
def order_points(pts):
    # 按y坐标升序排序（y值小的在上方）
    y_sorted = pts[np.argsort(pts[:, 1])]
    # 取y坐标最小的两个点（上边点）
    top_points = y_sorted[:2]
    top_points = top_points[np.argsort(top_points[:, 0])]
    tl = top_points[0]  # 左上角（y最小且x最小）
    tr = top_points[1]  # 右上角（y最小但x较大）
    # 取y坐标最大的两个点（下边点）
    bottom_points = y_sorted[2:]
    # 在y坐标最大的两个点中，按x坐标升序排序（x值小的在左侧）
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    bl = bottom_points[0]  # 左下角（y最大但x最小）
    br = bottom_points[1]  # 右下角（y最大且x最大）
    return np.array([tl, tr, br, bl], dtype=np.float32)
while True:
    ret, frame = video.read()
    if not ret:
        print("视频播放完毕")
        break

    frame_count += 1
    frame_1 = frame.copy()

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建黄色掩膜
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 重置检测点
    detected_count = 0
    # 处理每个轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 筛选条件 - 只处理大面积的黄色区域
        if 300000 < area < 600000:
            # 多边形逼近
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            # 计算凸包
            hull = cv2.convexHull(approx)
            # 确保是四边形
            if len(approx) == 4:
                # 确保凸包是凸四边形
                if len(hull) == 4:
                    # 存储顶点
                    quad_pts = approx.reshape(-1, 2)

                    # 对顶点进行排序
                    ordered_pts = order_points(quad_pts)

                    # 存储到zb
                    for i in range(4):
                        zb[i][0] = ordered_pts[i][0]
                        zb[i][1] = ordered_pts[i][1]

                    detected_count = 1  # 只需要检测一个四边形

                    # 绘制四边形轮廓
                    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 3)
    # 只有检测到四边形时才计算位姿
    if detected_count == 1:
        # 定义实际坐标（单位：毫米）
        objectPoints = np.array([
            [0, 0, 0],  # 左上角 (TL)
            [174, 0, 0],  # 右上角 (TR)
            [174, 113, 0],  # 右下角 (BR)
            [0, 113, 0]  # 左下角 (BL)
        ], dtype=np.float32)
        # 图像点坐标
        imagePoints = np.array(zb, dtype=np.float32).reshape(-1, 1, 2)
        cameraMatrix = np.array([
            [964.4264, 0, 644.4063],
            [0, 965.7578, 362.3195],
            [0, 0, 1]], dtype=np.float32)
        distCoeffs = np.array([0.073972, -0.005636, 0.000213, 0.001868, -0.313172], dtype=np.float32)
        # 使用solvePnP求解相机姿态
        _, rvec, tvec = cv2.solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs
        )
        axis_points = np.array([
                [0, 0, 0],  # 原点
                [100, 0, 0],  # X轴
                [0, 100, 0],  # Y轴
                [0, 0, -90]  # Z轴
            ], dtype=np.float32)
        projected_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, cameraMatrix, distCoeffs)
        projected_axis_points = projected_axis_points.reshape(-1, 2).astype(int)
        origin = tuple(projected_axis_points[0])
        for i in range(1, len(projected_axis_points)):
            end_point = tuple(projected_axis_points[i])
            cv2.line(frame, origin, end_point, (0, 0, 255), 4)
    cv2.imshow("mp4_out", frame)
    # 查看掩膜效果以看筛选效果
    #cv2.imshow("Yellow Mask", mask)
    out.write(frame)

    # 按ESC退出
    key = cv2.waitKey(60)
    if key == 27:  # ESC键
        break
    elif key == ord(' '):  # 空格键暂停
        cv2.waitKey(0)
    elif key == ord('s'):  # 保存当前帧
        cv2.imwrite(f"frame_{frame_count}.png", frame)

# 释放资源
video.release()
cv2.destroyAllWindows()
out.release()
