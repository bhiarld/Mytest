import cv2
image=cv2.imread('duihuanzhan.png')
img_1=image
# 转化为灰度图及全局阈值二值化
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转换成灰度图
retval, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
# 显示图像
#cv2.imshow('original', image)
#cv2.imshow('binary', binary)
# 根据二值化图像框选图像轮廓
contours,_ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    # 计算一个简单的边界框
    x, y, w, h = cv2.boundingRect(contour)
    # 计算每个封闭的contour的周长，乘0.03赋值给epsilon作为拟合的精度参数
    epsilon = 0.03 * cv2.arcLength(contour, True)
    # 对contour做多边形逼近，epsilon定义了原始轮廓和逼近多边形之间的最大距离，
    # epsilon越小逼近的多边形就越接近原始的轮廓
    approx = cv2.approxPolyDP(contour, epsilon, True)
    #求多边形边数
    lens = len(approx)
    #求多边形面积
    area = cv2.contourArea(contour)
    if lens==6:
        if 1000<area<3000:
            # 画出边界框
            img_1 = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
cv2.imshow("image", img_1)
cv2.imwrite("binary.png", img_1)
cv2.waitKey(0)

