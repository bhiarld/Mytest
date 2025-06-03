import cv2 # 导入视觉库
image = cv2.imread("duihuanzhan.png")# 把引号内的文件名修改为你的图片名
img_1=image
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)
mask_h = cv2.inRange(img_h, 0, 180)
mask_s = cv2.inRange(img_s,0,150)
mask_v = cv2.inRange(img_v,200,255)
mask_h_and_s = cv2.bitwise_and(mask_h, mask_s)
mask = cv2.bitwise_and(mask_h_and_s, mask_v)
img_out = cv2.bitwise_and(image, image, mask = mask)
# 转化为灰度图及全局阈值二值化
gray = cv2.cvtColor(img_out,cv2.COLOR_BGR2GRAY)#转换成灰度图
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
    if lens==6 and 500<area<3000 :
        # 画出边界框
        img_1 = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
cv2.imshow("image", img_1)
#cv2.imwrite("binary.png", img_1)
cv2.waitKey(0)

