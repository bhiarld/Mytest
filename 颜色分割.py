import cv2 # 导入视觉库
img_bgr = cv2.imread("chuang.png")# 把引号内的文件名修改为你的图片名
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)
mask_h = cv2.inRange(img_h, 166, 175)
mask_s = cv2.inRange(img_s,0,255)
mask_v = cv2.inRange(img_v,0,255)
mask_h_and_s = cv2.bitwise_and(mask_h, mask_s)
mask = cv2.bitwise_and(mask_h_and_s, mask_v)
img_out = cv2.bitwise_and(img_bgr, img_bgr, mask = mask)
cv2.imshow("img", img_out)
cv2.imwrite("img_out.png", img_out)
cv2.waitKey(0) # 显示图片直到有任何键盘点击操作
