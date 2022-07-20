from audioop import reverse
import cv2
from cv2 import floodFill
from cv2 import contourArea
import numpy as np
from matplotlib import pyplot as plt 


def show_resize_img(img,image_type):
  img_resize = cv2.resize(img,None,fx=0.25,fy=0.3)
  cv2.imshow(image_type,img_resize)
  cv2.waitKey(0)

#原圖
image_ori_path = './20220623_Data/0000.jpg'
img_ori = cv2.imread(image_ori_path)
image_type = 'Original'
imge_show = show_resize_img(img_ori,image_type)

#灰階影像 
img_gray = cv2.imread(image_ori_path,0) 
image_type = 'Gray'
imge_show = show_resize_img(img_gray,image_type)

# 二值化處理
ret,thresh = cv2.threshold(img_gray,180,225,cv2.THRESH_BINARY)
image_type = 'Threshold'
imge_show = show_resize_img(thresh,image_type)

# 高斯去模糊
image_gs = cv2.GaussianBlur(thresh,(3,3),0)

# 進行 Sobel
x = cv2.Sobel(image_gs,cv2.CV_16S,1,0)
y = cv2.Sobel(image_gs,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
image_type = 'Sobel'
imge_show = show_resize_img(dst,image_type)

contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

largest_area = 0
largest_area_index = 0

for i in range(0,len(contours)):
  a = contourArea(contours[i])  
  if a> largest_area :
    largest_area = a
    largest_area_index = i

print(largest_area_index)

# 取得輪廓的座標值
x,y,w,h = cv2.boundingRect(contours[largest_area_index]) # x、y 是矩陣左上角點座標
print(x,y,w,h)

# 進行畫圖
img_rect = cv2.rectangle(img_ori, (x, y), (x+w, y+h), (0, 0, 225), 20) # 最後兩個引數為方框的顏色、及 Piexl 寬度
image_type = 'Draw'
imge_show = show_resize_img(img_rect,image_type)

# 
cropped = img_ori[y+1:y+h,x+1:x+w]
height, width = cropped.shape[:2]
cv2.imshow('Cropped Image',cropped)
key = cv2.waitKey(0)

# 按空白鍵
if key == 32:   # ASCII Code
  cv2.destroyAllWindows()
# 按's'存圖
elif key == ord('s'):
  cv2.imwrite('cropped_image.jpg', cropped)
  cv2.destroyAllWindows()
