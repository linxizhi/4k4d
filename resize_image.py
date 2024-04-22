import numpy as np
import os
path="data/my_387/masks"
out_path=path+"0"
import cv2
images_data=os.listdir(path)
for i in images_data:
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_angle=os.path.join(path,i)
    image_out_angle=os.path.join(out_path,i)
    if not os.path.exists(image_out_angle):
        os.makedirs(image_out_angle)
    
    for index,j in enumerate(os.listdir(image_angle)):
        img_final_path=os.path.join(image_angle,j)
        img=cv2.imread(img_final_path)
        img_resized = cv2.resize(img, (256, 256))
        image_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        oo=os.path.join(image_out_angle,j)
        cv2.imwrite(oo,image_gray)
        # if index>1:
        #     break