# -*- coding:utf8 -*-
# @TIME     : 2021/8/21 21:26
# @Author   : SuHao
# @File     : demo.py

import os
import torch
from apphelper.image import xy_rotate_box,box_rotate,solve
import model
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt


def plot_box(img,boxes):
    blue = (0, 0, 0) #18
    tmp = np.copy(img)
    for box in boxes:
        cv2.rectangle(tmp, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), blue, 1) #19

    return Image.fromarray(tmp)


def plot_boxes(img,angle, result,color=(0,0,0)):
    tmp = np.array(img)
    c = color
    h,w = img.shape[:2]
    thick = int((h + w) / 300)
    i = 0
    if angle in [90,270]:
        imgW,imgH = img.shape[:2]

    else:
        imgH,imgW= img.shape[:2]

    for line in result:
        cx =line['cx']
        cy = line['cy']
        degree =line['degree']
        w  = line['w']
        h = line['h']

        x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(cx, cy, w, h, degree/180*np.pi)

        x1,y1,x2,y2,x3,y3,x4,y4 = box_rotate([x1,y1,x2,y2,x3,y3,x4,y4],angle=(360-angle)%360,imgH=imgH,imgW=imgW)
        cx  =np.mean([x1,x2,x3,x4])
        cy  = np.mean([y1,y2,y3,y4])
        cv2.line(tmp,(int(x1),int(y1)),(int(x2),int(y2)),c,1)
        cv2.line(tmp,(int(x2),int(y2)),(int(x3),int(y3)),c,1)
        cv2.line(tmp,(int(x3),int(y3)),(int(x4),int(y4)),c,1)
        cv2.line(tmp,(int(x4),int(y4)),(int(x1),int(y1)),c,1)
        mess=str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)),0, 1e-3 * h, c, thick // 2)
        i+=1
    return Image.fromarray(tmp).convert('RGB')


if __name__ == "__main__":
    GPUID='0'##调用GPU序号
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
    p = './test_samples/img.jpeg'
    img = cv2.imread(p)
    h,w = img.shape[:2]
    timeTake = time.time()
    _,result,angle= model.model(img,
                                detectAngle=False,##是否进行文字方向检测
                                config=dict(MAX_HORIZONTAL_GAP=50,##字符之间的最大间隔，用于文本行的合并
                                            MIN_V_OVERLAPS=0.6,
                                            MIN_SIZE_SIM=0.6,
                                            TEXT_PROPOSALS_MIN_SCORE=0.1,
                                            TEXT_PROPOSALS_NMS_THRESH=0.3,
                                            TEXT_LINE_NMS_THRESH = 0.7,##文本行之间测iou值
                                            ),
                                leftAdjust=True,##对检测的文本行进行向左延伸
                                rightAdjust=True,##对检测的文本行进行向右延伸
                                alph=0.01,##对检测的文本行进行向右、左延伸的倍数
                                )
    timeTake = time.time()-timeTake
    print('It take:{}s'.format(timeTake))
    for line in result:
        print(line['text'])
    out = plot_boxes(img, angle, result, color=(0,0,0))
    plt.figure()
    plt.imshow(out)
    plt.xticks([])
    plt.yticks([])
    plt.show()

