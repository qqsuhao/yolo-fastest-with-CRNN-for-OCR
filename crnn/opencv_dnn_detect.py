from config import yoloCfg,yoloWeights,opencvFlag
from config import AngleModelPb,AngleModelPbtxt
from config import IMGSIZE
from PIL import Image
import numpy as np
import cv2


def text_detect(img, TEXT_PROPOSALS_MIN_SCORE):
    if yoloWeights.endswith("onnx"):
        textNet = cv2.dnn.readNetFromONNX(yoloWeights)
    else:
        textNet  = cv2.dnn.readNetFromDarknet(yoloCfg, yoloWeights)##文字定位
    thresh = TEXT_PROPOSALS_MIN_SCORE
    img_height,img_width = img.shape[:2]
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=IMGSIZE,swapRB=True ,crop=False);
    textNet.setInput(inputBlob/255.0)
    outputName = textNet.getUnconnectedOutLayersNames()
    outputs = textNet.forward(outputName)
    if yoloWeights.endswith("onnx"):
        out = []
        for i in range(len(outputs)):
            tmp = cv2.dnn.imagesFromBlob(outputs[i])[0]
            tmp[:, [0, 2]] = tmp[:, [0, 4]] / img_width
            tmp[:, [1, 3]] = tmp[:, [1, 3]] / img_height
            out.append(tmp)
        outputs = out
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:              # * 根据置信度门限筛选检测结果
            for detection in output:
                # scores = detection[5:]
                # class_id = np.argmax(scores)
                # confidence = scores[class_id]
                scores = detection[4]
                confidence = scores
                class_id = np.argmax(detection[5:])
                if confidence > thresh:
                    center_x = int(detection[0] * img_width)
                    center_y = int(detection[1] * img_height)
                    width = int(detection[2] * img_width)
                    height = int(detection[3] * img_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    if left + width > img_width-1:
                        left = img_width-1-width
                    if top + height > img_height-1:
                        top = img_height-1-height
                    if class_id==1:
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([left, top,left+width, top+height ])
        
    return np.array(boxes),np.array(confidences)



def angle_detect_dnn(img,adjust=True):
    """
    文字方向检测
    """
    angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb, AngleModelPbtxt)##dnn 文字方向检测
    h,w = img.shape[:2]
    ROTATE = [0,90,180,270]
    if adjust:
       thesh = 0.05
       xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
       img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
    inputBlob = cv2.dnn.blobFromImage(img, 
                                      scalefactor=1.0, 
                                      size=(224, 224),
                                      swapRB=True ,
                                      mean=[103.939,116.779,123.68],crop=False);
    angleNet.setInput(inputBlob)
    pred = angleNet.forward()
    index = np.argmax(pred,axis=1)[0]
    return ROTATE[index]


def angle_detect_tf(img,adjust=True):
    """
    文字方向检测
    """
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.python.platform import gfile
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile(AngleModelPb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    inputImg =  sess.graph.get_tensor_by_name('input_1:0')
    predictions = sess.graph.get_tensor_by_name('predictions/Softmax:0')
    keep_prob = tf.placeholder(tf.float32)
    h,w = img.shape[:2]
    ROTATE = [0,90,180,270]
    if adjust:
       thesh = 0.05
       xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
       img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
    img = cv2.resize(img,(224,224))
    img = img[..., ::-1].astype(np.float32)
        
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img          = np.array([img])
    out = sess.run(predictions, feed_dict={inputImg: img,
                                              keep_prob: 0
                                             })
    index = np.argmax(out,axis=1)[0]
    return ROTATE[index]


def angle_detect(img,adjust=True):
    """
    文字方向检测
    """
    if opencvFlag=='keras':
        return angle_detect_tf(img,adjust=adjust)
    else:
        return angle_detect_dnn(img,adjust=adjust)