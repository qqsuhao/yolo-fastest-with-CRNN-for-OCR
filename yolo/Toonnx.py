# -*- coding:utf8 -*-
# @TIME     : 2021/3/21 21:59
# @Author   : SuHao
# @File     : Toonnx.py


from yolo.models.model import YOLOv3
import onnx
import torch
import cv2
import numpy as np

conifg_path = "./configs/yolo-fastest-xl-3yolo.cfg"
# weights_path = "./weights/yolo-fastest-xl-3yolo.pth"
weights_path = "./checkpoints/yolov3_ckpt_20.pth"
save_path = "./weights/yolo-fastest-xl-3yolo.onnx"
save_path_t7 = "./weights/yolo-fastest-xl-3yolo.t7"


net = YOLOv3(conifg_path)
# If specified we start from checkpoint
if weights_path:
    if weights_path.endswith(".pth"):
        net.load_state_dict(torch.load(weights_path))       # 加载pytorch格式的权重文件
        print("load_state_dict")
    else:
        net.load_darknet_weights(weights_path)          # 加载darknet格式的权重文件。以.weight为后缀
        print("load_darknet_weights")

net.eval()
print('===> Saving models...')
state = {
    'state':net.state_dict(),
}
torch.save(state, save_path_t7)
# net.save_darknet_weights(path=save_path_weights)


#############
imagesize = 608
inputs = torch.rand(1, 3, imagesize, imagesize)
torch.onnx.export(net, inputs, save_path, input_names=["input"], output_names=["outputs0", "outputs1",  "outputs2"],
                  verbose=True, opset_version=11)
model = onnx.load(save_path)
onnx.checker.check_model(model)

# opencv dnn加载
net = cv2.dnn.readNetFromONNX(save_path)
img = inputs.numpy() * 255
img = img[0]
img = img.transpose((1, 2, 0))
img = img.astype('uint8')
blob = cv2.dnn.blobFromImage(img, size=(imagesize, imagesize))      # img 必须是uint8
print(blob.shape)
net.setInput(blob)
out_blob = net.forward(net.getUnconnectedOutLayersNames())
print(len(out_blob))
out_blob = cv2.dnn.imagesFromBlob(out_blob[1])
print(out_blob[0].shape)
