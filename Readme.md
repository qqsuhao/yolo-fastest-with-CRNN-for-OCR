# 基于YOLO-fastest-xl的OCR

## 项目介绍
- 本项目参考chineseOCR项目，使用了其代码架构和CRNN部分相关的代码与训练模型。
- 基于pytorch进行训练，基于opencv的dnn模块进行推理。
- 本项目的主要区别在于将yolov3的模型替换为yolo-fastest-xl，使得原本大小为200MB左右的模型缩减为3.5MB。
- 发现了chineseOCR中的一处bug：在本项目的crnn/opencv_dnn_detect.py的text_detect函数中的33-35行，改为36-38行。

## 对于yolo-fastest-xl的结构的更改
- 原本的yolo-fastest-xl模型只有两层yolo层，难以应对小目标检测，尤其是文本检测。因此增加一层yolo层，更改后的模型文件存放于yolo/configs/yolo-fastest-xl-3yolo.cfg中。

## 运行方法
- 克隆本项目以后安装相关的运行环境。
- 建议使用pycharm打开本项目，使用vscode的话可能会出现import文件路径错误。
- 在configs.py中进行配置。其他参数不用更改，需要关注第16行。如果使用第16行，则为使用yolov3进行推理，如果使用第17行，则为使用yolo-fastest-xl进行推理。
- 运行demo.py就可以完成文字检测到识别的整个过程。
- 由于yolo-fastest-xl权值文件较小，所以随着项目一起上传。但是yolov3的权值文件太大，没有上传。读者可以从文末的链接下载（包含yolov3的权值和crnn的权值），然后把所有文件放在yolov3-ocr/weights中。
- 其他关键文件说明：
  - yolo/Toonnx.py 该文件将模型加载权值以后将模型转换为onnx文件。demo.py中的推理是使用onnx文件完成的。
  - yolo/weights 和 yolo/checkpoints中存放着相关的权值文件，yolo/checkpoints是训练过程中存放断点的文件夹。
  - yolo/中的大部分文件都和我其他的的yolo-fastest-xl相关的项目一致，可以参考我的其他项目。

## 效果总结
- 在test_samples/0.jpg上进行测试，yolov3需要0.8s左右，yolo-fastest-xl需要0.4s左右。

## 注意
- demo.py中有些参数在不同的模型和测试样本下可能需要进行微调。

## 链接
感谢chineseOCR项目
链接: https://pan.baidu.com/s/1z2ry3Pbi2A_w_lktnQibDg  密码: 03fn