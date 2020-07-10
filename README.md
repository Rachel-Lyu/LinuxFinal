# 基于深度学习的实现手写数字及字母识别

本项目利用数据挖掘中的pytorch框架和keras库，根据从大量手写数字和英文字母图像中提取出的原始特征属性，对手写数字和字母进行了计算机算法自动分类，从而达到对手写数字和字母识别的目的。

项目报告见report.md；

手写数字和英文字母图像识别分析源码见number_identify.ipynb和iden_letters.ipynb；

项目前端GUI见main.py；

本项目使用的库如下表
| Package     | Version |
| ----------- | ------- |
| numpy       | 1.18.5  |
| torch       | 1.5.1   |
| Pillow      | 7.1.2   |
| torchvision | 0.6.1   |
| Keras       | 2.4.3   |
| pandas      | 1.0.5   |
| seaborn     | 0.10.1  |
| tensorflow  | 2.2.0   |
| matplotlib  | 3.2.2   |

环境缺失可使用ANACONDA的Environments进行配置



## 文件介绍

`main.py`主程序，可用`python  main.py`命令运行

`models/` 数据探索、网络训练代码和模型参数保存

`utils/` 神经网络的封装调用函数

`examples/`示例图片