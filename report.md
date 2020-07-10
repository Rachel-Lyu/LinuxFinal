# **基于深度学习的实现手写数字及字母识别**

苏尧 吕瑞祺 刘茗玮 李一鹭

## **摘要**

数据挖掘是指通过多种算法从海量数据中搜索隐藏于其中有用信息的过程。在无序中寻找有序、在纷乱中发现规律,是数据挖掘的核心价值所在。它主要通过数理统计、在线分析处理、情报检索、机器学习、模式识别等诸多方法来实现既定目标。本文利用数据挖掘中的pytorch框架和keras库,根据从大量手写数字和英文字母图像中提取出的原始特征属性,对手写数字和字母进行了计算机算法自动分类,从而达到对手写数字和字母识别的目的。

## **前言**

神经网络最近的成功推动了模式识别和数据挖掘的进展。许多机器学习任务，例如目标检测、机器翻译、语音识别，曾严重依赖于手工特征工程抽取信息特征集，现在通过各种端到端深度学习范式得到了深刻的改变，例如 CNN、RNN、自编码器。深度学习在许多领域的成功部分归因于快速发展的计算资源（例如GPU），大量训练数据的可用性、以及深度学习从欧几里得数据抽取的潜在表示的有效性。

虽然深度学习有效地捕获了欧几里得数据的隐藏模式，但越来越多的应用以图的形式表示数据。所以本项目从基础的手写数字及字母识别开始，尝试发掘图像与数据间的联系，并入门理解图神经网络内涵。

手写数字识别是计算机视觉领域的一个重要研究方向。目前，仍有大量的手写数字单据需要录入计算机进行管理，比如物流行业中手写快递单据、银行业的手写支票和汇款单、公司账本等，如果能进行智能化的手写数字识别，则可以节省大量的人力。手写数字识别本质上是一个包含10个类别的分类问题，对应0、1、2、…、9共10个数字。同时，手写字母识别也是深度学习在图像识别领域的典型应用场景之一。传统的人工提取图像特征方式逐渐被基于深度学习的深层网络学习特征方式所取代。由于深度学习免去了对图像的复杂前期预处理，可以直接输入原始图像，因而得到广泛应用。将深度学习应用于手写数字及字母识别在一定程度降低了分类错误率。

## **材料与方法**

### **多层感知器**

多层感知器（MLP，Multilayer Perceptron）是一种前馈人工神经网络模型，也叫人工神经网络（ANN，Artificial Neural Network），其将输入的多个数据集映射到单一的输出的数据集上,除了输入输出层，它中间可以有多个隐层，相邻两层的所有神经元两两相连，且网络中不存在回路，如图所示。

![image-20200710104921714](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710104921714.png)

**Pytorch框架**

Pytorch是最近流行的一个深度学习框架，不仅能实现强大的GPU加速，还能支持动态神经网络。Pytorch既可以看作加入了GPU支持的numpy，同时也可以看成一个拥有自动求导功能的强大的深度神经网络。

**Keras库**

Keras是一个由Python编写的开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的高阶应用程序接口，进行深度学习模型的设计、调试、评估、应用和可视化。

Keras在代码结构上由面向对象方法编写，完全模块化并具有可扩展性，并支持现代人工智能领域的主流算法，包括前馈结构和递归结构的神经网络，也可以通过封装参与构建统计学习模型。

**MNIST数据集**

数字的训练集由MNIST数据集提供，而MNIST数据集来自美国国家标准与技术研究所(NIST)。训练集由来自250个不同人手写的数字构成，其中50%是高中学生，50%来自人口普查局的工作人员。测试集也是同样比例的手写数字数据。其中包含 70000 张手写数字的灰度图片，数字范围从 0到9，每一张图片有28\*28个像素点。

**A-Z Handwritten Alphabets in .csv format数据集**

数据集包含26个文件夹（A-Z），其中包含大小为28×28像素的手写图像，图像中的每个字母均居中设置为2020像素框。

每个图像都存储为灰度级

可能还会包含一些嘈杂的图像

图像取自NIST（https://www.nist.gov/srd/nist-special-database-19）和NMIST大型数据集以及少数其他来源，然后如上所述进行了格式化。

**分析流程**

**数字部分**

第一步：导入需要用的库，如torch、numpy等。

第二步：设置批处理尺寸batch\_size大小为20，分别定义训练数据集和训练批处理数据，测试数据集和测试批处理数据。

第三步：创建训练集、验证机、测试集数据的loader

第四步：可视化数据

![image-20200710105019097](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105019097.png)

第五步：定义神经网络结构

![image-20200710105027805](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105027805.png)

第六步：定义损失函数和梯度下降优化器

![image-20200710105035483](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105035483.png)

第七步：训练神经网络

第八步：可视化训练误差和验证误差

![image-20200710105124509](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105124509.png)

第九步：载入验证集损失最低的模型

第十步：对测试集中一个batch的数据进行评估

第十一步：对整个测试集进行评估

![image-20200710105138886](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105138886.png)

**字母部分**

第一步：导入需要用的库

第二步：载入数据集，划分输入输出

第三步：数据集探索、可视化、查看目标的分布状况

![image-20200710105241176](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105241176.png)

第四步：数据集的标准化和修改维度

![image-20200710105356603](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105356603.png)

![image-20200710105426976](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105426976.png)

第五步：定义神经网络结构（keras卷积神经网络）

第六步：选择损失函数和梯度下降优化器

第七步：训练神经网络，保存模型

![image-20200710105502468](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105502468.png)

第八步：可视化训练误差和验证误差

![image-20200710105749445](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105749445.png)

第九步：挑选一个数据进行评估

![image-20200710105843223](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105843223.png)

**结果**

将数据分析的结果或者案例进行展示

数字识别准确率：

![image-20200710105138886](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710105138886.png)

识别错误样例：

![image-20200710110108322](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710110108322.png)

手写字母识别：

测试集准确率：$99.32\%$

**讨论**

从识别错误的数字中，不难发现有以下原因：第一，数字笔画简单，笔画差距相对较小，字形相差不大; 第二，受地域影响，同一数字的写法风格千差万别; 第三，数字间缺乏上下文关联。因此，研究高效率、高精度的手写数字识别算法既具有实际意义又极富挑战性。

从 http://projector.tensorflow.org/ 对MNIST数据集的聚类分析可以看出，由于有些手写数字样本与其他数字有着相近的特点，这些噪声对训练的准确率造成负面影响，也在测试中容易产生误判：

![image-20200710110911570](https://gitee.com/linuxgroup2/finalproj/raw/master/report.assets/image-20200710110911570.png)

随着神经网络深度的增加，在模型训练过程中会出现过拟合、训练低效、泛化能力弱等问题。在后续的文献查阅过程中，发现集成学习可以将多个具有差异性的弱分类器联合起来，通过优势互补来提高集成分类性能，从而将弱分类器提升为强分类器。集成方法的核心就是通过降低个体复杂度，增加群体差异性，来提高集成分类器的预测性能，这为改善深度学习的泛化能力和过拟合问题提供了新的思路。

就比如在本文工作的基础上，可以以MLP作为基分类算法，再训练出多个具有差异性的基分类器，以自适应增强算法(AdaBoost)作为集成策略，构成集成深度学习模型———自适应增强多层感知器，从而寻优确定AdaBMLP的参数。

**参考文献**

[https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format)

**贡献**

苏尧：使用Pytorch框架编写和训练识别手写数字的模型、编写模型接口、修改报告

吕瑞祺：使用keras框架编写和训练识别手写字母的模型、编写模型接口、做报告视频

刘茗玮：

李一鹭：