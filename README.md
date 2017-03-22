# GRU-CNN
本模型利用GRU或CNN对存在某种关系的两个句子进行建模。模型大致结构为利用GRU（[Cho et al., 2014b](http://arxiv.org/abs/1406.1078), RNN中的一种）或CNN学习句子表示（两个句子不共享一套参数），再用一层神经网络学习两个句子的联合表示，最后利用一个sigmoid层对两个句子进行打分，输入关系强弱的值，训练方法采用正负例训练。模型结构如下图所示

![model](model.jpg?raw=true "model")

该模型可用于：连贯性任务（相当于窗口取2，只看前后两句话）；答案选取任务（针对QA数据集，问-答对正好是具有关联的两个句子）；以及对话质量评估（针对单轮对话，有点类似于一问一答那种形式（也是两个句子），模型评价对话的质量，即评价在聊天机器人系统中生成的对话质量如何）。

--------------------------------------------------------

## 输入文件格式
由于模型目前仅对两个句子进行建模，所以输入文件为两个文件，一个文件存储第一句，另一个文件存储下句（对应存储，对于中文需要分词，按空格隔开）。注意：除了修改main.py中的file1（第一句）和file2（第二句）以外，还需要修改ReadDate函数中的数值来确定训练数据和测试数据的规模。

## 模型参数
main.py文件里面有以下参数可以设定：
- margin：正负例得分间隔
- iter：总共迭代次数
- learning_rate：学习率
- test_freq：每迭代多少次进行一次测试
- h_dim：隐层维度，即句子向量的维度
- vocab_size：词表大小，选取最高频的N个词
- w_dim：词向量维度
- neg_sample：负例采样的数目
- up_dim：句子联合表示的向量维度
- CNN_Flag：是否使用CNN模型，为False时不使用（使用GRU模型）
- save_file：保存测试结果的文件名

## 运行说明
在命令行中输入：

    python main.py

## 实验结果
实验所用的数据为100W个对话对，有点类似于QA语料。实验设置为90W用于训练，10W用于测试，测试数据中5W为正例，5W为负例，使用GRU模型。实验结果如下：

**Iter 0:**

    cost: 3.025
    cost time: 195146.85 s
    Test...
    Accuracy: 0.75045
    Test Done

**Iter 1:**

    cost: 2.428
    cost time: 190828.23 s
    Test...
    Accuracy: **0.79202**
    Test Done

**Iter 2:**

    cost: 2.255
    cost time: 187904.05 s
    Test...
    Accuracy: 0.76932
    Test Done

**Iter 3:**

    cost: 2.169
    cost time: 155178.83 s
    Test...
    Accuracy: 0.78361
    Test Done

使用CNN模型，实验结果如下：

**Iter 0:**

    cost: 0.998
    cost time: 137159.67 s
    Test...
    Accuracy: 0.68731
    Test Done

**Iter 1:**

    cost: 0.753
    cost time: 72665.73 s
    Test...
    Accuracy: 0.7221
    Test Done

**Iter 2:**

    cost: 0.737
    cost time: 68464.48 s
    Test...
    Accuracy: **0.75117**
    Test Done
