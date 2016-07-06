# GRU-CNN
本模型利用GRU或CNN对存在某种关系的两个句子进行建模。模型大致结构为利用GRU（[Cho et al., 2014b](http://arxiv.org/abs/1406.1078), RNN中的一种）或CNN学习句子表示（两个句子不共享一套参数），再用一层神经网络学习两个句子的联合表示，最后利用一个sigmoid层对两个句子进行打分，输入关系强弱的值，训练方法采用正负例训练。模型结构如下图所示

![model](model.jpg?raw=true "model")

该模型可用于连贯性任务（相当于窗口取2，只看前后两句话），答案选取任务（正对QA数据集，问-答对正好是存在关系的两个句子），以及对话质量评估（针对单轮对话，有点类似于一问一答那种形式，也是两个句子，模型评价对话的质量）。

========================================================

输入文件格式
--------------------------------------------------------
由于模型目前仅对两个句子进行建模，所以输入文件为两个文件，一个文件存储第一句，另一个文件存储下句（对应存储，对于中文需要分词，按空格隔开）。注意：除了修改main.py中的file1（第一句）和file2（第二句）以外，还需要ReadDate函数中的数值来确定训练数据和测试数据的规模。

运行说明
--------------------------------------------------------
在命令行中输入：

    python main.py

实验结果
--------------------------------------------------------
实验所用的数据为100W个对话对，有点类似于QA语料。实验设置为90W用于训练，10W用于测试，测试数据中5W为正例，5W为负例。

**Iter 0:**

>cost: 3.025

>cost time: 195146.85 s

>Test...

>Accuracy: 0.75045

>Test Done

**Iter 1:**

>cost: 2.428

>cost time: 190828.23 s

>Test...

>Accuracy: **0.79202**

>Test Done

**Iter 2:**

>cost: 2.255

>cost time: 187904.05 s

>Test...

>Accuracy: 0.76932

>Test Done

**Iter 3:**

>cost: 2.169

>cost time: 155178.83 s

>Test...

>Accuracy: 0.78361

>Test Done
