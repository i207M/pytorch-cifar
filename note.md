实现论文中的Resnet v1，使用Option A（即，不使用1x1卷积解决channel数不匹配的问题）。

无quantization：Acc 93.27；原论文93.03

理解：实现一个scaled binary connect

## 疑惑

为什么lr改变时，acc会猛增？

## 收获

数据增强提高generalization

使用cosine lr再高一个点

batch_size大小对acc的影响有好有坏

## 探索

Binarized Neural Networks: Training Neural Networks with Weights and
Activations Constrained to +1 or −1

不仅仅二值化了weights，而且二值化了activations

## 待做实验

- PreAct Multistep (Weight 1e-4, 5e-4)
- PreAct Cosine
- PreAct+BC Cosine
- PreAct+BWN Cosine

(all small batch)
