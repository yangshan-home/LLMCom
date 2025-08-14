Semi-supervised Graph Convolutional Community Detection Empowered by Large Language Models

<img width="873" height="630" alt="image" src="https://github.com/user-attachments/assets/54c4b5df-5ba9-4dfb-a2e3-801a05ca4e1e" />

Overview
1、这篇文章的研究目标是利用LLMs强大的理解和学习能力为社区检测进行赋能，从而提高社区检测的效果。注意，社区检测分为重叠社区检测和非重叠社区检测，所以本文直接统一了这两种任务，直接使用一个模型LLMCom来解决这两个问题。
2、具体如何赋能以及检测重叠和非重叠社区可以看论文。
3、从我的角度客观的评价这篇论文：还有一些方面可以优化。比如：不需要预先知道社区数量，采用一些方法来解决这个社区数量预置问题；探索如何直接利用LLMs生成重叠或非重叠社区。

Usage
1、准备好数据集（均是公开数据集，实验部分有描述）；
2、利用utils.py生成prompt
3、利用llms.py生成节点的embedding
4、利用mian.py训练模型

环境参考论文的4.1.1节。
