
# Awesome Semi-Supervised Learning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yassouali/awesome-semi-supervised-learning/graphs/commit-activity)


<p align="center">
  <img width="300" src="https://i.imgur.com/Ky2jxnj.png" "Awesome!">
</p>

A curated list of awesome Semi-Supervised Learning resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-self-supervised-learning](https://github.com/jason718/awesome-self-supervised-learning).

## Background

# [<img src="https://i.imgur.com/xXi9N40.png">](https://github.com/yassouali/awesome-semi-supervised-learning/)

#### What is Semi-Supervised Learning?
It is a special form of classification. Traditional classifiers use only labeled data (feature / label pairs)
to train. Labeled instances however are often difficult, expensive, or time consuming to obtain, as they require the efforts
of experienced human annotators. Meanwhile unlabeled data may be relatively easy to collect,
but there has been few ways to use them.  **Semi-supervised learning** addresses this problem by
using large amount of unlabeled data, together with the labeled data, to build better classifiers.
Because semi-supervised learning requires less human effort and gives higher accuracy, it is of great interest both in theory and in practice.

#### How many semi-supervised learning methods are there?
Many. Some often-used methods include: EM with generative mixture models, self-training, consistency regularization,
co-training, transductive support vector machines, and graph-based methods.
And with the advent of deep learning, the majority of these methods were adapted and intergrated
into existing deep learning frameworks to take advantage of unlabled data.

#### How do semi-supervised learning methods use unlabeled data?
Semi-supervised learning methods use unlabeled data to either modify or reprioritize hypotheses obtained
from labeled data alone. Although not all methods are probabilistic, it is easier to look at methods that
represent hypotheses by *p(y|x)*, and unlabeled data by *p(x)*. Generative models have common parameters
for the joint distribution *p(x,y)*.  It is easy to see that *p(x)* influences *p(y|x)*. 
Mixture models with EM is in this category, and to some extent self-training.
Many other methods are discriminative, including transductive SVM, Gaussian processes, information regularization,
graph-based and the majority of deep learning based methods.
Original discriminative training cannot be used for semi-supervised learning, since *p(y|x)* is estimated ignoring *p(x)*. To solve the problem,
*p(x)* dependent terms are often brought into the objective function, which amounts to assuming *p(y|x)* and *p(x)* share parameters

(source: [SSL Literature Survey.](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf))

 <figure>
  <p align="center">
    <img src="https://i.imgur.com/PJ340SK.png" width="600">
    <figcaption>An example of the influence of unlabeled data in semi-supervised learning.
    (Image source: <a href="https://en.wikipedia.org/wiki/Semi-supervised_learning">Wikipedia</a>)
    </figcaption>
  </p>
</figure> 

## Contributing

If you find any errors, or you wish to add some papers, please feel free to contribute to this list by contacting [me](https://yassouali.github.io/) or by creating a [pull request](https://github.com/yassouali/awesome-semi-supervised-learning/pulls) using the following Markdown format:

```markdown
- Paper Name. 
  [[pdf]](link) 
  [[code]](link)
  - Author 1, Author 2, and Author 3. *Conference Year*
```

## Table of contents

  - [Books](#books)
  - [Surveys & Overview](#surveys--overview)
  - [Computer Vision](#computer-vision)
    - [Image Classification](#image-classification)
    - [Semantic and Instance Segmentation](#semantic-and-instance-segmentation)
    - [Object Detection](#object-detection)
    - [Other tasks](#other-tasks)
  - [NLP](#nlp)
  - [Generative Models & Tasks](#generative-models--tasks)
  - [Graph Based SSL](#graph-based-ssl)
  - [Theory](#theory)
  - [Reinforcement Learning, Meta-Learning & Robotics](#reinforcement-learning-meta-learning--robotics)
  - [Regression](#regression)
  - [Other](#other)
  - [Talks](#talks)
  - [Thesis](#thesis)
  - [Blogs](#blogs)


## Books

- Semi-Supervised Learning Book.
  [[pdf]](http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf)
  - Olivier Chapelle, Bernhard Schölkopf, Alexander Zien. *IEEE Transactions on Neural Networks 2009*


## Codebase

- Unified SSL Benchmark: A Unified Semi-supervised learning Benchmark for CV, NLP, and Audio.
[[code]](https://github.com/microsoft/Semi-supervised-learning)

- TorchSSL: A PyTorch-based Toolbox for Semi-Supervised Learning.
[[code]](https://github.com/TorchSSL/TorchSSL)


## Surveys & Overview

- Realistic Evaluation of Deep Semi-Supervised Learning Algorithms.
  [[pdf]](https://arxiv.org/abs/1804.09170)
  [[code]](https://github.com/brain-research/realistic-ssl-evaluation)
  - Avital Oliver, Augustus Odena, Colin Raffel, Ekin D. Cubuk, Ian J. Goodfellow. *NeurIPS 2018*

- Semi-Supervised Learning Literature Survey.
  [[pdf]](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf)
  - Xiaojin Zhu. *2008*

- An Overview of Deep Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/2006.05278)
  - Yassine Ouali, Céline Hudelot, Myriam Tami. *2020*

- A survey on semi-supervised learning.
  [[pdf]](https://link.springer.com/content/pdf/10.1007/s10994-019-05855-6.pdf)
  - Jesper E Van Engelen, Holger H Hoos. *2020*

- A Survey on Deep Semi-Supervised Learning
  [[pdf]](https://arxiv.org/pdf/2103.00550.pdf)
  - Xiangli Yang, Zixing Song, Irwin King. *2021*



## Computer Vision
Note that for Image and Object segmentation tasks, we also include weakly-supervised
learning methods, that uses weak labels (eg, image classes) for detection and segmentation.


### Image Classification [Here](img_classification.md)



### Semantic and Instance Segmentation

#### 2022

- Semi-Supervised Video Semantic Segmentation With Inter-Frame Feature Reconstruction.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhuang_Semi-Supervised_Video_Semantic_Segmentation_With_Inter-Frame_Feature_Reconstruction_CVPR_2022_paper.pdf)
  [[code]](https://github.com/hzhupku/SemiSeg-AEL](https://github.com/jfzhuang/IFR)
  - Jiafan Zhuang, Zilei Wang, Yuan Gao. *CVPR 2022*

- TWIST: Two-Way Inter-Label Self-Training for Semi-Supervised 3D Instance Segmentation.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chu_TWIST_Two-Way_Inter-Label_Self-Training_for_Semi-Supervised_3D_Instance_Segmentation_CVPR_2022_paper.pdf)
  - Ruihang Chu, Xiaoqing Ye, Zhengzhe Liu, Xiao Tan, Xiaojuan Qi, Chi-Wing Fu, Jiaya Jia. *CVPR 2022*

- Semi-supervised Semantic Segmentation with Error Localization Network.
  [[pdf]](https://arxiv.org/abs/2204.02078)
  - Donghyeon Kwon, Suha Kwak. *CVPR 2022*

- UCC: Uncertainty guided Cross-head Co-training for Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2205.10334)
  - Jiashuo Fan, Bin Gao, Huan Jin, Lihui Jiang. *CVPR 2022*

- Perturbed and Strict Mean Teachers for Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2111.12903)
  [[code]](https://github.com/yyliu01/PS-MT)
  - Yuyuan Liu, Yu Tian, Yuanhong Chen, Fengbei Liu, Vasileios Belagiannis, Gustavo Carneiro. *CVPR 2022*

- Cross-Patch Dense Contrastive Learning for Semi-Supervised Segmentation of Cellular Nuclei in Histopathologic Images.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Cross-Patch_Dense_Contrastive_Learning_for_Semi-Supervised_Segmentation_of_Cellular_Nuclei_CVPR_2022_paper.html)
  - Huisi Wu, Zhaoze Wang, Youyi Song, Lin Yang, Jing Qin. *CVPR 2022*

- Unbiased Subclass Regularization for Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2203.10026)
  - Dayan Guan, Jiaxing Huang, Aoran Xiao, Shijian Lu. *CVPR 2022*

- Noisy Boundaries: Lemon or Lemonade for Semi-supervised Instance Segmentation?.
  [[pdf]](https://arxiv.org/abs/2203.13427)
  - Zhenyu Wang, Yali Li, Shengjin Wang. *CVPR 2022*

- Rethinking Bayesian Deep Learning Methods for Semi-Supervised Volumetric Medical Image Segmentation.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Rethinking_Bayesian_Deep_Learning_Methods_for_Semi-Supervised_Volumetric_Medical_Image_CVPR_2022_paper.html)
  - Jianfeng Wang, Thomas Lukasiewicz. *CVPR 2022*

- Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels.
  [[pdf]](https://arxiv.org/abs/2203.03884)
  - Yuchao Wang, Haochen Wang, Yujun Shen, Jingjing Fei, Wei Li, Guoqiang Jin, Liwei Wu, Rui Zhao, Xinyi Le. *CVPR 2022*

#### 2021

- Self-Paced Contrastive Learning for Semi-supervised Medical Image Segmentation with Meta-labels.
  [[pdf]](https://arxiv.org/abs/2107.13741)
  - Jizong Peng, Ping Wang, Chrisitian Desrosiers, Marco Pedersoli. *NeurIPS 2021*

- Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning.
  [[pdf]](https://arxiv.org/abs/2110.05474)
  [[code]](https://github.com/hzhupku/SemiSeg-AEL)
  - Hanzhe Hu, Fangyun Wei, Han Hu, Qiwei Ye, Jinshi Cui, Liwei Wang. *NeurIPS 2021*

- Semi-Supervised Semantic Segmentation With Pixel-Level Contrastive Learning From a Class-Wise Memory Bank.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/html/Alonso_Semi-Supervised_Semantic_Segmentation_With_Pixel-Level_Contrastive_Learning_From_a_Class-Wise_ICCV_2021_paper.html)
  - Iñigo Alonso, Alberto Sabater, David Ferstl, Luis Montesano, Ana C. Murillo. *ICCV 2021*

- Pixel Contrastive-Consistent Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2108.09025)
  - Yuanyi Zhong, Bodi Yuan, Hong Wu, Zhiqiang Yuan, Jian Peng, Yu-Xiong Wang. *ICCV 2021*

- A Simple Baseline for Semi-supervised Semantic Segmentation with Strong Data Augmentation.
  [[pdf]](https://arxiv.org/abs/2104.07256)
  - Jianlong Yuan, Yifan Liu, Chunhua Shen, Zhibin Wang, Hao Li. *ICCV 2021*

- Warp-Refine Propagation: Semi-Supervised Auto-labeling via Cycle-consistency.
  [[pdf]](https://arxiv.org/abs/2109.13432)
  - Aditya Ganeshan, Alexis Vallet, Yasunori Kudo, Shin-ichi Maeda, Tommi Kerola, Rares Ambrus, Dennis Park, Adrien Gaidon. *ICCV 2021*

- Re-Distributing Biased Pseudo Labels for Semi-Supervised Semantic Segmentation: A Baseline Investigation.
  [[pdf]](http://arxiv.org/abs/2107.11279)
  [[code]](https://github.com/CVMI-Lab/DARS)
  - Ruifei He, Jihan Yang, Xiaojuan Qi. *ICCV 2021*

- Collaborative and Adversarial Learning of Focused and Dispersive Representations for Semi-Supervised Polyp Segmentation.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Collaborative_and_Adversarial_Learning_of_Focused_and_Dispersive_Representations_for_ICCV_2021_paper.html)
  - Huisi Wu, Guilian Chen, Zhenkun Wen, Jing Qin. *ICCV 2021*

- Guided Point Contrastive Learning for Semi-Supervised Point Cloud Semantic Segmentation.
  [[pdf]](http://arxiv.org/abs/2110.08188)
  - Li Jiang, Shaoshuai Shi, Zhuotao Tian, Xin Lai, Shu Liu, Chi-Wing Fu, Jiaya Jia. *ICCV 2021*

- Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning.
  [[pdf]](https://arxiv.org/abs/2110.05474)
  [[code]](https://github.com/hzhupku/SemiSeg-AEL)
  - Hanzhe Hu, Fangyun Wei, Han Hu, Qiwei Ye, Jinshi Cui, Liwei Wang. *NeurIPS 2021*

- Contrastive Semi-Supervised Learning for 2D Medical Image Segmentation.
  [[pdf]](https://arxiv.org/abs/2106.06801)
  - Prashant Pandey, Ajey Pai, Nisarg Bhatt, Prasenjit Das, Govind Makharia, Prathosh AP, Mausam. *MICCAI 2021*

- ATSO: Asynchronous Teacher-Student Optimization for Semi-Supervised Image Segmentation.
  [[pdf]](https://arxiv.org/abs/2006.13461)
  [[code]](https://cvir.github.io/TCL/)
  - Xinyue Huo, Lingxi Xie, Jianzhong He, Zijie Yang, Wengang Zhou, Houqiang Li, Qi Tian. *CVPR 2021*

- Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization.
  [[pdf]](https://arxiv.org/abs/2104.05833)
  [[code]](https://nv-tlabs.github.io/semanticGAN/)
  - Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, Sanja Fidler. *CVPR 2021*

- ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2106.05095)
  [[code]](https://github.com/LiheYoung/ST-PlusPlus) 
  - Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi and Yang Gao. *CVPR 2021*.

- Learning Dynamic Network Using a Reuse Gate Function in Semi-supervised Video Object Segmentation.
  [[pdf]](https://arxiv.org/abs/2012.11655)
  [[code]](https://github.com/HYOJINPARK/Reuse_VOS)
  - Hyojin Park, Jayeon Yoo, Seohyeong Jeong, Ganesh Venkatesh, Nojun Kwak. *CVPR 2021*

- Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2103.08896)
  - Jungbeom Lee, Eunji Kim, Sungroh Yoon. *CVPR 2021*

- Semi-Supervised Semantic Segmentation With Cross Pseudo Supervision.
  [[pdf]](https://arxiv.org/abs/2106.01226)
  [[code]](https://github.com/charlesCXK/TorchSemiSeg) 
  - Xiaokang Chen, Yuhui Yuan, Gang Zeng, Jingdong Wang. *CVPR 2021*.

- Semi-supervised Semantic Segmentation with Directional Context-aware Consistency.
  [[pdf]](https://jiaya.me/papers/semiseg_cvpr21.pdf)
  [[code]](https://github.com/Jia-Research-Lab/Context-Aware-Consistency) 
  - Xin Lai, Zhuotao Tian, Li Jiang, Shu Liu, Hengshuang Zhao, Liwei Wang, Jiaya Jia. *CVPR 2021*

- ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/pdf/2007.07936)
  [[code]](https://github.com/WilhelmT/ClassMix) 
  - Viktor Olsson, Wilhelm Tranheden, Juliano Pinto, Lennart Svensson. *WACV 2021*

#### 2020

- Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum.
  [[pdf]](https://arxiv.org/pdf/2004.08514)
  [[code]](https://github.com/voldemortX/DST-CBC) 
  - Zhengyang Feng, Qianyu Zhou, Guangliang Cheng, Xin Tan, Jianping Shi, Lizhuang Ma. *Preprint 2020*

- PseudoSeg: Designing Pseudo Labels for Semantic Segmentation.
  [[pdf]](https://arxiv.org/pdf/2010.09713)
  [[code]](https://github.com/googleinterns/wss) 
  - Yuliang Zou, Zizhao Zhang, Han Zhang, Chun-Liang Li, Xiao Bian, Jia-Bin Huang, Tomas Pfister. *Preprint 2020*

- Semi-supervised semantic segmentation needs strong, varied perturbations.
  [[pdf]](https://arxiv.org/pdf/1906.01916)
  [[code]](https://github.com/Britefury/cutmix-semisup-seg) 
  - Geoff French, Samuli Laine, Timo Aila, Michal Mackiewicz, Graham Finlayson. *BMVC 2020*

- Guided Collaborative Training for Pixel-wise Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/pdf/2008.05258)
  [[code]](https://github.com/ZHKKKe/PixelSSL) 
  - Zhanghan Ke, Di Qiu, Kaican Li, Qiong Yan, Rynson W.H. Lau. *ECCV 2020*

- Naive-Student: Leveraging Semi-Supervised Learning in Video Sequences for Urban Scene Segmentation.
  [[pdf]](https://arxiv.org/abs/2005.10266)
  - Liang-Chieh Chen, Raphael Gontijo Lopes, Bowen Cheng, Maxwell D. Collins, Ekin D. Cubuk, Barret Zoph, Hartwig Adam, Jonathon Shlens. *ECCV 2020*

- A Three-Stage Self-Training Framework for Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/pdf/2012.00827)
  - Rihuan Ke, Angelica Aviles-Rivero, Saurabh Pandey, Saikumar Reddy, Carola-Bibiane Schönlieb. *CVPR 2020*

- Learning Saliency Propagation for Semi-Supervised Instance Segmentation.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Learning_Saliency_Propagation_for_Semi-Supervised_Instance_Segmentation_CVPR_2020_paper.pdf)
  [[code]](https://github.com/ucbdrive/ShapeProp) 
  - Yanzhao Zhou, Xin Wang, Jianbin Jiao, Trevor Darrell, Fisher Yu. *CVPR 2020*

- Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/2004.04581)
  [[code]](https://github.com/YudeWang/SEAM) 
  - Yude Wang, Jie Zhang, Meina Kan, Shiguang Shan, Xilin Chen. *CVPR 2020*

- Semi-Supervised Semantic Segmentation with Cross-Consistency Training.
  [[pdf]](https://arxiv.org/abs/2003.09005)
  [[code]](https://github.com/yassouali/CCT) 
  - Yassine Ouali, Céline Hudelot, Myriam Tami. *CVPR 2020*

- Semi-Supervised Semantic Image Segmentation with Self-correcting Networks.
  [[pdf]](https://arxiv.org/abs/1811.07073) 
  - Mostafa S. Ibrahim, Arash Vahdat, Mani Ranjbar, William G. Macready. *CVPR 2020*

#### 2019

- Semi-Supervised Semantic Segmentation with High- and Low-level Consistency.
  [[pdf]](https://arxiv.org/abs/1908.05724)
  [[code]](https://github.com/sud0301/semisup-semseg) 
  - Sudhanshu Mittal, Maxim Tatarchenko, Thomas Brox. *TPAMI 2019*

- CapsuleVOS: Semi-Supervised Video Object Segmentation Using Capsule Routing.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Duarte_CapsuleVOS_Semi-Supervised_Video_Object_Segmentation_Using_Capsule_Routing_ICCV_2019_paper.pdf)
  [[code]](https://github.com/KevinDuarte/CapsuleVOS) 
  - Kevin Duarte, Yogesh S. Rawat, Mubarak Shah. *ICCV 2019*

- Universal Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1811.10323)
  [[code]](https://github.com/tarun005/USSS_ICCV19) 
  - Tarun Kalluri, Girish Varma, Manmohan Chandraker, C V Jawahar. *ICCV 2019*

- Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations.
  [[pdf]](https://arxiv.org/abs/1904.05044)
  [[code]](https://github.com/jiwoon-ahn/irn) 
  - Jiwoon Ahn, Sunghyun Cho, Suha Kwak. *CVPR 2019*

- FickleNet: Weakly and Semi-Supervised Semantic Image Segmentation Using Stochastic Inference.
  [[pdf]](https://arxiv.org/abs/1902.10421) 
  - Jungbeom Lee, Eunji Kim, Sungmin Lee, Jangho Lee, Sungroh Yoon. *CVPR 2019*

#### 2018

- Adversarial Learning for Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1806.04659)
  [[code]](https://github.com/hfslyc/AdvSemiSeg)
  - Wei-Chih Hung, Yi-Hsuan Tsai, Yan-Ting Liou, Yen-Yu Lin, Ming-Hsuan Yang. *BMVC 2018*

- Weakly-Supervised Semantic Segmentation by Iteratively Mining Common Object Features.
  [[pdf]](https://arxiv.org/abs/1806.04659)
  - Xiang Wang, Shaodi You, Xi Li, Huimin Ma. *CVPR 2018*

- Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1803.10464)
  [[code]](https://github.com/jiwoon-ahn/psa) 
  - Jiwoon Ahn, Suha Kwak. *CVPR 2018*

- Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach.
  [[pdf]](https://arxiv.org/abs/1703.08448) 
  - Yunchao Wei, Jiashi Feng, Xiaodan Liang, Ming-Ming Cheng, Yao Zhao, Shuicheng Yan. *CVPR 2018*

- Tell Me Where to Look: Guided Attention Inference Network.
  [[pdf]](https://arxiv.org/abs/1802.10171) 
  - Kunpeng Li, Ziyan Wu, Kuan-Chuan Peng, Jan Ernst, Yun Fu. *CVPR 2018*

- Revisiting Dilated Convolution: A Simple Approach for Weakly- and Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1805.04574) 
  - Yunchao Wei, Huaxin Xiao, Honghui Shi, Zequn Jie, Jiashi Feng, Thomas S. Huang. *CVPR 2018*

- Weakly- and Semi-Supervised Panoptic Segmentation.
  [[pdf]](https://arxiv.org/abs/1808.03575)
  [[code]](https://github.com/qizhuli/Weakly-Supervised-Panoptic-Segmentation)
  - Qizhu Li, Anurag Arnab, Philip H.S. Torr. *ECCV 2018*

- Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf)
  [[code]](https://github.com/speedinghzl/DSRG)
  - Zilong Huang, Xinggang Wang, Jiasi Wang, Wenyu Liu, Jingdong Wang.. *ECCV 2018*

- Transferable Semi-Supervised Semantic Segmentation.
  [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16348)
  - Huaxin Xiao, Yunchao Wei, Yu Liu, Maojun Zhang, Jiashi Feng. *AAAI 2018*

#### 2017

- Semi Supervised Semantic Segmentation Using Generative Adversarial Network.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Souly__Semi_Supervised_ICCV_2017_paper.pdf)
  - Nasim Souly, Concetto Spampinato, Mubarak Shah. *ICCV 2017*

- Simple Does It: Weakly Supervised Instance and Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1603.07485)
  [[code]](https://github.com/johnnylu305/Simple-does-it-weakly-supervised-instance-and-semantic-segmentation)
  - Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele. *CVPR 2017*

- Learning random-walk label propagation for weakly-supervised semantic segmentation.
  [[pdf]](https://arxiv.org/abs/1802.00470)
  - Paul Vernaza, Manmohan Chandraker. *CVPR 2017*

#### 2015

- Semi-Supervised Normalized Cuts for Image Segmentation.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2015/papers/Chew_Semi-Supervised_Normalized_Cuts_ICCV_2015_paper.pdf) 
  - Selene E. Chew, Nathan D. Cahill. *ICCV 2015*

- Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation.
  [[pdf]](https://arxiv.org/abs/1502.02734)
  [[code]](https://github.com/TheLegendAli/DeepLab-Context)
  - George Papandreou, Liang-Chieh Chen, Kevin Murphy, Alan L. Yuille. *ICCV 2015*

- Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1506.04924)
  - Seunghoon Hong, Hyeonwoo Noh, Bohyung Han. *NeurIPS 2015*

- BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1503.01640)
  - Jifeng Dai, Kaiming He, Jian Sun. *CVPR 2015*

- SSHMT: Semi-supervised Hierarchical Merge Tree for Electron Microscopy Image Segmentation.
  [[pdf]](https://arxiv.org/abs/1608.04051) 
  - Ting Liu, Miaomiao Zhang, Mehran Javanmardi, Nisha Ramesh, Tolga Tasdizen. *ECCV 2015*

#### 2013

- Semi-supervised Learning for Large Scale Image Cosegmentation.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2013/papers/Wang_Semi-supervised_Learning_for_2013_ICCV_paper.pdf) 
  - Zhengxiang Wang, Rujie Liu. *ICCV 2013*

- Semi-supervised Learning for Large Scale Image Cosegmentation.
  [[pdf]](https://www.ijcai.org/Proceedings/13/Papers/279.pdf) 
  - Ke Zhang, Wei Zhang, Yingbin Zheng, Xiangyang Xue. *AAAI 2013*













### Object Detection

#### 2022

- Dense Learning Based Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Dense_Learning_Based_Semi-Supervised_Object_Detection_CVPR_2022_paper.pdf)
  - Binghui Chen, Pengyu Li, Xiang Chen, Biao Wang, Lei Zhang, Xian-Sheng Hua. *CVPR 2022*

- Label Matching Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Label_Matching_Semi-Supervised_Object_Detection_CVPR_2022_paper.pdf)
  - Binbin Chen, Weijie Chen, Shicai Yang, Yunyi Xuan, Jie Song, Di Xie, Shiliang Pu, Mingli Song, Yueting Zhuang. *CVPR 2022*

- Semi-Supervised Object Detection via Multi-Instance Alignment With Global Class Prototypes.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Semi-Supervised_Object_Detection_via_Multi-Instance_Alignment_With_Global_Class_Prototypes_CVPR_2022_paper.pdf)
  - Aoxue Li, Peng Yuan, Zhenguo Li. *CVPR 2022*

- Active Teacher for Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Mi_Active_Teacher_for_Semi-Supervised_Object_Detection_CVPR_2022_paper.pdf)
  - Peng Mi, Jianghang Lin, Yiyi Zhou, Yunhang Shen, Gen Luo, Xiaoshuai Sun, Liujuan Cao, Rongrong Fu, Qiang Xu, Rongrong Ji. *CVPR 2022*

- Scale-Equivalent Distillation for Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection_CVPR_2022_paper.pdf)
  - Qiushan Guo, Yao Mu, Jianyu Chen, Tianqi Wang, Yizhou Yu, Ping Luo. *CVPR 2022*

- Unbiased Teacher v2: Semi-Supervised Object Detection for Anchor-Free and Anchor-Based Detectors.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.pdf)
  - Yen-Cheng Liu, Chih-Yao Ma, Zsolt Kira. *CVPR2022*

- MUM: Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_MUM_Mix_Image_Tiles_and_UnMix_Feature_Tiles_for_Semi-Supervised_CVPR_2022_paper.pdf)
  - JongMok Kim, JooYoung Jang, Seunghyeon Seo, Jisoo Jeong, Jongkeun Na, Nojun Kwak. *CVPR 2022*

- Group R-CNN for Weakly Semi-supervised Object Detection with Points.
  [[pdf]](https://arxiv.org/abs/2205.05920)
  - Shilong Zhang, Zhuoran Yu, Liyang Liu, Xinjiang Wang, Aojun Zhou, Kai Chen. *CVPR 2022*

- Group R-CNN for Weakly Semi-supervised Object Detection with Points.
  [[pdf]](https://arxiv.org/abs/2205.05920)
  - Shilong Zhang, Zhuoran Yu, Liyang Liu, Xinjiang Wang, Aojun Zhou, Kai Chen. *CVPR 2022*


#### 2021

- Combating Noise: Semi-supervised Learning by Region Uncertainty Quantification.
  [[pdf]](https://arxiv.org/abs/2111.00928)
  - Zhenyu Wang, Yali Li, Ye Guo, Shengjin Wang. *NeurIPS 2021*

- End-to-End Semi-Supervised Object Detection with Soft Teacher.
  [[pdf]](https://arxiv.org/abs/2106.09018)
  - Mengde Xu, Zheng Zhang, Han Hu, Jianfeng Wang, Lijuan Wang, Fangyun Wei, Xiang Bai, Zicheng Liu. *ICCV 2021*

- Humble Teachers Teach Better Students for Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Humble_Teachers_Teach_Better_Students_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.pdf)
  - Yihe Tang, Weifeng Chen, Yijun Luo, Yuting Zhang. *CVPR 2021*

- 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection.
  [[pdf]](https://arxiv.org/abs/2012.04355)
  - He Wang, Yezhen Cong, Or Litany, Yue Gao, Leonidas J. Guibas. *CVPR 2021*

- Interpolation-Based Semi-Supervised Learning for Object Detection.
  [[pdf]](https://arxiv.org/abs/2006.02158)
  - Jisoo Jeong, Vikas Verma, Minsung Hyun, Juho Kannala, Nojun Kwak. *CVPR 2021*

- Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Instant-Teaching_An_End-to-End_Semi-Supervised_Object_Detection_Framework_CVPR_2021_paper.pdf)
  - Qiang Zhou, Chaohui Yu, Zhibin Wang, Qi Qian, Hao Li . *CVPR 2021*

- Interactive Self-Training With Mean Teachers for Semi-Supervised Object Detection.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Interactive_Self-Training_With_Mean_Teachers_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.pdf)
  - Qize Yang, Xihan Wei, Biao Wang, Xian-Sheng Hua, Lei Zhang. *CVPR 2021*

- Data-Uncertainty Guided Multi-Phase Learning for Semi-Supervised Object Detection.
  [[pdf]](https://arxiv.org/abs/2103.16368)
  - Zhenyu Wang, Yali Li, Ye Guo, Lu Fang, Shengjin Wang. *CVPR 2021*

- Points as Queries: Weakly Semi-supervised Object Detection by Points.
  [[pdf]](https://arxiv.org/abs/2104.07434)
  - Liangyu Chen, Tong Yang, Xiangyu Zhang, Wei Zhang, Jian Sun. *CVPR 2021*

- Rethinking Pseudo Labels for Semi-Supervised Object Detection.
  [[pdf]](https://arxiv.org/abs/2106.00168)
  - Hengduo Li, Zuxuan Wu, Abhinav Shrivastava, Larry S. Davis. *Preprint 2021*

- Unbiased Teacher for Semi-Supervised Object Detection.
  [[pdf]](https://openreview.net/forum?id=MJIve1zgR_)
  [[code]](https://github.com/facebookresearch/unbiased-teacher)
  - Yen-Cheng Liu, Chih-Yao Ma, Zijian He, Chia-Wen Kuo, Kan Chen, Peizhao Zhang, Bichen Wu, Zsolt Kira, Peter Vajda. *ICLR 2021*

#### 2020

- SESS: Self-Ensembling Semi-Supervised 3D Object Detection.
  [[pdf]](https://arxiv.org/abs/1912.11803)
  [[code]](https://github.com/Na-Z/sess)
  - Na Zhao, Tat-Seng Chua, Gim Hee Lee. *CVPR 2020*

- A Simple Semi-Supervised Learning Framework for Object Detection.
  [[pdf]](https://arxiv.org/abs/2005.04757)
  [[code]](https://github.com/google-research/ssl_detection/)
  - Kihyuk Sohn, Zizhao Zhang, Chun-Liang Li, Han Zhang, Chen-Yu Lee, Tomas Pfister. *Preprint 2020*
  
- Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection.
  [[pdf]](https://arxiv.org/abs/2004.04725)
  [[code]](https://github.com/NVlabs/wetectron) 
  - Zhongzheng Ren, Zhiding Yu, Xiaodong Yang, Ming-Yu Liu, Yong Jae Lee, Alexander G. Schwing, Jan Kautz. *CVPR 2020*

#### 2019

- Consistency-based Semi-supervised Learning for Object Detection.
  [[pdf]](https://papers.NeurIPS.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection.pdf)
  [[code]](https://github.com/soo89/CSD-SSD)
  - Jisoo Jeong, Seungeui Lee, Jeesoo Kim, Nojun Kwak. *ICCV 2019*

- NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_NOTE-RCNN_NOise_Tolerant_Ensemble_RCNN_for_Semi-Supervised_Object_Detection_ICCV_2019_paper.pdf) 
  - Jiyang Gao, Jiang Wang, Shengyang Dai, Li-Jia Li, Ram Nevatia. *ICCV 2019*

- Semi-Supervised Video Salient Object Detection Using Pseudo-Labels.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Semi-Supervised_Video_Salient_Object_Detection_Using_Pseudo-Labels_ICCV_2019_paper.pdf) 
  - Pengxiang Yan, Guanbin Li, Yuan Xie, Zhen Li, Chuan Wang, Tianshui Chen, Liang Lin. *ICCV 2019*

- Transferable Semi-Supervised 3D Object Detection From RGB-D Data.
  [[pdf]](https://arxiv.org/abs/1904.10300) 
  - Yew Siang Tang, Gim Hee Lee. *ICCV 2019*

- Box-driven Class-wise Region Masking and Filling Rate Guided Loss for Weakly Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1904.11693) 
  - Chunfeng Song, Yan Huang, Wanli Ouyang, Liang Wang. *CVPR 2019*

#### 2018

- Adversarial Complementary Learning for Weakly Supervised Object Localization.
  [[pdf]](https://arxiv.org/abs/1804.06962) 
  - Xiaolin Zhang, Yunchao Wei, Jiashi Feng, Yi Yang, Thomas Huang. *CVPR 2018*

#### 2017

- ExtremeWeather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events.
  [[pdf]](https://arxiv.org/abs/1612.02095) 
  - Evan Racah, Christopher Beckham, Tegan Maharaj, Samira Ebrahimi Kahou, Prabhat, Christopher Pal. *NeurIPS 2017*

#### 2016

- Large Scale Semi-Supervised Object Detection Using Visual and Semantic Knowledge Transfer.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Tang_Large_Scale_Semi-Supervised_CVPR_2016_paper.pdf) 
  - Yuxing Tang, Josiah Wang, Boyang Gao, Emmanuel Dellandrea, Robert Gaizauskas, Liming Chen. *CVPR 2016*

#### 2015

- Watch and Learn: Semi-Supervised Learning for Object Detectors From Video.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Misra_Watch_and_Learn_2015_CVPR_paper.pdf) 
  - Ishan Misra, Abhinav Shrivastava, Martial Hebert. *CVPR 2015*

#### 2013

- Semi-supervised Learning of Feature Hierarchies for Object Detection in a Video.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2013/papers/Yang_Semi-supervised_Learning_of_2013_CVPR_paper.pdf)
  - Yang Yang, Guang Shu, Mubarak Shah. *CVPR 2013*















### Other tasks

#### 2022

- PLATINUM: Semi-Supervised Model Agnostic Meta-Learning using Submodular Mutual Information.
  [[pdf]](https://arxiv.org/abs/2201.12928)
  - Changbin Li, Suraj Kothawade, Feng Chen, Rishabh Iyer. *ICML 2022*

- HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning.
  [[pdf]](https://arxiv.org/abs/2201.04182)
  - Andrey Zhmoginov, Mark Sandler, Max Vladymyrov. *ICML 2022*

- End-to-End Semi-Supervised Learning for Video Action Detection.
  [[pdf]](https://arxiv.org/abs/2203.04251)
  [[code]](https://github.com/AKASH2907/End-to-End-Semi-Supervised-Learning-for-Video-Action-Detection)
  - Akash Kumar, Yogesh Singh Rawat. *CVPR 2022*

- Learning From Temporal Gradient for Semi-Supervised Action Recognition.
  [[pdf]](https://arxiv.org/abs/2111.13241)
  - Junfei Xiao, Longlong Jing, Lin Zhang, Ju He, Qi She, Zongwei Zhou, Alan Yuille, Yingwei Li. *CVPR 2022*

- Semi-Weakly-Supervised Learning of Complex Actions From Instructional Task Videos.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_Semi-Weakly-Supervised_Learning_of_Complex_Actions_From_Instructional_Task_Videos_CVPR_2022_paper.html)
  - Yuhan Shen, Ehsan Elhamifar. *CVPR 2022*

- BoostMIS: Boosting Medical Image Semi-supervised Learning with Adaptive Pseudo Labeling and Informative Active Annotation.
  [[pdf]](https://arxiv.org/abs/2203.02533)
  - Wenqiao Zhang, Lei Zhu, James Hallinan, Andrew Makmur, Shengyu Zhang, Qingpeng Cai, Beng Chin Ooi. *CVPR 2022*

- Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer.
  [[pdf]](https://arxiv.org/abs/2109.08024)
  - Fushun Zhu, Shan Zhao, Peng Wang, Hao Wang, Hua Yan, Shuaicheng Liu. *CVPR 2022*

- FisherMatch: Semi-Supervised Rotation Regression via Entropy-based Filtering.
  [[pdf]](https://arxiv.org/abs/2203.15765)
  - Yingda Yin, Yingcheng Cai, He Wang, Baoquan Chen. *CVPR 2022*

- Semi-Supervised Few-Shot Learning via Multi-Factor Clustering.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Ling_Semi-Supervised_Few-Shot_Learning_via_Multi-Factor_Clustering_CVPR_2022_paper.html)
  - Jie Ling, Lei Liao, Meng Yang, Jia Shuai. *CVPR 2022*

- Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin.
  [[pdf]](https://arxiv.org/abs/2203.12341)
  [[code]](https://github.com/hangyu94/Ada-CM)
  - Hangyu Li, Nannan Wang, Xi Yang, Xiaoyu Wang, Xinbo Gao. *CVPR 2022*

- Semi-Supervised Video Paragraph Grounding With Contrastive Encoder.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Semi-Supervised_Video_Paragraph_Grounding_With_Contrastive_Encoder_CVPR_2022_paper.html)
  - Xun Jiang, Xing Xu, Jingran Zhang, Fumin Shen, Zuo Cao, Heng Tao Shen. *CVPR 2022*

- End-to-End Semi-Supervised Learning for Video Action Detection.
  [[pdf]](https://arxiv.org/abs/2203.04251)
  - Akash Kumar, Yogesh Singh Rawat. *CVPR 2022*

- Cross-Model Pseudo-Labeling for Semi-Supervised Action Recognition.
  [[pdf]](https://arxiv.org/abs/2112.09690)
  - Yinghao Xu, Fangyun Wei, Xiao Sun, Ceyuan Yang, Yujun Shen, Bo Dai, Bolei Zhou, Stephen Lin. *CVPR 2022*

- Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Sommer_Gradient-SDF_A_Semi-Implicit_Surface_Representation_for_3D_Reconstruction_CVPR_2022_paper.html)
  - Christiane Sommer, Lu Sang, David Schubert, Daniel Cremers. *CVPR 2022*

- Semi-Supervised Learning of Semantic Correspondence with Pseudo-Labels.
  [[pdf]](https://arxiv.org/abs/2203.16038)
  - Jiwon Kim, Kwangrok Ryoo, Junyoung Seo, Gyuseong Lee, Daehwan Kim, Hansang Cho, Seungryong Kim. *CVPR 2022*

#### 2021

- CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/2107.00085)
  - Ankit Singh. *NeurIPS 2021*

- RETRIEVE: Coreset Selection for Efficient and Robust Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/2106.07760)
  [[code]](https://github.com/decile-team/cords)
  - Krishnateja Killamsetty, Xujiang Zhao, Feng Chen, Rishabh Iyer. *NeurIPS 2021*

- Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose.
  [[pdf]](https://arxiv.org/abs/2110.14213)
  [[code]](https://github.com/Angtian/NeuralVS)
  - Angtian Wang, Shenxiao Mei, Alan Yuille, Adam Kortylewski. *NeurIPS 2021*

- Trash To Treasure: Harvesting OOD Data With Cross-Modal Matching for Open-Set Semi-Supervised Learning.
  [[pdf]](http://arxiv.org/abs/2108.05617)
  - Junkai Huang, Chaowei Fang, Weikai Chen, Zhenhua Chai, Xiaolin Wei, Pengxu Wei, Liang Lin, Guanbin Li. *ICCV 2021*

- Iterative Label Cleaning for Transductive and Semi-Supervised Few-Shot Learning.
  [[pdf]](http://arxiv.org/abs/2012.07962)
  - Michalis Lazarou, Tania Stathaki, Yannis Avrithis. *ICCV 2021*

- Deep Co-Training With Task Decomposition for Semi-Supervised Domain Adaptation.
  [[pdf]](http://arxiv.org/abs/2007.12684)
  - Luyu Yang, Yan Wang, Mingfei Gao, Abhinav Shrivastava, Kilian Q. Weinberger, Wei-Lun Chao, Ser-Nam Lim. *ICCV 2021*

- Semi-Supervised Active Learning With Temporal Output Discrepancy.
  [[pdf]](http://arxiv.org/abs/2107.14153)
  - Siyu Huang, Tianyang Wang, Haoyi Xiong, Jun Huan, Dejing Dou. *ICCV 2021*

- Multiview Pseudo-Labeling for Semi-Supervised Learning From Video.
  [[pdf]](http://arxiv.org/abs/2104.00682)
  - Bo Xiong, Haoqi Fan, Kristen Grauman, Christoph Feichtenhofer. *ICCV 2021*

- ECACL: A Holistic Framework for Semi-Supervised Domain Adaptation.
  [[pdf]](http://arxiv.org/abs/2104.09136)
  - Kai Li, Chang Liu, Handong Zhao, Yulun Zhang, Yun Fu. *ICCV 2021*

- Pseudo-Loss Confidence Metric for Semi-Supervised Few-Shot Learning.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Pseudo-Loss_Confidence_Metric_for_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.html)
  - Kai Huang, Jie Geng, Wen Jiang, Xinyang Deng, Zhe Xu. *ICCV 2021*

- Just a Few Points Are All You Need for Multi-View Stereo: A Novel Semi-Supervised Learning Method for Multi-View Stereo.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Just_a_Few_Points_Are_All_You_Need_for_Multi-View_ICCV_2021_paper.pdf)
  - Taekyung Kim, Jaehoon Choi, Seokeon Choi, Dongki Jung, Changick Kim. *ICCV 2021*

- SemiHand: Semi-Supervised Hand Pose Estimation With Consistency.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_SemiHand_Semi-Supervised_Hand_Pose_Estimation_With_Consistency_ICCV_2021_paper.html)
  - Linlin Yang, Shicheng Chen, Angela Yao. *ICCV 2021*

- Spatial Uncertainty-Aware Semi-Supervised Crowd Counting.
  [[pdf]](http://arxiv.org/abs/2107.13271)
  - Yanda Meng, Hongrun Zhang, Yitian Zhao, Xiaoyun Yang, Xuesheng Qian, Xiaowei Huang, Yalin Zheng. *ICCV 2021*

- An Empirical Study of the Collapsing Problem in Semi-Supervised 2D Human Pose Estimation.
  [[pdf]](https://arxiv.org/abs/2011.12498)
  - Rongchang Xie, Chunyu Wang, Wenjun Zeng, Yizhou Wang. *ICCV 2021*

- Semi-Supervised Visual Representation Learning for Fashion Compatibility.
  [[pdf]](https://arxiv.org/pdf/2109.08052.pdf)
  - Ambareesh Revanur, Vijay Kumar, Deepthi Sharma  *ACM RecSys 2021*

- Semi-Supervised Action Recognition with Temporal Contrastive Learning.
  [[pdf]](https://arxiv.org/abs/2102.02751)
  [[code]](https://cvir.github.io/TCL/)
  - Ankit Singh, Omprakash Chakraborty, Ashutosh Varshney, Rameswar Panda, Rogerio Feris, Kate Saenko, Abir Das. *CVPR 2021*

- SSLayout360: Semi-Supervised Indoor Layout Estimation From 360deg Panorama.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tran_SSLayout360_Semi-Supervised_Indoor_Layout_Estimation_From_360deg_Panorama_CVPR_2021_paper.pdf)
  - Phi Vu Tran. *CVPR 2021*

- Semi-Supervised Video Deraining With Dynamical Rain Generator.
  [[pdf]](https://arxiv.org/abs/2103.07939)
  - Zongsheng Yue, Jianwen Xie, Qian Zhao, Deyu Meng. *CVPR 2021*

- Memory Oriented Transfer Learning for Semi-Supervised Image Deraining.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Memory_Oriented_Transfer_Learning_for_Semi-Supervised_Image_Deraining_CVPR_2021_paper.pdf)
  - Huaibo Huang, Aijing Yu, Ran He. *CVPR 2021*

- ORDisCo: Effective and Efficient Usage of Incremental Unlabeled Data for Semi-Supervised Continual Learning.
  [[pdf]](https://arxiv.org/abs/2101.00407)
  - Liyuan Wang, Kuo Yang, Chongxuan Li, Lanqing Hong, Zhenguo Li, Jun Zhu. *CVPR 2021*

- Learning Invariant Representations and Risks for Semi-supervised Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/2010.04647)
  - Bo Li, Yezhen Wang, Shanghang Zhang, Dongsheng Li, Trevor Darrell, Kurt Keutzer, Han Zhao. *CVPR 2021*

- Leveraging Large-Scale Weakly Labeled Data for Semi-Supervised Mass Detection in Mammograms.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Leveraging_Large-Scale_Weakly_Labeled_Data_for_Semi-Supervised_Mass_Detection_in_CVPR_2021_paper.pdf)
  - Yuxing Tang, Zhenjie Cao, Yanbo Zhang, Zhicheng Yang, Zongcheng Ji, Yiwei Wang, Mei Han, Jie Ma, Jing Xiao, Peng Chang. *CVPR 2021*

- MonoRec: Semi-Supervised Dense Reconstruction in Dynamic Environments From a Single Moving Camera.
  [[pdf]](https://arxiv.org/abs/2011.11814)
  [[code]](https://vision.in.tum.de/research/monorec)
  - Yuxing Tang, Zhenjie Cao, Yanbo Zhang, Zhicheng Yang, Zongcheng Ji, Yiwei Wang, Mei Han, Jie Ma, Jing Xiao, Peng Chang. *CVPR 2021*

- Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/2104.09415)
  - Jichang Li, Guanbin Li, Yemin Shi, Yizhou Yu. *CVPR 2021*

- More Photos Are All You Need: Semi-Supervised Learning for Fine-Grained Sketch Based Image Retrieval.
  [[pdf]](https://arxiv.org/abs/2103.13990)
  - Ayan Kumar Bhunia, Pinaki Nath Chowdhury, Aneeshan Sain, Yongxin Yang, Tao Xiang, Yi-Zhe Song. *CVPR 2021*

- Semi-Supervised 3D Hand-Object Poses Estimation With Interactions in Time.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Semi-Supervised_3D_Hand-Object_Poses_Estimation_With_Interactions_in_Time_CVPR_2021_paper.pdf)
  - Shaowei Liu, Hanwen Jiang, Jiarui Xu, Sifei Liu, Xiaolong Wang. *CVPR 2021*

- OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning with Outliers.
  [[pdf]](https://arxiv.org/abs/2105.14148)
  - Kuniaki Saito, Donghyun Kim, Kate Saenko. *Preprint 2021*

- More Photos are All You Need: Semi-Supervised Learning for Fine-Grained Sketch Based Image Retrieval.
  [[code]](https://github.com/AyanKumarBhunia/semisupervised-FGSBIR)
  - Ayan Kumar Bhunia, Pinaki nath Chowdhury, Aneeshan Sain, Yongxin Yang, Tao Xiang, Yi-Zhe Song. *CVPR 2021*

- Semi-supervised Keypoint Localization.
  [[pdf]](https://arxiv.org/abs/2101.07988)
  - Olga Moskvyak, Frederic Maire, Feras Dayoub, Mahsa Baktashmotlagh. *ICLR 2021*

- Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning.
  [[pdf]](https://arxiv.org/abs/2006.12097)
  - Wonyong Jeong, Jaehong Yoon, Eunho Yang, Sung Ju Hwang. *ICLR 2021*

#### 2020

- Consistency-based Semi-supervised Active Learning: Towards Minimizing Labeling Cost.
  [[pdf]](https://arxiv.org/abs/1910.07153)
  - Mingfei Gao, Zizhao Zhang, Guo Yu, Sercan O. Arik, Larry S. Davis, Tomas Pfister. *ECCV 2020*

- Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/2007.11330)
  - Qing Yu, Daiki Ikami, Go Irie, Kiyoharu Aizawa. *ECCV 2020*

- Semi-supervised Crowd Counting via Self-training on Surrogate Tasks.
  [[pdf]](https://arxiv.org/abs/2007.03207)
  - Yan Liu, Lingqiao Liu, Peng Wang, Pingping Zhang, Yinjie Lei. *ECCV 2020*

- Online Meta-Learning for Multi-Source and Semi-Supervised Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/2004.04398)
  - Da Li, Timothy Hospedales. *ECCV 2020*

- Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/2007.09375)
  - Taekyung Kim, Changick Kim. *ECCV 2020*

- Semi-supervised Learning with a Teacher-student Network for Generalized Attribute Prediction.
  [[pdf]](https://arxiv.org/abs/2007.06769)
  - Minchul Shin. *ECCV 2020*

- Adversarial Self-Supervised Learning for Semi-Supervised 3D Action Recognition.
  [[pdf]](https://arxiv.org/abs/2007.05934)
  - Chenyang Si, Xuecheng Nie, Wei Wang, Liang Wang, Tieniu Tan, Jiashi Feng. *ECCV 2020*

- Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/2007.07695)
  [[code]](https://github.com/YBZh/Label-Propagation-with-Augmented-Anchors)
  - Yabin Zhang, Bin Deng, Kui Jia, Lei Zhang. *ECCV 2020*

- DeepDeform: Learning Non-rigid RGB-D Reconstruction with Semi-supervised Data.
  [[pdf]](https://arxiv.org/abs/1912.04302)
  [[code]](https://github.com/AljazBozic/DeepDeform)
  - Aljaž Božič, Michael Zollhöfer, Christian Theobalt, Matthias Nießner. *CVPR 2020*

- Gait Recognition via Semi-supervised Disentangled Representation Learning to Identity and Covariate Features.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Gait_Recognition_via_Semi-supervised_Disentangled_Representation_Learning_to_Identity_and_CVPR_2020_paper.pdf)
  - Xiang Li, Yasushi Makihara, Chi Xu, Yasushi Yagi, Mingwu Ren. *CVPR 2020*

- Optical Flow in Dense Foggy Scenes using Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/2004.01905)
  - Wending Yan, Aashish Sharma, Robby T. Tan. *CVPR 2020*

- TransMatch: A Transfer-Learning Scheme for Semi-Supervised Few-Shot Learning.
  [[pdf]](https://arxiv.org/abs/1912.09033)
  - Zhongjie Yu, Lin Chen, Zhongwei Cheng, Jiebo Luo. *CVPR 2020*

- Data-Efficient Semi-Supervised Learning by Reliable Edge Mining.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Data-Efficient_Semi-Supervised_Learning_by_Reliable_Edge_Mining_CVPR_2020_paper.pdf)
  - Peibin Chen, Tao Ma, Xu Qin, Weidi Xu, Shuchang Zhou. *CVPR 2020*

- Multiview-Consistent Semi-Supervised Learning for 3D Human Pose Estimation.
  [[pdf]](https://arxiv.org/abs/1908.05293)
  - Rahul Mitra, Nitesh B. Gundavarapu, Abhishek Sharma, Arjun Jain. *CVPR 2020*

- A Multi-Task Mean Teacher for Semi-Supervised Shadow Detection.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Multi-Task_Mean_Teacher_for_Semi-Supervised_Shadow_Detection_CVPR_2020_paper.pdf)
  - Zhihao Chen, Lei Zhu, Liang Wan, Song Wang, Wei Feng, Pheng-Ann Heng. *CVPR 2020*

- Semi-Supervised Learning for Few-Shot Image-to-Image Translation.
  [[pdf]](https://arxiv.org/abs/2003.13853)
  [[code]](https://github.com/yaxingwang/SEMIT)
  - Yaxing Wang, Salman Khan, Abel Gonzalez-Garcia, Joost van de Weijer, Fahad Shahbaz Khan. *CVPR 2020*

- 3D Sketch-Aware Semantic Scene Completion via Semi-Supervised Structure Prior.
  [[pdf]](https://arxiv.org/abs/2003.14052)
  - Xiaokang Chen, Kwan-Yee Lin, Chen Qian, Gang Zeng, Hongsheng Li. *CVPR 2020*

- A Semi-Supervised Assessor of Neural Architectures.
  [[pdf]](https://arxiv.org/abs/2005.06821)
  - Yehui Tang, Yunhe Wang, Yixing Xu, Hanting Chen, Chunjing Xu, Boxin Shi, Chao Xu, Qi Tian, Chang Xu. *CVPR 2020*

- FocalMix: Semi-Supervised Learning for 3D Medical Image Detection.
  [[pdf]](https://arxiv.org/abs/2003.09108)
  - Dong Wang, Yuan Zhang, Kexin Zhang, Liwei Wang. *CVPR 2020*

- Learning to Detect Important People in Unlabelled Images for Semi-Supervised Important People Detection.
  [[pdf]](https://arxiv.org/abs/2004.07568)
  - Fa-Ting Hong, Wei-Hong Li, Wei-Shi Zheng. *CVPR 2020*

- Generalized Product Quantization Network for Semi-Supervised Image Retrieval.
  [[pdf]](https://arxiv.org/abs/2002.11281)
  - Young Kyun Jang, Nam Ik Cho. *CVPR 2020*

- From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.pdf)
  - Wenhan Yang, Shiqi Wang, Yuming Fang, Yue Wang, Jiaying Liu. *CVPR 2020*

- Deep Semi-Supervised Anomaly Detection.
  [[pdf]](https://arxiv.org/abs/1906.02694)
  [[code]](https://github.com/lukasruff/Deep-SAD-PyTorch)
  - Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Alexander Binder, Emmanuel Müller, Klaus-Robert Müller, Marius Kloft. *ICLR 2020*

#### 2019

- Learning to Self-Train for Semi-Supervised Few-Shot Classification.
  [[pdf]](https://arxiv.org/abs/1906.00562)
  [[code]](https://github.com/xinzheli1217/learning-to-self-train)
  - Xinzhe Li, Qianru Sun, Yaoyao Liu, Shibao Zheng, Qin Zhou, Tat-Seng Chua, Bernt Schiele. *NeurIPS 2019*

- Multi-label Co-regularization for Semi-supervised Facial Action Unit Recognition.
  [[pdf]](https://arxiv.org/abs/1906.02694)
  [[code]](https://github.com/nxsEdson/MLCR)
  - Xuesong Niu, Hu Han, Shiguang Shan, Xilin Chen. *NeurIPS 2019*

- Semi-Supervised Monocular 3D Face Reconstruction With End-to-End Shape-Preserved Domain Transfer.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Semi-Supervised_Monocular_3D_Face_Reconstruction_With_End-to-End_Shape-Preserved_Domain_Transfer_ICCV_2019_paper.pdf) 
  - Jingtan Piao, Chen Qian, Hongsheng Li. *ICCV 2019*

- SO-HandNet: Self-Organizing Network for 3D Hand Pose Estimation With Semi-Supervised Learning.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_SO-HandNet_Self-Organizing_Network_for_3D_Hand_Pose_Estimation_With_Semi-Supervised_ICCV_2019_paper.pdf) 
  - Yujin Chen, Zhigang Tu, Liuhao Ge, Dejun Zhang, Ruizhi Chen, Junsong Yuan. *ICCV 2019*

- Semi-Supervised Pedestrian Instance Synthesis and Detection With Mutual Reinforcement.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Semi-Supervised_Pedestrian_Instance_Synthesis_and_Detection_With_Mutual_Reinforcement_ICCV_2019_paper.pdf) 
  - Si Wu, Sihao Lin, Wenhao Wu, Mohamed Azzam, Hau-San Wong. *ICCV 2019*

- Semi-Supervised Skin Detection by Network With Mutual Guidance.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Semi-Supervised_Skin_Detection_by_Network_With_Mutual_Guidance_ICCV_2019_paper.pdf) 
  - Yi He, Jiayuan Shi, Chuan Wang, Haibin Huang, Jiaming Liu, Guanbin Li, Risheng Liu, Jue Wang. *ICCV 2019*

- MONET: Multiview Semi-Supervised Keypoint Detection via Epipolar Divergence.
  [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yao_MONET_Multiview_Semi-Supervised_Keypoint_Detection_via_Epipolar_Divergence_ICCV_2019_paper.pdf) 
  - Yuan Yao, Yasamin Jafarian, Hyun Soo Park. *ICCV 2019*

- 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.pdf) 
  - Dario Pavllo, Christoph Feichtenhofer, David Grangier, Michael Auli. *CVPR 2019*

- Semi-Supervised Transfer Learning for Image Rain Removal.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Semi-Supervised_Transfer_Learning_for_Image_Rain_Removal_CVPR_2019_paper.pdf) 
  - Wei Wei, Deyu Meng, Qian Zhao, Zongben Xu, Ying Wu. *CVPR 2019*

- KE-GAN: Knowledge Embedded Generative Adversarial Networks for Semi-Supervised Scene Parsing.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qi_KE-GAN_Knowledge_Embedded_Generative_Adversarial_Networks_for_Semi-Supervised_Scene_Parsing_CVPR_2019_paper.pdf) 
  - Mengshi Qi, Yunhong Wang, Jie Qin, Annan Li. *CVPR 2019*

#### 2018

- Semi-Supervised Generative Adversarial Hashing for Image Retrieval.
  [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guanan_Wang_Semi-Supervised_Generative_Adversarial_ECCV_2018_paper.pdf)
  - Guan'an Wang, Qinghao Hu, Jian Cheng, Zengguang Hou. *ECCV 2018*

- Improving Landmark Localization With Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1709.01591) 
  - Sina Honari, Pavlo Molchanov, Stephen Tyree, Pascal Vincent, Christopher Pal, Jan Kautz. *CVPR 2018*

- Semi-Supervised Bayesian Attribute Learning for Person Re-Identification.
  [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17325) 
  - Wenhe Liu, Xiaojun Chang, Ling Chen, Yi Yang. *AAAI 2018*

#### 2017

- Semi-Supervised Deep Learning for Monocular Depth Map Prediction.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kuznietsov_Semi-Supervised_Deep_Learning_CVPR_2017_paper.pdf) 
  - Yevhen Kuznietsov, Jorg Stuckler, Bastian Leibe. *CVPR 2017*

#### 2016

- SemiContour: A Semi-Supervised Learning Approach for Contour Detection.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_SemiContour_A_Semi-Supervised_CVPR_2016_paper.pdf) 
  - Zizhao Zhang, Fuyong Xing, Xiaoshuang Shi, Lin Yang. *CVPR 2016*

- Semi-Supervised Vocabulary-Informed Learning
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Fu_Semi-Supervised_Vocabulary-Informed_Learning_CVPR_2016_paper.pdf) 
  - Yanwei Fu, Leonid Sigal. *CVPR 2016*

#### 2015

- Adaptively Unified Semi-Supervised Dictionary Learning With Active Points.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2015/papers/Wang_Adaptively_Unified_Semi-Supervised_ICCV_2015_paper.pdf) 
  - Xiaobo Wang, Xiaojie Guo, Stan Z. Li. *ICCV 2015*

- Semi-Supervised Zero-Shot Classification With Label Representation Learning.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2015/papers/Li_Semi-Supervised_Zero-Shot_Classification_ICCV_2015_paper.pdf) 
  - Xin Li, Yuhong Guo, Dale Schuurmans. *ICCV 2015*

#### 2014

- Semi-Supervised Coupled Dictionary Learning for Person Re-identification.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Liu_Semi-Supervised_Coupled_Dictionary_2014_CVPR_paper.pdf) 
  - Xiao Liu, Mingli Song, Dacheng Tao, Xingchen Zhou, Chun Chen, Jiajun Bu. *CVPR 2014*

- A Convex Formulation for Semi-Supervised Multi-Label Feature Selection. 
  [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8594) 
  - Xiaojun Chang, Feiping Nie, Yi Yang, Heng Huang. *AAAI 2014*

#### 2013

- Heterogeneous Image Features Integration via Multi-modal Semi-supervised Learning Model. 
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2013/papers/Cai_Heterogeneous_Image_Features_2013_ICCV_paper.pdf) 
  - Xiao Cai, Feiping Nie, Weidong Cai, Heng Huang. *ICCV 2013*

- Semi-supervised Learning with Constraints for Person Identification in Multimedia Data.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2013/papers/Bauml_Semi-supervised_Learning_with_2013_CVPR_paper.pdf) 
  - Martin Bauml, Makarand Tapaswi, Rainer Stiefelhagen. *CVPR 2013*

#### 2010-2000

-  Multimodal semi-supervised learning for image classification. 
  [[pdf]](https://hal.inria.fr/inria-00548640/document) 
  - Matthieu Guillaumin, Jakob Verbeek, Cordelia Schmid. *CVPR 2010*

- Semi-supervised Discriminant Analysis.
  [[pdf]](http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/ICCV07_dengcai_SDA.pdf)
  - Deng Cai, Xiaofei He, Jiawei Han. *ICCV 2007*


## NLP

- Industry Scale Semi-Supervised Learning for Natural Language Understanding. 
  [[pdf]](https://arxiv.org/abs/2103.15871)
  - Luoxin Chen, Francisco Garcia, Varun Kumar, He Xie, Jianhua Lu. *NAACL 2021*

- Event Representation with Sequential, Semi-Supervised Discrete Variables. 
  [[pdf]](https://arxiv.org/abs/2010.04361)
  - Mehdi Rezaee, Francis Ferraro. *NAACL 2021*

- Framing Unpacked: A Semi-Supervised Interpretable Multi-View Model of Media Frames. 
  [[pdf]](https://arxiv.org/abs/2104.11030)
  - Shima Khanehzar, Trevor Cohn, Gosia Mikolajczak, Andrew Turpin, Lea Frermann. *NAACL 2021*

#### 2020

- To BERT or Not to BERT: Comparing Task-specific and Task-agnostic Semi-Supervised Approaches for Sequence Tagging. 
  [[pdf]](https://arxiv.org/abs/2010.14042)
  - Kasturi Bhattacharjee, Miguel Ballesteros, Rishita Anubhai, Smaranda Muresan, Jie Ma, Faisal Ladhak, Yaser Al-Onaizan. *EMNLP 2020*

- A Probabilistic End-To-End Task-Oriented Dialog Model with Latent Belief States towards Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/2009.08115)
  [[code]](https://github.com/thu-spmi/LABES) 
  - Yichi Zhang, Zhijian Ou, Huixin Wang, Junlan Feng. *EMNLP 2020*

- Semi-Supervised Bilingual Lexicon Induction with Two-way Interaction. 
  [[pdf]](https://arxiv.org/abs/2010.07101)
  [[code]](https://github.com/BestActionNow/SemiSupBLI) 
  - Xu Zhao, Zihao Wang, Hao Wu, Yong Zhang. *EMNLP 2020*

- Local Additivity Based Data Augmentation for Semi-supervised NER. 
  [[pdf]](https://arxiv.org/abs/2010.01677)
  [[code]](https://github.com/GT-SALT/LADA) 
  - Jiaao Chen, Zhenghui Wang, Ran Tian, Zichao Yang, Diyi Yang. *EMNLP 2020*

- Semi-supervised New Event Type Induction and Event Detection. 
  [[pdf]](https://www.aclweb.org/anthology/2020.emnlp-main.53/)
  [[code]](https://github.com/wilburOne/SSVQVAE) 
  - Lifu Huang and Heng Ji. *EMNLP 2020*

- MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification. 
  [[pdf]](https://arxiv.org/abs/2004.12239)
  [[code]](https://github.com/GT-SALT/MixText) 
  - Jiaao Chen, Zichao Yang, Diyi Yang. *ACL 2020*

- Semi-Supervised Dialogue Policy Learning via Stochastic Reward Estimation. 
  [[pdf]](https://arxiv.org/abs/2005.04379)
  - Xinting Huang, Jianzhong Qi, Yu Sun, Rui Zhang. *ACL 2020*

- SeqVAT: Virtual Adversarial Training for Semi-Supervised Sequence Labeling. 
  [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.777/)
  - Luoxin Chen, Weitong Ruan, Xinyue Liu, Jianhua Lu. *ACL 2020*

- Semi-Supervised Semantic Dependency Parsing Using CRF Autoencoders. 
  [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.607/)
  [[code]](https://github.com/JZXXX/Semi-SDP) 
  - Zixia Jia, Youmi Ma, Jiong Cai, Kewei Tu. *ACL 2020*

- Revisiting self-training for neural sequence generation. 
  [[pdf]](https://arxiv.org/abs/1909.13788)
  [[code]](https://github.com/jxhe/self-training-text-generation) 
  - Junxian He, Jiatao Gu, Jiajun Shen, Marc'Aurelio Ranzato. *ICLR 2020*

#### 2019

- Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training. 
  [[pdf]](https://arxiv.org/abs/1909.00415) 
  - Giannis Karamanolakis, Daniel Hsu, Luis Gravano. *EMNLP 2019*

- Semi-supervised Semantic Role Labeling Using the Latent Words Language Model. 
  [[pdf]](https://www.aclweb.org/anthology/D09-1003.pdf) 
  - Koen Deschacht, Marie-Francine Moens. *EMNLP 2019*

- Semi-Supervised Semantic Role Labeling with Cross-View Training. 
  [[pdf]](https://www.aclweb.org/anthology/D19-1094.pdf) 
  - Rui Cai, Mirella Lapata. *EMNLP 2019*

- Delta-training: Simple Semi-Supervised Text Classification using Pretrained Word Embeddings. 
  [[pdf]](https://www.aclweb.org/anthology/D19-1347.pdf) 
  - Hwiyeol Jo, Ceyda Cinarel. *EMNLP 2019*

- Semi-supervised Entity Alignment via Joint Knowledge Embedding Model and Cross-graph Model. 
  [[pdf]](https://www.aclweb.org/anthology/D19-1274.pdf) 
  - Chengjiang Li, Yixin Cao, Lei Hou, Jiaxin Shi, Juanzi Li, Tat-Seng Chua. *EMNLP 2019*

- A Cross-Sentence Latent Variable Model for Semi-Supervised Text Sequence Matching. 
  [[pdf]](https://www.aclweb.org/anthology/P19-1469.pdf) 
  - Jihun Choi, Taeuk Kim, Sang-goo Lee. *ACL 2019*

- A Semi-Supervised Stable Variational Network for Promoting Replier-Consistency in Dialogue Generation. 
  [[pdf]](https://www.aclweb.org/anthology/D19-1200.pdf) 
  - Jinxin Chang, Ruifang He, Longbiao Wang, Xiangyu Zhao, Ting Yang, Ruifang Wang. *ACL 2019*

- No Army, No Navy: BERT Semi-Supervised Learning of Arabic Dialects. 
  [[pdf]](https://www.aclweb.org/anthology/W19-4637.pdf) 
  - Chiyu Zhang, Muhammad Abdul-Mageedl. *ACL 2019*

- Paraphrase Generation for Semi-Supervised Learning in NLU. 
  [[pdf]](https://www.aclweb.org/anthology/W19-2306.pdf) 
  - Eunah Cho, He Xie, William M. Campbell. *NAACL 2019*

- Graph-Based Semi-Supervised Learning for Natural Language Understanding. 
  [[pdf]](https://www.aclweb.org/anthology/D19-5318.pdf) 
  - Zimeng Qiu, Eunah Cho, Xiaochun Ma, William Campbell. *EMNLP 2019*

-  Revisiting LSTM Networks for Semi-Supervised Text Classification via Mixed Objective Function. 
  [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/4672) 
  - Devendra Singh Sachan, Manzil Zaheer, Ruslan Salakhutdinov. *AAAI 2019*

#### 2018

- Strong Baselines for Neural Semi-supervised Learning under Domain Shift. 
  [[pdf]](https://arxiv.org/abs/1804.09530) 
  [[code]](https://github.com/ambujojha/SemiSupervisedLearning)
  - Sebastian Ruder, Barbara Plank. *ACL 2018*

- Simple and Effective Semi-Supervised Question Answering. 
  [[pdf]](https://www.aclweb.org/anthology/N18-2092.pdf) 
  - Bhuwan Dhingra, Danish Danish, Dheeraj Rajagopal. *NAACL 2018*

- Semi-Supervised Disfluency Detection. 
  [[pdf]](https://www.aclweb.org/anthology/C18-1299.pdf) 
  - Feng Wang, Wei Chen, Zhen Yang, Qianqian Dong, Shuang Xu, Bo Xu. *COLING 2018*

- Variational Sequential Labelers for Semi-Supervised Learning. 
  [[pdf]](https://www.aclweb.org/anthology/D18-1020.pdf) 
  - Mingda Chen, Qingming Tang, Karen Livescu, Kevin Gimpel. *EMNLP 2018*

- Towards Semi-Supervised Learning for Deep Semantic Role Labeling. 
  [[pdf]](https://www.aclweb.org/anthology/D18-1538.pdf) 
  - Sanket Vaibhav Mehta, Jay Yoon Lee, Jaime Carbonell. *EMNLP 2018*

- Adaptive Semi-supervised Learning for Cross-domain Sentiment Classification. 
  [[pdf]](https://www.aclweb.org/anthology/D18-1383.pdf) 
  - Ruidan He, Wee Sun Lee, Hwee Tou Ng, Daniel Dahlmeier. *EMNLP 2018*

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification. 
  [[pdf]](https://www.aclweb.org/anthology/D19-1488.pdf) 
  - Hu Linmei, Tianchi Yang, Chuan Shi, Houye Ji, Xiaoli Li. *EMNLP 2018*

- Semi-Supervised Learning for Neural Keyphrase Generation. 
  [[pdf]](https://www.aclweb.org/anthology/D18-1447.pdf) 
  - Hai Ye, Lu Wang. *EMNLP 2018*

- Semi-Supervised Sequence Modeling with Cross-View Training. 
  [[pdf]](https://www.aclweb.org/anthology/D18-1217.pdf) 
  - Kevin Clark, Minh-Thang Luong, Christopher D. Manning, Quoc Le. *ACL 2018*

- Semi-Supervised Learning with Declaratively Specified Entropy Constraints. 
  [[pdf]](https://arxiv.org/abs/1605.07725) 
  - Haitian Sun, William W. Cohen, Lidong Bing. *NeurIPS 2018*

- Semi-Supervised Prediction-Constrained Topic Models. 
  [[pdf]](http://proceedings.mlr.press/v84/hughes18a/hughes18a.pdf) 
  - Michael Hughes, Gabriel Hope, Leah Weiner, Thomas McCoy, Roy Perlis, Erik Sudderth, Finale Doshi-Velez. *AISTATS 2018*

- SEE: Towards Semi-Supervised End-to-End Scene Text Recognition. 
  [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16270) 
  -  Christian Bartz, Haojin Yang, Christoph Meinel. *AAAI 2018*

- Inferring Emotion from Conversational Voice Data: A Semi-Supervised Multi-Path Generative Neural Network Approach. 
  [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17236) 
  - Suping Zhou, Jia Jia, Qi Wang, Yufei Dong, Yufeng Yin, Kehua Leis. *AAAI 2018*

#### 2017

- Semi-supervised Multitask Learning for Sequence Labeling. 
  [[pdf]](https://www.aclweb.org/anthology/P17-1194.pdf) 
  [[code]](https://github.com/marekrei/sequence-labeler)
  - Marek Rei. *ACL 2017*

- Semi-supervised Structured Prediction with Neural CRF Autoencoder. 
  [[pdf]](https://www.aclweb.org/anthology/D17-1179.pdf) 
  - Xiao Zhang, Yong Jiang, Hao Peng, Kewei Tu, Dan Goldwasser. *EMNLP 2017*

- Semi-supervised sequence tagging with bidirectional language models. 
  [[pdf]](https://www.aclweb.org/anthology/P17-1161.pdf) 
  - Matthew Peters, Waleed Ammar, Chandra Bhagavatula, Russell Power. *ACL 2017*

- Variational Autoencoder for Semi-Supervised Text Classification. 
  [[pdf]](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14299) 
  -  Weidi Xu, Haoze Sun, Chao Deng, Ying Tan. *AAAI 2017*

- Semi-Supervised Multi-View Correlation Feature Learning with Application to Webpage Classification. 
  [[pdf]](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14582) 
  - Xiao-Yuan Jing, Fei Wu, Xiwei Dong, Shiguang Shan, Songcan Chen. *AAAI 2017*

- Adversarial Training Methods for Semi-Supervised Text Classification. 
  [[pdf]](https://arxiv.org/abs/1605.07725) 
  [[code]](https://github.com/tensorflow/models/tree/master/research/adversarial_text)
  - Chelsea Finn, Tianhe Yu, Justin Fu, Pieter Abbeel, Sergey Levine. *ICLR 2017*

#### 2016

- Dual Learning for Machine Translation. 
  [[pdf]](https://arxiv.org/abs/1611.00179) 
  - Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma. *NeurIPS 2016*

- Semi-supervised Clustering for Short Text via Deep Representation Learning. 
  [[pdf]](https://www.aclweb.org/anthology/K16-1004.pdf) 
  - Zhiguo Wang, Haitao Mi, Abraham Ittycheriah. *CoNLL 2016*

- Semi-supervised Question Retrieval with Gated Convolutions. 
  [[pdf]](https://www.aclweb.org/anthology/N16-1153.pdf) 
  - Tao Lei, Hrishikesh Joshi, Regina Barzilay, Tommi Jaakkola, Kateryna Tymoshenko, Alessandro Moschitti, Lluís Màrquez. *NAACL 2016*

- Semi-supervised Word Sense Disambiguation with Neural Models. 
  [[pdf]](https://www.aclweb.org/anthology/C16-1130.pdf) 
  - Dayu Yuan, Julian Richardson, Ryan Doherty, Colin Evans, Eric Altendorf. *COLING 2016*

- Semi-Supervised Learning for Neural Machine Translation. 
  [[pdf]](https://www.aclweb.org/anthology/P16-1185.pdf) 
  - Yong Cheng, Wei Xu, Zhongjun He, Wei He, Hua Wu, Maosong Sun, Yang Liu. *ACL 2016*

- A Semi-Supervised Learning Approach to Why-Question Answering. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12208) 
  -  Jong-Hoon Oh, Kentaro Torisawa, Chikara Hashimoto, Ryu Iida, Masahiro Tanaka, Julien Kloetzer. *AAAI 2016*

- Semi-Supervised Multinomial Naive Bayes for Text Classification by Leveraging Word-Level Statistical Constraint. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12019) 
  - Li Zhao, Minlie Huang, Ziyu Yao, Rongwei Su, Yingying Jiang, Xiaoyan Zhu. *AAAI 2016*

- Supervised and Semi-Supervised Text Categorization using LSTM for Region Embeddings. 
  [[pdf]](http://proceedings.mlr.press/v48/johnson16.pdf) 
  [[code]](https://github.com/tensorflow/models/tree/master/research/adversarial_text)
  - Rie Johnson, Tong Zhang. *ICML 2016*

#### 2015

- Semi-Supervised Word Sense Disambiguation Using Word Embeddings in General and Specific Domains. 
  [[pdf]](https://www.aclweb.org/anthology/N15-1035.pdf) 
  - Kaveh Taghipour, Hwee Tou Ng. *NACCL 2015*

- Semi-supervised Sequence Learning. 
  [[pdf]](https://arxiv.org/abs/1511.01432) 
  [[code]](https://github.com/tensorflow/models/tree/master/research/adversarial_text)
  - Andrew M. Dai, Quoc V. Le. *NeurIPS 2015*

- Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding. 
  [[pdf]](https://arxiv.org/abs/1504.01255) 
  - Rie Johnson, Tong Zhang. *NeurIPS 2015*

- Mining User Intents in Twitter: A Semi-Supervised Approach to Inferring Intent Categories for Tweets. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9794) 
  -  Jinpeng Wang, Gao Cong, Xin Wayne Zhao, Xiaoming Li. *AAAI 2015*

#### 2014

- Semi-Supervised Matrix Completion for Cross-Lingual Text Classification. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8298) 
  - Min Xiao, Yuhong Guo. *AAAI 2014*

#### 2013

- Effective Bilingual Constraints for Semi-Supervised Learning of Named Entity Recognizers. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/view/6346) 
  - Mengqiu Wang, Wanxiang Che, Christopher D. Manning. *AAAI 2013*

#### 2011

- Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions. 
  [[pdf]](https://www.aclweb.org/anthology/D11-1014.pdf) 
  - Richard Socher, Jeffrey Pennington, Eric H. Huang, Andrew Y. Ng, Christopher D. Manning. *EMNLP 2011*

#### 2010

- Cross Language Text Classification by Model Translation and Semi-Supervised Learning. 
  [[pdf]](https://www.aclweb.org/anthology/D10-1103.pdf) 
  - Lei Shi, Rada Mihalcea, Mingjun Tian. *EMNLP 2010*

- Simple Semi-Supervised Training of Part-Of-Speech Taggers. 
  [[pdf]](https://www.aclweb.org/anthology/P10-2038.pdf) 
  - Anders Søgaard. *ACL 2010*

- Word Representations: A Simple and General Method for Semi-Supervised Learning. 
  [[pdf]](https://www.aclweb.org/anthology/P10-1040.pdf) 
  - Joseph Turian, Lev-Arie Ratinov, Yoshua Bengio. *ACL 2010*

- A Semi-Supervised Method to Learn and Construct Taxonomies Using the Web. 
  [[pdf]](https://www.aclweb.org/anthology/D10-1108.pdf) 
  - Zornitsa Kozareva, Eduard Hovy. *EMNLP 2010*

#### 2009

- A Simple Semi-supervised Algorithm For Named Entity Recognition. 
  [[pdf]](https://www.aclweb.org/anthology/W09-2208.pdf) 
  - Wenhui Liao, Sriharsha Veeramachaneni. *NACCL 2009*

#### 2008

- SemiBoost: Boosting for Semi-Supervised Learning. 
  [[pdf]](https://www.cse.msu.edu/~rongjin/publications/pami_finalformat.pdf) 
  - Pavan Kumar Mallapragada, Rong Jin, Anil K. Jain, Yi Liu. *IEEE Transactions on Pattern Analysis and Machine Intelligence 2008*

- Simple Semi-supervised Dependency Parsing. 
  [[pdf]](https://www.aclweb.org/anthology/P08-1068.pdf) 
  - Terry Koo, Xavier Carreras, Michael Collins. *ACL 2008*

#### 2006

- Self-Training for Enhancement and Domain Adaptation of Statistical Parsers Trained on Small Datasets. 
  [[pdf]](https://www.aclweb.org/anthology/P07-1078.pdf) 
  - Roi Reichart, Ari Rappoport. *ACL 2007*

#### 2006

- Effective Self-Training for Parsing. 
  [[pdf]](https://www.aclweb.org/anthology/N06-1020.pdf) 
  - David McClosky, Eugene Charniak, Mark Johnson. *ACL 2006*

- Reranking and Self-Training for Parser Adaptation. 
  [[pdf]](https://www.aclweb.org/anthology/P06-1043.pdf) 
  - David McClosky, Eugene Charniak, Mark Johnson. *ACL 2006*










## Generative Models & Tasks

#### 2022

- SphericGAN: Semi-supervised Hyper-spherical Generative Adversarial Networks for Fine-grained Image Synthesis.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_SphericGAN_Semi-Supervised_Hyper-Spherical_Generative_Adversarial_Networks_for_Fine-Grained_Image_Synthesis_CVPR_2022_paper.pdf)
  - Tianyi Chen, Yi Liu, Yunfei Zhang, Si Wu, Yong Xu, Feng Liangbing, Hau San Wong. *CVPR 2022*

- OSSGAN: Open-Set Semi-Supervised Image Generation.
  [[pdf]](https://arxiv.org/abs/2204.14249) 
  [[code]](https://github.com/raven38/OSSGAN)
  - Kai Katsumata, Duc Minh Vo, Hideki Nakayama. *CVPR 2022*



#### 2021

- Semi-Supervised Single-Stage Controllable GANs for Conditional Fine-Grained Image Generation.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Semi-Supervised_Single-Stage_Controllable_GANs_for_Conditional_Fine-Grained_Image_Generation_ICCV_2021_paper.html)
  - Yingda Yin, Yingcheng Cai, He Wang, Baoquan Chen. *ICML 2021*

- Unified Robust Semi-Supervised Variational Autoencoder.
  [[pdf]](http://proceedings.mlr.press/v139/chen21a.html)
  - Xu Chen. *ICML 2021*

- Semi-Supervised Synthesis of High-Resolution Editable Textures for 3D Humans.
  [[pdf]](https://arxiv.org/abs/2103.17266)
  - Aamir Mustafa, Rafal K. Mantiuk. *CVPR 2021*

- Mask-Embedded Discriminator With Region-Based Semantic Regularization for Semi-Supervised Class-Conditional Image Synthesis.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Mask-Embedded_Discriminator_With_Region-Based_Semantic_Regularization_for_Semi-Supervised_Class-Conditional_Image_CVPR_2021_paper.pdf)
  - Yi Liu, Xiaoyang Huo, Tianyi Chen, Xiangping Zeng, Si Wu, Zhiwen Yu, Hau-San Wong. *CVPR 2021*



#### 2020

- Transformation Consistency Regularization- A Semi-Supervised Paradigm for Image-to-Image Translation.
  [[pdf]](https://arxiv.org/abs/2007.07867)
  - Aamir Mustafa, Rafal K. Mantiuk. *ECCV 2020*

- Regularizing Discriminative Capability of CGANs for Semi-Supervised Generative Learning.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Regularizing_Discriminative_Capability_of_CGANs_for_Semi-Supervised_Generative_Learning_CVPR_2020_paper.pdf) 
  - Yi Liu, Guangchang Deng, Xiangping Zeng, Si Wu, Zhiwen Yu, Hau-San Wong. *CVPR 2020*

- ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation.
  [[pdf]](https://arxiv.org/abs/2003.10557) 
  - Sharon Fogel, Hadar Averbuch-Elor, Sarel Cohen, Shai Mazor, Roee Litman. *CVPR 2020*

- Semi-Supervised Learning with Normalizing Flows.
  [[pdf]](https://arxiv.org/abs/1912.13025) 
  [[code]](https://github.com/izmailovpavel/flowgmm) 
  - Pavel Izmailov, Polina Kirichenko, Marc Finzi, Andrew Gordon Wilson. *ICML 2020*
 
- Semi-Supervised StyleGAN for Disentanglement Learning.
  [[pdf]](https://arxiv.org/abs/2003.03461) 
  [[code]](https://github.com/NVlabs/High-res-disentanglement-datasets) 
  - Weili Nie, Tero Karras, Animesh Garg, Shoubhik Debnath, Anjul Patney, Ankit B. Patel, Anima Anandkumar. *ICML 2020*

- Semi-Supervised Generative Modeling for Controllable Speech Synthesis.
  [[pdf]](https://arxiv.org/abs/1910.01709) 
  - Raza Habib, Soroosh Mariooryad, Matt Shannon, Eric Battenberg, RJ Skerry-Ryan, Daisy Stanton, David Kao, Tom Bagby. *ICLR 2020*

#### 2019

- MarginGAN: Adversarial Training in Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1704.03817)
  [[code]](https://github.com/xdu-DJhao/MarginGAN) 
  - Jinhao Dong, Tong Lin. *NeurIPS 2019*

- Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder.
  [[pdf]](https://arxiv.org/abs/1807.09875) 
  - Caio Corro, Ivan Titov. *ICLR 2019*

- Enhancing TripleGAN for Semi-Supervised Conditional Instance Synthesis and Classification.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Enhancing_TripleGAN_for_Semi-Supervised_Conditional_Instance_Synthesis_and_Classification_CVPR_2019_paper.pdf) 
  - Si Wu, Guangchang Deng, Jichang Li, Rui Li, Zhiwen Yu, Hau-San Wong. *CVPR 2019*

#### 2018

- Semi-supervised Adversarial Learning to Generate Photorealistic Face Images of New Identities from 3D Morphable Model.
  [[pdf]](https://arxiv.org/abs/1804.03675) 
  [[code]](https://github.com/barisgecer/facegan)
  - Baris Gecer, Binod Bhattarai, Josef Kittler, Tae-Kyun Kim. *ECCV 2018*

#### 2017

- Triple Generative Adversarial Nets.
  [[pdf]](https://arxiv.org/abs/1703.02291) 
  - Chongxuan Li, Kun Xu, Jun Zhu, Bo Zhang. *NeurIPS 2017*

- Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks.
  [[pdf]](https://arxiv.org/abs/1705.08850) 
  - Wei-Sheng Lai, Jia-Bin Huang, Ming-Hsuan Yang. *NeurIPS 2017*

- Semi-supervised Learning with GANs: Manifold Invariance with Improved Inference.
  [[pdf]](https://arxiv.org/abs/1705.08850) 
  - Abhishek Kumar, Prasanna Sattigeri, P. Thomas Fletcher. *NeurIPS 2017*

- Learning Disentangled Representations with Semi-Supervised Deep Generative Models.
  [[pdf]](https://arxiv.org/pdf/1705.09783v3.pdf) 
  [[code]](https://github.com/probtorch/probtorch)
  - N. Siddharth, Brooks Paige, Jan-Willem van de Meent, Alban Desmaison, Noah D. Goodman, Pushmeet Kohli, Frank Wood, Philip H.S. Torr. *NeurIPS 2017*

- Good Semi-supervised Learning that Requires a Bad GAN.
  [[pdf]](https://arxiv.org/pdf/1705.09783v3.pdf) 
  [[code]](https://github.com/kimiyoung/ssl_bad_gan)
  - Zihang Dai, Zhilin Yang, Fan Yang, William W. Cohen, Ruslan Salakhutdinov. *NeurIPS 2017*

- Infinite Variational Autoencoder for Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1611.07800) 
  - Ehsan Abbasnejad, Anthony Dick, Anton van den Hengel. *CVPR 2017*

- Semi-Supervised QA with Generative Domain-Adaptive Nets.
  [[pdf]](https://arxiv.org/abs/1702.02206) 
  - Zhilin Yang, Junjie Hu, Ruslan Salakhutdinov, William W. Cohen. *ACL 2017*

- Learning Loss Functions for Semi-supervised Learning via Discriminative Adversarial Networks.
  [[pdf]](https://arxiv.org/abs/1707.02198) 
  - Cicero Nogueira dos Santos, Kahini Wadhawan, Bowen Zhou. *JMLR 2017*

 
#### 2016

- Auxiliary Deep Generative Models.
  [[pdf]](https://arxiv.org/abs/1602.05473) 
  - Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, Ole Winther. *ICML 2016*

- Improved Techniques for Training GANs.
  [[pdf]](https://arxiv.org/abs/1606.03498) 
  - Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. *NeurIPS 2016*

- Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks.
  [[pdf]](https://arxiv.org/abs/1511.06390) 
  - Jost Tobias Springenberg. *ICLR 2016*

- Semi-Supervised Learning with Generative Adversarial Networks.
  [[pdf]](https://arxiv.org/abs/1606.01583) 
  - Augustus Odena. *Preprint 2016*

#### 2014

- Semi-supervised Learning with Deep Generative Models.
  [[pdf]](https://arxiv.org/abs/1406.5298) 
  - Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling. *NeurIPS 2014*












## Graph Based SSL

#### 2021

- Contrastive Graph Poisson Networks: Semi-Supervised Learning with Extremely Limited Labels.
  [[pdf]](https://openreview.net/forum?id=ek0RuhPoGiD)
  - Sheng Wan, Yibing Zhan, Liu Liu, Baosheng Yu, Shirui Pan, Chen Gong. *NeurIPS 2021*

- Topology-Imbalance Learning for Semi-Supervised Node Classification.
  [[pdf]](https://arxiv.org/abs/2110.04099)
  [[code]](https://github.com/victorchen96/ReNode)
  - Deli Chen, Yankai Lin, Guangxiang Zhao, Xuancheng Ren, Peng Li, Jie Zhou, Xu Sun. *NeurIPS 2021*

- Graph-BAS3Net: Boundary-Aware Semi-Supervised Segmentation Network With Bilateral Graph Convolution.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Graph-BAS3Net_Boundary-Aware_Semi-Supervised_Segmentation_Network_With_Bilateral_Graph_Convolution_ICCV_2021_paper.html)
  - Huimin Huang, Lanfen Lin, Yue Zhang, Yingying Xu, Jing Zheng, XiongWei Mao, Xiaohan Qian, Zhiyi Peng, Jianying Zhou, Yen-Wei Chen, Ruofeng Tong. *ICML 2021*

- Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization.
  [[pdf]](https://arxiv.org/abs/2102.06966)
  - Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath. *ICML 2021*

- Message Passing Adaptive Resonance Theory for Online Active Semi-supervised Learning.
  [[pdf]](https://arxiv.org/abs/2012.01227)
  - Taehyeong Kim, Injune Hwang, Hyundo Lee, Hyunseo Kim, Won-Seok Choi, Joseph J. Lim, Byoung-Tak Zhang *ICML 2021*

- Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and OOD Generalization.
  [[pdf]](https://arxiv.org/abs/2102.06966)
  - Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath *ICML 2021*

- Class-Attentive Diffusion Network for Semi-Supervised Classification.
  [[pdf]](https://arxiv.org/abs/2006.10222)
  [[code]](https://github.com/ljin0429/CAD-Net)
  - Jongin Lim, Daeho Um, Hyung Jin Chang, Dae Ung Jo, Jin Young Choi *AAAI 2021*

#### 2020

- Strongly local p-norm-cut algorithms for semi-supervised learning and local graph clustering.
  [[pdf]](https://arxiv.org/abs/2006.08569)
  [[code]](https://github.com/MengLiuPurdue/SLQ)
  - Meng Liu, David F. Gleich. *NeurIPS 2020*

- Deep Graph Pose: a semi-supervised deep graphicalmodel for improved animal pose tracking.
  [[pdf]](https://www.biorxiv.org/content/10.1101/2020.08.20.259705v1.full.pdf)
  - Anqi Wu, E. Kelly Buchanan et al. *NeurIPS 2020*

- Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label Rates.
  [[pdf]](https://arxiv.org/abs/2006.11184)
  - Jeff Calder, Brendan Cook, Matthew Thorpe, Dejan Slepcev. *ICML 2020*

- Density-Aware Graph for Deep Semi-Supervised Visual Recognition. 
  [[pdf]](https://arxiv.org/abs/2003.13194)
  - Suichan Li, Bin Liu, Dongdong Chen, Qi Chu, Lu Yuan, Nenghai Yu. *CVPR 2020*

- Shoestring: Graph-Based Semi-Supervised Classification With Severely Limited Labeled Data. 
  [[pdf]](https://arxiv.org/abs/1910.12976)
  - Wanyu Lin, Zhaolin Gao, Baochun Li. *CVPR 2020*

- InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. 
  [[pdf]](https://arxiv.org/abs/1908.01000)
  - Chunyan Xu, Zhen Cui, Xiaobin Hong, Tong Zhang, Jian Yang, Wei Liu. *ICLR 2020*

- Graph Inference Learning for Semi-supervised Classification. 
  [[pdf]](https://arxiv.org/abs/2001.06137) 
  - Chunyan Xu, Zhen Cui, Xiaobin Hong, Tong Zhang, Jian Yang, Wei Liu. *ICLR 2020*

#### 2019

- Improved Semi-Supervised Learning with Multiple Graphs. 
  [[pdf]](http://proceedings.mlr.press/v89/viswanathan19a/viswanathan19a.pdf)
  - Krishnamurthy Viswanathan, Sushant Sachdeva, Andrew Tomkins, Sujith Ravi, Partha Talukdar. *AISTATS 2019*

- Confidence-based Graph Convolutional Networks for Semi-Supervised Learning. 
  [[pdf]](http://proceedings.mlr.press/v89/vashishth19a/vashishth19a.pdf)
  [[code]](https://github.com/malllabiisc/ConfGCN)
  - Shikhar Vashishth, Prateek Yadav, Manik Bhandari, Partha Talukdar. *AISTATS 2019*

- Generalized Matrix Means for Semi-Supervised Learning with Multilayer Graphs. 
  [[pdf]](https://arxiv.org/abs/1910.14147)
  [[code]](https://github.com/melopeo/PM_SSL)
  - Pedro Mercado, Francesco Tudisco, Matthias Hein. *NeurIPS 2019*

- A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1910.14147)
  - Xuanqing Liu, Si Si, Xiaojin Zhu, Yang Li, Cho-Jui Hsieh. *NeurIPS 2019*

- Graph Agreement Models for Semi-Supervised Learning. 
  [[pdf]](http://papers.NeurIPS.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning.pdf)
  [[code]](https://github.com/tensorflow/neural-structured-learning)
  - Otilia Stretcu, Krishnamurthy Viswanathan, Dana Movshovitz-Attias, Emmanouil Platanios, Sujith Ravi, Andrew Tomkins. *NeurIPS 2019*

- Graph Based Semi-supervised Learning with Convolution Neural Networks to Classify Crisis Related Tweets. 
  [[pdf]](https://arxiv.org/abs/1805.06289)
  [[code]](https://github.com/mlzxzhou/keras-gnm)
  - Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, Bin Luo. *NeurIPS 2019*

- A Flexible Generative Framework for Graph-based Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1905.10769) 
  [[code]](https://github.com/jiaqima/G3NN)
  - Jiaqi Ma, Weijing Tang, Ji Zhu, Qiaozhu Mei. *NeurIPS 2019*

- Semi-Supervised Learning With Graph Learning-Convolutional Networks. 
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) 
  - Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, Bin Luo. *CVPR 2019*

- Label Efficient Semi-Supervised Learning via Graph Filtering. 
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Label_Efficient_Semi-Supervised_Learning_via_Graph_Filtering_CVPR_2019_paper.pdf) 
  - Qimai Li, Xiao-Ming Wu, Han Liu, Xiaotong Zhang, Zhichao Guan. *CVPR 2019*

- Graph Convolutional Networks Meet Markov Random Fields: Semi-Supervised Community Detection in Attribute Networks. 
  [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/3780) 
  - Di Jin, Ziyang Liu, Weihao Li, Dongxiao He, Weixiong Zhang. *AAAI 2019*

- Matrix Completion for Graph-Based Deep Semi-Supervised Learning. 
  [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/4438) 
  - Fariborz Taherkhani, Hadi Kazemi, Nasser M. Nasrabadi. *AAAI 2019*

- Bayesian Graph Convolutional Neural Networks for Semi-Supervised Classification. 
  [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/4531) 
  - Yingxue Zhang, Soumyasundar Pal, Mark Coates, Deniz Ustebay. *AAAI 2019*

#### 2018

- Semi-Supervised Learning via Compact Latent Space Clustering. 
  [[pdf]](http://proceedings.mlr.press/v80/kamnitsas18a/kamnitsas18a.pdf) 
  - Konstantinos Kamnitsas, Daniel Castro, Loic Le Folgoc, Ian Walker, Ryutaro Tanno, Daniel Rueckert, Ben Glocker, Antonio Criminisi, Aditya Nori. *ICML 2018*

- Bayesian Semi-supervised Learning with Graph Gaussian Processes. 
  [[pdf]](https://arxiv.org/abs/1809.04379) 
  - Yin Cheng Ng, Nicolo Colombo, Ricardo Silva. *NeurIPS 2018*

- Smooth Neighbors on Teacher Graphs for Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1711.00258) 
  - Yucen Luo, Jun Zhu, Mengxi Li, Yong Ren, Bo Zhang. *CVPR 2018*

- Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning. 
  [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16098) 
  - Y Qimai Li, Zhichao Han, Xiao-ming W. *AAAI 2018*

- Interpretable Graph-Based Semi-Supervised Learning via Flows. 
  [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16396) 
  - Raif M. Rustamov, James T. Klosowski. *AAAI 2018*


#### 2017

- Semi-Supervised Classification with Graph Convolutional Networks.
  [[pdf]](https://arxiv.org/abs/1609.02907)
  [[code]](https://github.com/tkipf/pygcn)
  - Thomas N. Kipf, Max Welling. *ICLR 2017*

#### 2016

-  Large-Scale Graph-Based Semi-Supervised Learning via Tree Laplacian Solver.
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11943)
  - Yan-Ming Zhang, Xu-Yao Zhang, Xiao-Tong Yuan, Cheng-Lin Liu. *AAAI 2016*

- Revisiting Semi-Supervised Learning with Graph Embeddings.
  [[pdf]](http://proceedings.mlr.press/v48/yanga16.pdf)
  [[code]](https://github.com/tkipf/gcn)
  - Zhilin Yang, William Cohen, Ruslan Salakhudinov. *ICML 2016*

#### 2014

- Graph-based Semi-supervised Learning: Realizing Pointwise Smoothness Probabilistically. 
  [[pdf]](http://proceedings.mlr.press/v32/fang14.pdf) 
  - Yuan Fang, Kevin Chang, Hady Lauw. *ICML 2014*

- A Multigraph Representation for Improved Unsupervised/Semi-supervised Learning of Human Actions. 
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Jones_A_Multigraph_Representation_2014_CVPR_paper.pdf) 
  - Simon Jones, Ling Shao. *CVPR 2014*

#### 2014

- Semi-supervised Eigenvectors for Locally-biased Learning. 
  [[pdf]](https://papers.NeurIPS.cc/paper/4560-semi-supervised-eigenvectors-for-locally-biased-learning.pdf) 
  - Toke Hansen, Michael W. Mahoney. *NeurIPS 2012*

#### 2012

- Semi-supervised Regression via Parallel Field Regularization. 
  [[pdf]](https://papers.NeurIPS.cc/paper/4398-semi-supervised-regression-via-parallel-field-regularization.pdf) 
  - Binbin Lin, Chiyuan Zhang, Xiaofei He. *NeurIPS 2011*

#### 2011

- Unsupervised and semi-supervised learning via L1-norm graph. 
  [[pdf]](http://www.escience.cn/system/file?fileId=69891) 
  - Feiping Nie, Hua Wang, Heng Huang, Chris Ding. *ICCV 2011*

- Semi-supervised Regression via Parallel Field Regularization. 
  [[pdf]](https://papers.NeurIPS.cc/paper/4398-semi-supervised-regression-via-parallel-field-regularization.pdf) 
  - Binbin Lin, Chiyuan Zhang, Xiaofei He. *NeurIPS 2011*

#### 2010 

- Semi-Supervised Learning with Max-Margin Graph Cuts. 
  [[pdf]](http://proceedings.mlr.press/v9/kveton10a/kveton10a.pdf) 
  - Branislav Kveton, Michal Valko, Ali Rahimi, Ling Huang. *AISTATS 2010*

- Large Graph Construction for Scalable Semi-Supervised Learning. 
  [[pdf]](https://icml.cc/Conferences/2010/papers/16.pdf) 
  - Wei Liu, Junfeng He, Shih-Fu Chang. *ICML 2010*

#### 2009

- Graph construction and b-matching for semi-supervised learning. 
  [[pdf]](https://dl.acm.org/doi/pdf/10.1145/1553374.1553432?download=true) 
  - Tony Jebara, Jun Wang, Shih-Fu Chang. *ICML 2009*

#### 2005 

- Cluster Kernels for Semi-Supervised Learning. 
  [[pdf]](http://papers.NeurIPS.cc/paper/2257-cluster-kernels-for-semi-supervised-learning.pdf) 
  - Olivier Chapelle, Jason Weston, Bernhard Scholkopf. *NeurIPS 2005*

#### 2004 

- Regularization and Semi-supervised Learning on Large Graphs. 
  [[pdf]](https://link.springer.com/content/pdf/10.1007%2Fb98522.pdf) 
  - Mikhail Belkin, Irina Matveeva, Partha Niyogi. *COLT 2004*








## Theory 

#### 2021

- Overcoming the curse of dimensionality with Laplacian regularization in semi-supervised learning.
  [[pdf]](https://arxiv.org/abs/2009.04324)
  -Vivien Cabannes, Loucas Pillaud-Vivien, Francis Bach, Alessandro Rudi. *NeurIPS 2021*

#### 2020

- Semi-Supervised Learning with Meta-Gradient. 
  [[pdf]](https://arxiv.org/pdf/2007.03966)
  - Xin-Yu Zhang, Hao-Lin Jia, Taihong Xiao, Ming-Ming Cheng, Ming-Hsuan Yang. *Preprint 2020*

- TCGM: An Information-Theoretic Framework for Semi-Supervised Multi-Modality Learning.
  [[pdf]](https://arxiv.org/abs/2007.06793)
  - Xinwei Sun, Yilun Xu, Peng Cao, Yuqing Kong, Lingjing Hu, Shanghang Zhang, Yizhou Wang. *ECCV 2020*

- Meta-Semi: A Meta-learning Approach for Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/2007.02394)
  - Yulin Wang, Jiayi Guo, Shiji Song, Gao Huang. *Preprint 2020*

- Not All Unlabeled Data are Equal: Learning to Weight Data in Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/2007.01293)
  - Zhongzheng Ren, Raymond A. Yeh, Alexander G. Schwing. *Preprint 2020*

#### 2019

- The information-theoretic value of unlabeled data in semi-supervised learning. 
  [[pdf]](https://arxiv.org/abs/1901.05515) 
  - Alexander Golovnev, David Pal, Balazs Szorenyi. *ICML 2019*

- Analysis of Network Lasso for Semi-Supervised Regression. 
  [[pdf]](http://proceedings.mlr.press/v89/jung19a/jung19a.pdf) 
  - Alexander Jung, Natalia Vesselinova. *AISTATS 2019*

- Semi-supervised clustering for de-duplication. 
  [[pdf]](http://proceedings.mlr.press/v89/kushagra19a/kushagra19a.pdf) 
  - Shrinu Kushagra, Shai Ben-David, Ihab Ilyas. *AISTATS 2019*
  
- Learning to Impute: A General Framework for Semi-supervised Learning.
  [[pdf]](https://arxiv.org/abs/1912.10364)
  [[code]](https://github.com/VICO-UoE/L2I)
  - Wei-Hong Li, Chuan-Sheng Foo, Hakan Bilen. *Preprint 2019*

#### 2018

- Semi-Supervised Learning with Competitive Infection Models. 
  [[pdf]](http://proceedings.mlr.press/v84/rosenfeld18a/rosenfeld18a.pdf) 
  - Nir Rosenfeld, Amir Globerson. *AISTATS 2018*

- The Pessimistic Limits and Possibilities of Margin-based Losses in Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1612.08875) 
  - Jesse H. Krijthe, Marco Loog. *NeurIPS 2018*

- The Sample Complexity of Semi-Supervised Learning with Nonparametric Mixture Models. 
  [[pdf]](https://papers.NeurIPS.cc/paper/8144-the-sample-complexity-of-semi-supervised-learning-with-nonparametric-mixture-models.pdf) 
  - Chen Dan, Liu Leqi, Bryon Aragam, Pradeep Ravikumar, Eric P. Xing. *NeurIPS 2018*

#### 2017

- Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data. 
  [[pdf]](http://proceedings.mlr.press/v70/sakai17a/sakai17a.pdf) 
  - Tomoya Sakai, Marthinus Christoffel Plessis, Gang Niu, Masashi Sugiyama. *ICML 2017*

#### 2016

- Semi-Supervised Learning with Adaptive Spectral Transform. 
  [[pdf]](http://proceedings.mlr.press/v51/liu16.pdf) 
  - Hanxiao Liu, Yiming Yang. *AISTATS 2016*

- Large Scale Distributed Semi-Supervised Learning Using Streaming Approximation. 
  [[pdf]](http://proceedings.mlr.press/v51/ravi16.pdf) 
  - Sujith Ravi, Qiming Diao. *AISTATS 2016*

#### 2014

- Wasserstein Propagation for Semi-Supervised Learning. 
  [[pdf]](http://proceedings.mlr.press/v32/solomon14.pdf) 
  - Justin Solomon, Raif Rustamov, Leonidas Guibas, Adrian Butscher. *ICML 2014*

- High Order Regularization for Semi-Supervised Learning of Structured Output Problems. 
  [[pdf]](http://proceedings.mlr.press/v32/lif14.pdf) 
  - Yujia Li, Rich Zemel. *ICML 2014*

#### 2013

- Correlated random features for fast semi-supervised learning. 
  [[pdf]](https://papers.NeurIPS.cc/paper/5000-correlated-random-features-for-fast-semi-supervised-learning.pdf) 
  - Brian McWilliams, David Balduzzi, Joachim M. Buhmann. *NeurIPS 2013*

- Squared-loss Mutual Information Regularization: A Novel Information-theoretic Approach to Semi-supervised Learning. 
  [[pdf]](http://proceedings.mlr.press/v28/niu13.pdf) 
  - Gang Niu, Wittawat Jitkrittum, Bo Dai, Hirotaka Hachiya, Masashi Sugiyama. *ICML 2013*

- Infinitesimal Annealing for Training Semi-Supervised Support Vector Machines. 
  [[pdf]](http://proceedings.mlr.press/v28/ogawa13a.pdf) 
  - Kohei Ogawa, Motoki Imamura, Ichiro Takeuchi, Masashi Sugiyama. *ICML 2013*

- Semi-supervised Clustering by Input Pattern Assisted Pairwise Similarity Matrix Completion. 
  [[pdf]](http://proceedings.mlr.press/v28/yi13.pdf) 
  - Jinfeng Yi, Lijun Zhang, Rong Jin, Qi Qian, Anil Jain. *ICML 2013*

#### 2012

- A Simple Algorithm for Semi-supervised Learning withImproved Generalization Error Bound. 
  [[pdf]](https://arxiv.org/pdf/1206.6412.pdf) 
  - Ming Ji, Tianbao Yang, Binbin Lin, Rong Jin, Jiawei Han. *ICML 2012*

- Deterministic Annealing for Semi-Supervised Structured Output Learning. 
  [[pdf]](http://proceedings.mlr.press/v22/dhillon12/dhillon12.pdf) 
  - Paramveer Dhillon, Sathiya Keerthi, Kedar Bellare, Olivier Chapelle, Sundararajan Sellamanickam. *AISTATS 2012*

#### 2011

- Semi-supervised Learning by Higher Order Regularization. 
  [[pdf]](http://proceedings.mlr.press/v15/zhou11b/zhou11b.pdf) 
  - Xueyuan Zhou, Mikhail Belkin. *AISTATS 2011*

- Error Analysis of Laplacian Eigenmaps for Semi-supervised Learning. 
  [[pdf]](http://proceedings.mlr.press/v15/zhou11c/zhou11c.pdf) 
  - Xueyuan Zhou, Nathan Srebro. *AISTATS 2011*

#### 2010

- Semi-Supervised Dimension Reduction for Multi-Label Classification. 
  [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/view/1911) 
  - Buyue Qian, Ian Davidson. *AAAI 2010*

- Semi-Supervised Learning via Generalized Maximum Entropy. 
  [[pdf]](http://proceedings.mlr.press/v9/erkan10a/erkan10a.pdf) 
  - Ayse Erkan, Yasemin Altun. *AISTATS 2010*

- Semi-supervised learning by disagreement. 
  [[pdf]](https://link.springer.com/content/pdf/10.1007/s10115-009-0209-z.pdf) 
  - Zhi-Hua Zhou, Ming Li. *Knowledge and Information Systems 2010*

#### 2009

- Semi-supervised Learning by Sparse Representation. 
  [[pdf]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.68) 
  - Shuicheng Yan and Huan Wang. *SIAM 2009*

#### 2008

- Worst-case analysis of the sample complexity of semi-supervised learning. 
  [[pdf]](http://colt2008.cs.helsinki.fi/papers/92-Ben-David.pdf) 
  - Shai Ben-David, Tyler Lu, David Pal. *COLT 2008*

#### 2007

- Generalization error bounds in semi-supervised classification under the cluster assumption. 
  [[pdf]](http://www.jmlr.org/papers/volume8/rigollet07a/rigollet07a.pdf) 
  - Philippe Rigollet. *JMLR 2007*

#### 2005

- Semi-supervised learning by entropy minimization. 
  [[pdf]](http://papers.NeurIPS.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf) 
  - Yves Grandvalet, Yoshua Bengio. *NeurIPS 2005*

- A co-regularization approach to semi-supervised learning with multiple views. 
  [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.7181&rep=rep1&type=pdf) 
  - Vikas Sindhwani, Partha Niyogi, Mikhail Belkin. *ICML 2005*

- Tri-Training: Exploiting Unlabeled DataUsing Three Classifiers. 
  [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2431&rep=rep1&type=pdf) 
  - Zhou Zhi-Hua and Li Ming. *IEEE Transactions on knowledge and Data Engineering 2005*

#### 2003

- Semi-supervised learning using gaussian fields and harmonic functions. 
  [[pdf]](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf?source=post_page---------------------------) 
  - Xiaojin Zhu, Zoubin Ghahramani, John Lafferty. *ICML 2003*

- Semi-supervised learning of mixture models. 
  [[pdf]](https://www.aaai.org/Papers/ICML/2003/ICML03-016.pdf) 
  - Fabio Gagliardi Cozman, Ira Cohen, Marcelo Cesar Cirelo. *ICML 2003*

#### 2002

- Learning from labeled and unlabeled data with label propagation. 
  [[pdf]](https://pdfs.semanticscholar.org/8a6a/114d699824b678325766be195b0e7b564705.pdf) 
  - Xiaojin Zhu, Zoubin Ghahramani. *NeurIPS 2002*

#### 1998

- Combining labeled and unlabeled data with co-training. 
  [[pdf]](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf) 
  - Tom Michael Mitchell, Tom Mitchell. *COLT 1998*









## Reinforcement Learning, Meta-Learning & Robotics

#### 2020

- Semi-Supervised Neural Architecture Search.
  [[pdf]](https://arxiv.org/abs/2002.10389)
  [[code]](https://github.com/renqianluo/SemiNAS)
  - Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Enhong Chen, Tie-Yan Liu. *NeurIPS 2020*

- Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery.
  [[pdf]](https://arxiv.org/abs/1907.08225)
  - Kristian Hartikainen, Xinyang Geng, Tuomas Haarnoja, Sergey Levine. *ICLR 2020*

#### 2018

- Meta-Learning for Semi-Supervised Few-Shot Classification.
  [[pdf]](https://arxiv.org/abs/1803.00676)
  [[code]](https://github.com/renmengye/few-shot-ssl-public)
  - Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richard S. Zemel. *ICLR 2018*

#### 2017

- Generalizing Skills with Semi-Supervised Reinforcement Learning. 
  [[pdf]](https://arxiv.org/abs/1612.00429) 
  - Takeru Miyato, Andrew M. Dai, Ian Goodfellow. *ICLR 2017*









## Regression

#### 2018

- Minimax-optimal semi-supervised regression on unknown manifolds. 
  [[pdf]](http://proceedings.mlr.press/v54/moscovich17a/moscovich17a.pdf) 
  - Amit Moscovich, Ariel Jaffe, Nadler Boaz. *AISTATS 2017*

- Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive Variance. 
  [[pdf]](https://arxiv.org/abs/1805.10407) 
  [[code]](https://github.com/ermongroup/ssdkl)
  - Neal Jean, Sang Michael Xie, Stefano Ermon. *NeurIPS 2018*

#### 2017

- Learning Safe Prediction for Semi-Supervised Regression. 
  [[pdf]](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14587) 
  - Yu-Feng Li, Han-Wen Zha, Zhi-Hua Zhou. *AAAI 2017*

#### 2015

- Semi-Supervised Factored Logistic Regression for High-Dimensional Neuroimaging Data. 
  [[pdf]](https://papers.NeurIPS.cc/paper/5646-semi-supervised-factored-logistic-regression-for-high-dimensional-neuroimaging-data.pdf) 
  - Danilo Bzdok, Michael Eickenberg, Olivier Grisel, Bertrand Thirion, Ga ̈el Varoquaux. *NeurIPS 2015*





## Other

#### 2022

- RSCFed: Random Sampling Consensus Federated Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/2203.13993) 
  - Xiaoxiao Liang, Yiqun Lin, Huazhu Fu, Lei Zhu, Xiaomeng Li. *CVPR 2022*

- NP-Match: When Neural Processes meet Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/2207.01066) 
  - Jianfeng Wang, Thomas Lukasiewicz, Daniela Massiceti, Xiaolin Hu, Vladimir Pavlovic, Alexandros Neophytou. *ICML 2022*

#### 2018

- Semi-Supervised Learning on Data Streams via Temporal Label Propagation. 
  [[pdf]](http://proceedings.mlr.press/v80/wagner18a/wagner18a.pdf) 
  - Tal Wagner, Sudipto Guha, Shiva Kasiviswanathan, Nina Mishra. *ICML 2018*

#### 2017

- Kernelized Evolutionary Distance Metric Learning for Semi-Supervised Clustering. 
  [[pdf]](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14714) 
  -  Wasin Kalintha, Satoshi Ono, Masayuki Numao, Ken-ichi Fukui. *AAAI 2017*

#### 2016

- Robust Semi-Supervised Learning through Label Aggregation. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12312) 
  - Yan Yan, Zhongwen Xu, Ivor W. Tsang, Guodong Long, Yi Yang. *AAAI 2016*

- Semi-Supervised Dictionary Learning via Structural Sparse Preserving. 
  [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11980) 
  - Di Wang, Xiaoqin Zhang, Mingyu Fan, Xiuzi Ye. *AAAI 2016*

#### 2013

- Efficient Semi-supervised and Active Learning of Disjunctions. 
  [[pdf]](http://proceedings.mlr.press/v28/balcan13.pdf) 
  - Nina Balcan, Christopher Berlind, Steven Ehrlich, Yingyu Liang. *ICML 2013*


## Talks
- Semi-Supervised Learning and Unsupervised Distribution Alignment. [[youtube]](https://www.youtube.com/watch?v=PXOhi6m09bA).
  - *CS294-158-SP20 UC Berkeley.* 
- Semi-Supervised Learning and Unsupervised Distribution Alignment. [[youtube]](https://www.youtube.com/watch?v=j_-JaMPnhr0).
  - *Pydata, Andreas Merentitis, Carmine Paolino, Vaibhav Singh.*
- Overview of Unsupervised & Semi-supervised learning. [[youtube]](https://www.youtube.com/watch?v=tnpXLK_AS_U).
  - *AISC, Shazia Akbar.* 
- Semi-Supervised Learning. [[youtube]](https://www.youtube.com/watch?v=OMRlnKupsXM)
  [[slides]](https://www.cs.cmu.edu/%7Etom/10701_sp11/slides/LabUnlab-3-17-2011.pdf).
  - *CMU Machine Learning 10-701, Tom M. Mitchell .* 


## Thesis
- Fundamental limitations of semi-supervised learnin. *Tyler Tian Lu.* [[pdf]](https://uwspace.uwaterloo.ca/bitstream/handle/10012/4387/lumastersthesis_electronic.pdf).
- Semi-Supervised Learning with Graphs. *Xiaojin Zhu.* [[pdf]](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf).
- Semi-Supervised Learning for Natural Language. *Percy Liang.* [[pdf]](https://www-cs.stanford.edu/~pliang/papers/meng-thesis.pdf).


## Blogs
- An overview of proxy-label approaches for semi-supervised learning. *Sebastian Ruder.* [[link]](https://ruder.io/semi-supervised/index.html).
- The Illustrated FixMatch for Semi-Supervised Learning. *Amit Chaudhary.* [[link]](https://amitness.com/2020/03/fixmatch-semi-supervised/)
- An Overview of Deep Semi-Supervised Learning. *Yassine Ouali* [[link]](https://yassouali.github.io/ml-blog/deep-semi-supervised/)
- Semi-Supervised Learning in Computer Vision. *Amit Chaudhary* [[link]](https://amitness.com/2020/07/semi-supervised-learning/)
