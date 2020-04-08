# Awesome Semi-Supervised Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

<p align="center">
  <img width="250" src="https://camo.githubusercontent.com/1131548cf666e1150ebd2a52f44776d539f06324/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f6d61737465722f6d656469612f6c6f676f2e737667" "Awesome!">
</p>

A curated list of awesome Semi-Supervised Learning resources. Inspired by [awesome-domain-adaptation](https://github.com/kjw0612/awesome-deep-vision), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-self-supervised-learning](https://github.com/jason718/awesome-self-supervised-learning)

#### What is Self-Supervised Learning?

#### Why Self-Supervised?

## Contributing
<p align="center">
  <img src="http://cdn1.sportngin.com/attachments/news_article/7269/5172/needyou_small.jpg" alt="We Need You!">
</p>

If you find any errors, or you wish to add some papers, please feel free to contribute to this list by contacting [me](https://yassouali.github.io/) or by creating a [pull request](https://github.com/yassouali/awesome-semi-supervised-learning/pulls) following the Markdown format bellow:

```markdown
- Paper Name. 
  [[pdf]](link) 
  [[code]](link)
  - Author 1, Author 2, and Author 3. *Conference Year*
```

## Table of Contents
  - [Computer Vision](#computer-vision)
  - [Machine Learning](#machine-learning)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Robotics](#robotics)
  - [NLP](#nlp)
  - [Talks](#talks)
  - [Thesis](#thesis)
  - [Blog](#blog)

## Computer Vision
Note that for Image and Object segmentation tasks, we also include weakly-supervised
learning methods, that uses weak labels (eg, image classes) for object detaction or image
segmentations.









## Surveys & Overview

- Realistic Evaluation of Deep Semi-Supervised Learning Algorithms 
  [[pdf]](https://arxiv.org/abs/1804.09170)
  [[code]](https://github.com/brain-research/realistic-ssl-evaluation)
  - Avital Oliver, Augustus Odena, Colin Raffel, Ekin D. Cubuk, Ian J. Goodfellow. *NIPS 2018*



### Image Classification

#### 2020

- ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring. 
  [[pdf]](https://arxiv.org/abs/1911.09785)
  [[code]](https://github.com/google-research/remixmatch)
  - David Berthelot, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, Colin Raffel. *ICLR 2020*

- DivideMix: Learning with Noisy Labels as Semi-supervised Learning. 
  [[pdf]](https://openreview.net/pdf?id=HJgExaVtwr)
  [[code]](https://github.com/LiJunnan1992/DivideMix)
  - Junnan Li, Richard Socher, Steven C.H. Hoi. *ICLR 2020*

#### 2019

- MixMatch: A Holistic Approach to Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1909.01804)
  [[code]](https://github.com/google-research/mixmatch)
  - David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel. *NIPS 2019*

- Dual Student: Breaking the Limits of the Teacher in Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1909.01804)
  [[code]](https://github.com/ZHKKKe/DualStudent)
  - Zhanghan Ke, Daoye Wang, Qiong Yan, Jimmy Ren, Rynson W.H. Lau. *ICCV 2019*

- S4L: Self-Supervised Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1905.03670)
  [[code]](https://github.com/google-research/s4l)
  - Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, Lucas Beyer. *ICCV 2019*

- Semi-Supervised Learning by Augmented Distribution Alignment. 
  [[pdf]](https://arxiv.org/abs/1905.08171)
  [[code]](https://github.com/qinenergy/adanet)
  - Qin Wang, Wen Li, Luc Van Gool. *ICCV 2019*

- Tangent-Normal Adversarial Regularization for Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1808.06088) 
  - Bing Yu, Jingfeng Wu, Jinwen Ma, Zhanxing Zhu. *CVPR 2019*

- Label Propagation for Deep Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1904.04717) 
  - Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, Ondrej Chum. *CVPR 2019*

- Joint Representative Selection and Feature Learning: A Semi-Supervised Approach. 
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Joint_Representative_Selection_and_Feature_Learning_A_Semi-Supervised_Approach_CVPR_2019_paper.pdf) 
  - Suchen Wang, Jingjing Meng, Junsong Yuan, Yap-Peng Tan. *CVPR 2019*

- Mutual Learning of Complementary Networks via Residual Correction for Improving Semi-Supervised Classification. 
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Mutual_Learning_of_Complementary_Networks_via_Residual_Correction_for_Improving_CVPR_2019_paper.pdf) 
  - Si Wu, Jichang Li, Cheng Liu, Zhiwen Yu, Hau-San Wong. *CVPR 2019*

- There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average. 
  [[pdf]](https://openreview.net/pdf?id=rkgKBhA5Y7)
  [[code]](https://github.com/benathi/fastswa-semi-sup)
  - Ben Athiwaratkun, Marc Finzi, Pavel Izmailov, Andrew Gordon Wilson. *ICLR 2019*

#### 2018

- Deep Co-Training for Semi-Supervised Image Recognition.
  [[pdf]](https://arxiv.org/abs/1803.05984)[[code]](https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition)
  - Siyuan Qiao, Wei Shen, Zhishuai Zhang, Bo Wang, Alan Yuille. *ECCV 2018*

- HybridNet: Classification and Reconstruction Cooperation for Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1807.11407) 
  - Thomas Robert, Nicolas Thome, Matthieu Cord . *ECCV 2018*

- Transductive Centroid Projection for Semi-supervised Large-scale Recognition.
  [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yu_Liu_Transductive_Centroid_Projection_ECCV_2018_paper.pdf) 
  - Yu Liu, Guanglu Song, Jing Shao, Xiao Jin, Xiaogang Wang. *ECCV 2018*

- Semi-Supervised Deep Learning with Memory.
  [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yanbei_Chen_Semi-Supervised_Deep_Learning_ECCV_2018_paper.pdf) 
  - Yanbei Chen, Xiatian Zhu, Shaogang Gong. *ECCV 2018*

- SaaS: Speed as a Supervisorfor Semi-supervised Learning.
  [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Safa_Cicek_SaaS_Speed_as_ECCV_2018_paper.pdf) 
  - Safa Cicek, Alhussein Fawzi and Stefano Soatto. *ECCV 2018*

#### 2017

- Learning by Association -- A Versatile Semi-Supervised Training Method for Neural Networks.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Haeusser_Learning_by_Association_CVPR_2017_paper.pdf) 
  - Philip Haeusser, Alexander Mordvintsev, Daniel Cremers. *CVPR 2017*

- Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data.
  [[pdf]](https://arxiv.org/abs/1610.05755) 
  [[code]](https://github.com/kamathhrishi/PATE)
  - Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian Goodfellow, Kunal Talwar.  *ICLR 2017*

-  Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
  [[pdf]](https://arxiv.org/abs/1703.01780) 
  [[code]](https://github.com/CuriousAI/mean-teacher)
  - Samuli Laine, Timo Aila.  *NIPS 2017*

-  Temporal Ensembling for Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1610.05755) 
  [[code]](https://github.com/benathi/fastswa-semi-sup)
  - Samuli Laine, Timo Aila.  *ICLR 2017*


#### 2016

- Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning.
  [[pdf]](https://papers.nips.cc/paper/6333-regularization-with-stochastic-transformations-and-perturbations-for-deep-semi-supervised-learning.pdf) 
  - Mehdi Sajjadi, Mehran Javanmardi, Tolga Tasdizen. *NIPS 2016*

#### 2015

- Learning Semi-Supervised Representation Towards a Unified Optimization Framework for Semi-Supervised Learning.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2015/papers/Li_Learning_Semi-Supervised_Representation_ICCV_2015_paper.pdf) 
  - Chun-Guang Li, Zhouchen Lin, Honggang Zhang, Jun Guo. *ICCV 2015*

- Semi-Supervised Low-Rank Mapping Learning for Multi-Label Classification.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Jing_Semi-Supervised_Low-Rank_Mapping_2015_CVPR_paper.pdf) 
  - Liping Jing, Liu Yang, Jian Yu, Michael K. Ng . *CVPR 2015*

- Semi-Supervised Learning With Explicit Relationship Regularization.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Kim_Semi-Supervised_Learning_With_2015_CVPR_paper.pdf) 
  - Kwang In Kim, James Tompkin, Hanspeter Pfister, Christian Theobalt. *CVPR 2015*

- Semi-supervised Learning with Ladder Networks.
  [[pdf]](https://arxiv.org/abs/1507.02672)
  [[code]](https://github.com/CuriousAI/ladder) 
  - Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, Tapani Raiko. *NIPS 2015*

#### 2014

- Semi-supervised Spectral Clustering for Image Set Classification.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Mahmood_Semi-supervised_Spectral_Clustering_2014_CVPR_paper.pdf) 
  - Arif Mahmood, Ajmal Mian, Robyn Owens. *CVPR 2014*

#### 2013

- Ensemble Projection for Semi-supervised Image Classification.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2013/papers/Dai_Ensemble_Projection_for_2013_ICCV_paper.pdf) 
  - Dengxin Dai, Luc Van Gool. *ICCV 2013*

- Dynamic Label Propagation for Semi-supervised Multi-class Multi-label Classification.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2013/papers/Wang_Dynamic_Label_Propagation_2013_ICCV_paper.pdf) 
  - Bo Wang, Zhuowen Tu, John K. Tsotsos. *ICCV 2013*
































### Image Retrieval

#### 2018

- Semi-Supervised Generative Adversarial Hashing for Image Retrieval.
  [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guanan_Wang_Semi-Supervised_Generative_Adversarial_ECCV_2018_paper.pdf)
  - Guan'an Wang, Qinghao Hu, Jian Cheng, Zengguang Hou. *ECCV 2018*

#### 2007

- Semi-supervised Discriminant Analysis.
  [[pdf]](http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/ICCV07_dengcai_SDA.pdf)
  - Deng Cai, Xiaofei He, Jiawei Han. *ICCV 2007*


















### Semantic and Instance Segmentation

- Semi-Supervised Semantic Segmentation with Cross-Consistency Training.
  [[pdf]](https://arxiv.org/abs/2003.09005) 
  - Yassine Ouali, Céline Hudelot, Myriam Tami. *CVPR 2020*

- Semi-Supervised Semantic Image Segmentation with Self-correcting Networks.
  [[pdf]](https://arxiv.org/abs/1811.07073) 
  - Mostafa S. Ibrahim, Arash Vahdat, Mani Ranjbar, William G. Macready. *CVPR 2020*

#### 2019

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
  - Seunghoon Hong, Hyeonwoo Noh, Bohyung Han. *NIPS 2015*

- BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1503.01640)
  - Jifeng Dai, Kaiming He, Jian Sun. *CVPR 2015*

- Simple Does It: Weakly Supervised Instance and Semantic Segmentation.
  [[pdf]](https://arxiv.org/abs/1603.07485)
  [[code]](https://github.com/johnnylu305/Simple-does-it-weakly-supervised-instance-and-semantic-segmentation)
  - Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele. *CVPR 2017*

#### 2016

- SSHMT: Semi-supervised Hierarchical Merge Tree for Electron Microscopy Image Segmentation.
  [[pdf]](https://arxiv.org/abs/1608.04051) 
  - Ting Liu, Miaomiao Zhang, Mehran Javanmardi, Nisha Ramesh, Tolga Tasdizen. *ECCV 2015*

#### 2013

- Semi-supervised Learning for Large Scale Image Cosegmentation.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2013/papers/Wang_Semi-supervised_Learning_for_2013_ICCV_paper.pdf) 
  - Zhengxiang Wang, Rujie Liu. *ICCV 2013*




















### Object Detection

#### 2019

- Consistency-based Semi-supervised Learning forObject Detection.
  [[pdf]](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection.pdf)
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
  - Evan Racah, Christopher Beckham, Tegan Maharaj, Samira Ebrahimi Kahou, Prabhat, Christopher Pal. *NIPS 2017*

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

#### 2020

- Deep Semi-Supervised Anomaly Detection.
  [[pdf]](https://arxiv.org/abs/1906.02694)
  [[code]](https://github.com/lukasruff/Deep-SAD-PyTorch)
  - Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Alexander Binder, Emmanuel Müller, Klaus-Robert Müller, Marius Kloft. *ICLR 2020*

#### 2019

- Learning to Self-Train for Semi-Supervised Few-Shot Classification.
  [[pdf]](https://arxiv.org/abs/1906.00562)
  [[code]](https://github.com/xinzheli1217/learning-to-self-train)
  - Xinzhe Li, Qianru Sun, Yaoyao Liu, Shibao Zheng, Qin Zhou, Tat-Seng Chua, Bernt Schiele. *NIPS 2019*

- Multi-label Co-regularization for Semi-supervised Facial Action Unit Recognition.
  [[pdf]](https://arxiv.org/abs/1906.02694)
  [[code]](https://github.com/nxsEdson/MLCR)
  - Xuesong Niu, Hu Han, Shiguang Shan, Xilin Chen. *NIPS 2019*

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

- Improving Landmark Localization With Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1709.01591) 
  - Sina Honari, Pavlo Molchanov, Stephen Tyree, Pascal Vincent, Christopher Pal, Jan Kautz. *CVPR 2018*

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
  - Xiaobo Wang, Xiaojie Guo, Stan Z. Li . *ICCV 2015*

- Semi-Supervised Zero-Shot Classification With Label Representation Learning.
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2015/papers/Li_Semi-Supervised_Zero-Shot_Classification_ICCV_2015_paper.pdf) 
  - Xin Li, Yuhong Guo, Dale Schuurmans. *ICCV 2015*

#### 2014

- Semi-Supervised Coupled Dictionary Learning for Person Re-identification.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Liu_Semi-Supervised_Coupled_Dictionary_2014_CVPR_paper.pdf) 
  - Xiao Liu, Mingli Song, Dacheng Tao, Xingchen Zhou, Chun Chen, Jiajun Bu. *CVPR 2014*

#### 2013

- Semi-supervised Learning with Constraints for Person Identification in Multimedia Data.
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2013/papers/Bauml_Semi-supervised_Learning_with_2013_CVPR_paper.pdf) 
  - Martin Bauml, Makarand Tapaswi, Rainer Stiefelhagen. *CVPR 2013*













## Multi-modal SSL

#### 2013

- Heterogeneous Image Features Integration via Multi-modal Semi-supervised Learning Model. 
  [[pdf]](http://openaccess.thecvf.com/content_iccv_2013/papers/Cai_Heterogeneous_Image_Features_2013_ICCV_paper.pdf) 
  - Xiao Cai, Feiping Nie, Weidong Cai, Heng Huang. *Conf*






## Regression

#### 2015

- Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive Variance. 
  [[pdf]](https://arxiv.org/abs/1805.10407) 
  [[code]](https://github.com/ermongroup/ssdkl)
  - Danilo Bzdok, Michael Eickenberg, Olivier Grisel, Bertrand Thirion, Ga ̈el Varoquaux. *NIPS 2018*


#### 2015

- Semi-Supervised Factored Logistic Regression for High-Dimensional Neuroimaging Data. 
  [[pdf]](https://papers.nips.cc/paper/5646-semi-supervised-factored-logistic-regression-for-high-dimensional-neuroimaging-data.pdf) 
  - Danilo Bzdok, Michael Eickenberg, Olivier Grisel, Bertrand Thirion, Ga ̈el Varoquaux. *NIPS 2015*












## Generative Models

#### 2020

- Semi-Supervised Generative Modeling for Controllable Speech Synthesis.
  [[pdf]](https://arxiv.org/abs/1910.01709) 
  - Raza Habib, Soroosh Mariooryad, Matt Shannon, Eric Battenberg, RJ Skerry-Ryan, Daisy Stanton, David Kao, Tom Bagby. *ICLR 2019*

#### 2019

- MarginGAN: Adversarial Training in Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1807.09875)
  [[code]](https://github.com/xdu-DJhao/MarginGAN) 
  - Jinhao Dong, Tong Lin. *NIPS 2019*

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

- Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks.
  [[pdf]](https://arxiv.org/abs/1705.08850) 
  - Wei-Sheng Lai, Jia-Bin Huang, Ming-Hsuan Yang. *NIPS 2017*

- Semi-supervised Learning with GANs: Manifold Invariance with Improved Inference.
  [[pdf]](https://arxiv.org/abs/1705.08850) 
  - Abhishek Kumar, Prasanna Sattigeri, P. Thomas Fletcher. *NIPS 2017*

- Learning Disentangled Representations with Semi-Supervised Deep Generative Models.
  [[pdf]](https://arxiv.org/pdf/1705.09783v3.pdf) 
  [[code]](https://github.com/probtorch/probtorch)
  - N. Siddharth, Brooks Paige, Jan-Willem van de Meent, Alban Desmaison, Noah D. Goodman, Pushmeet Kohli, Frank Wood, Philip H.S. Torr. *NIPS 2017*

- Good Semi-supervised Learning that Requires a Bad GAN.
  [[pdf]](https://arxiv.org/pdf/1705.09783v3.pdf) 
  [[code]](https://github.com/kimiyoung/ssl_bad_gan)
  - Zihang Dai, Zhilin Yang, Fan Yang, William W. Cohen, Ruslan Salakhutdinov. *NIPS 2017*

- Infinite Variational Autoencoder for Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/abs/1611.07800) 
  - Ehsan Abbasnejad, Anthony Dick, Anton van den Hengel. *CVPR 2017*

#### 2016

-  Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks.
  [[pdf]](https://arxiv.org/abs/1511.06390) 
  - Jost Tobias Springenberg. *ICLR 2016*

#### 2014

- Semi-supervised Learning with Deep Generative Models.
  [[pdf]](https://arxiv.org/abs/1406.5298) 
  - Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling. *NIPS 2016*






















## Graph Based SSL

#### 2020

- InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. 
  [[pdf]](https://arxiv.org/abs/1908.01000)
  - Chunyan Xu, Zhen Cui, Xiaobin Hong, Tong Zhang, Jian Yang, Wei Liu. *ICLR 2020*

- Graph Inference Learning for Semi-supervised Classification. 
  [[pdf]](https://arxiv.org/abs/2001.06137) 
  - Chunyan Xu, Zhen Cui, Xiaobin Hong, Tong Zhang, Jian Yang, Wei Liu. *ICLR 2020*

#### 2019

- Generalized Matrix Means for Semi-Supervised Learning with Multilayer Graphs. 
  [[pdf]](https://arxiv.org/abs/1910.14147)
  [[code]](https://github.com/melopeo/PM_SSL)
  - Pedro Mercado, Francesco Tudisco, Matthias Hein. *NIPS 2019*

- A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1910.14147)
  - Xuanqing Liu, Si Si, Xiaojin Zhu, Yang Li, Cho-Jui Hsieh. *NIPS 2019*

- Graph Agreement Models for Semi-Supervised Learning. 
  [[pdf]](http://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning.pdf)
  [[code]](https://github.com/tensorflow/neural-structured-learning)
  - Otilia Stretcu, Krishnamurthy Viswanathan, Dana Movshovitz-Attias, Emmanouil Platanios, Sujith Ravi, Andrew Tomkins. *NIPS 2019*

- Graph Based Semi-supervised Learning with Convolution Neural Networks to Classify Crisis Related Tweets. 
  [[pdf]](https://arxiv.org/abs/1805.06289)
  [[code]](https://github.com/mlzxzhou/keras-gnm)
  - Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, Bin Luo. *NIPS 2019*

- A Flexible Generative Framework for Graph-based Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1905.10769) 
  [[code]](https://github.com/jiaqima/G3NN)
  - Jiaqi Ma, Weijing Tang, Ji Zhu, Qiaozhu Mei. *NIPS 2019*

- Semi-Supervised Learning With Graph Learning-Convolutional Networks. 
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) 
  - Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, Bin Luo. *CVPR 2019*

- Label Efficient Semi-Supervised Learning via Graph Filtering. 
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Label_Efficient_Semi-Supervised_Learning_via_Graph_Filtering_CVPR_2019_paper.pdf) 
  - Qimai Li, Xiao-Ming Wu, Han Liu, Xiaotong Zhang, Zhichao Guan. *CVPR 2019*

#### 2018

- Bayesian Semi-supervised Learning with Graph Gaussian Processes. 
  [[pdf]](https://arxiv.org/abs/1809.04379) 
  - Yin Cheng Ng, Nicolo Colombo, Ricardo Silva. *NIPS 2018*

- Smooth Neighbors on Teacher Graphs for Semi-Supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1711.00258) 
  - Yucen Luo, Jun Zhu, Mengxi Li, Yong Ren, Bo Zhang. *CVPR 2018*

#### 2017

- Semi-Supervised Classification with Graph Convolutional Networks.
  [[pdf]](https://arxiv.org/abs/1609.02907)
  [[code]](https://github.com/tkipf/pygcn)
  - Thomas N. Kipf, Max Welling. *ICLR 2017*

#### 2014

- A Multigraph Representation for Improved Unsupervised/Semi-supervised Learning of Human Actions. 
  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Jones_A_Multigraph_Representation_2014_CVPR_paper.pdf) 
  - Simon Jones, Ling Shao. *CVPR 2014*

#### 2014

- Semi-supervised Eigenvectors for Locally-biased Learning. 
  [[pdf]](https://papers.nips.cc/paper/4560-semi-supervised-eigenvectors-for-locally-biased-learning.pdf) 
  - Toke Hansen, Michael W. Mahoney. *NIPS 2012*

#### 2012

- Semi-supervised Regression via Parallel Field Regularization. 
  [[pdf]](https://papers.nips.cc/paper/4398-semi-supervised-regression-via-parallel-field-regularization.pdf) 
  - Binbin Lin, Chiyuan Zhang, Xiaofei He. *NIPS 2011*

#### 2011

- Unsupervised and semi-supervised learning via L1-norm graph. 
  [[pdf]](http://www.escience.cn/system/file?fileId=69891) 
  - Feiping Nie, Hua Wang, Heng Huang, Chris Ding. *ICCV 2011*

- Semi-supervised Regression via Parallel Field Regularization. 
  [[pdf]](https://papers.nips.cc/paper/4398-semi-supervised-regression-via-parallel-field-regularization.pdf) 
  - Binbin Lin, Chiyuan Zhang, Xiaofei He. *NIPS 2011*

#### 2010 

- Semi-Supervised Learning with Max-Margin Graph Cuts. 
  [[pdf]](http://proceedings.mlr.press/v9/kveton10a/kveton10a.pdf) 
  - Branislav Kveton, Michal Valko, Ali Rahimi, Ling Huang. *AISTATS 2010*



















## Reinforcement Learning, Meta-Learning & Robotics

#### 2018

- Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery.
  [[pdf]](https://arxiv.org/abs/1907.08225)
  [[code]](https://github.com/renmengye/few-shot-ssl-public)
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


















## NLP

### Classification

#### 2018

- Semi-Supervised Learning with Declaratively Specified Entropy Constraints. 
  [[pdf]](https://arxiv.org/abs/1605.07725) 
  - Haitian Sun, William W. Cohen, Lidong Bing. *NIPS 2018*

#### 2017

- Adversarial Training Methods for Semi-Supervised Text Classification. 
  [[pdf]](https://arxiv.org/abs/1605.07725) 
  [[code]](https://github.com/tensorflow/models/tree/master/research/adversarial_text)
  - Chelsea Finn, Tianhe Yu, Justin Fu, Pieter Abbeel, Sergey Levine. *ICLR 2017*

#### 2015

- Semi-supervised Sequence Learning. 
  [[pdf]](https://arxiv.org/abs/1511.01432) 
  [[code]](https://github.com/tensorflow/models/tree/master/research/adversarial_text)
  - Andrew M. Dai, Quoc V. Le. *NIPS 2015*

- Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding. 
  [[pdf]](https://arxiv.org/abs/1504.01255) 
  - Rie Johnson, Tong Zhang. *NIPS 2015*

























## Theory

#### 2018

- The Pessimistic Limits and Possibilities of Margin-based Losses in Semi-supervised Learning. 
  [[pdf]](https://arxiv.org/abs/1612.08875) 
  - Jesse H. Krijthe, Marco Loog. *NIPS 2018*

- The Sample Complexity of Semi-Supervised Learning with Nonparametric Mixture Models. 
  [[pdf]](https://papers.nips.cc/paper/8144-the-sample-complexity-of-semi-supervised-learning-with-nonparametric-mixture-models.pdf) 
  - Chen Dan, Liu Leqi, Bryon Aragam, Pradeep Ravikumar, Eric P. Xing. *NIPS 2018*

#### 2013

- Correlated random features for fast semi-supervised learning. 
  [[pdf]](https://papers.nips.cc/paper/5000-correlated-random-features-for-fast-semi-supervised-learning.pdf) 
  - Brian McWilliams, David Balduzzi, Joachim M. Buhmann. *NIPS 2015*

#### 2012

- A Simple Algorithm for Semi-supervised Learning withImproved Generalization Error Bound. 
  [[pdf]](https://arxiv.org/pdf/1206.6412.pdf) 
  - Ming Ji, Tianbao Yang, Binbin Lin, Rong Jin, Jiawei Han. *ICML 2012*

#### 2011

- Semi-supervised Learning by Higher Order Regularization. 
  [[pdf]](http://proceedings.mlr.press/v15/zhou11b/zhou11b.pdf) 
  - Xueyuan Zhou, Mikhail Belkin. *AISTATS 2011*

- Error Analysis of Laplacian Eigenmaps for Semi-supervised Learning. 
  [[pdf]](http://proceedings.mlr.press/v15/zhou11c/zhou11c.pdf) 
  - Xueyuan Zhou, Nathan Srebro. *AISTATS 2011*

#### 2010

- Semi-Supervised Learning via Generalized Maximum Entropy. 
  [[pdf]](http://proceedings.mlr.press/v9/erkan10a/erkan10a.pdf) 
  - Ayse Erkan, Yasemin Altun. *AISTATS 2010*



## Talks

## Thesis
- xxxxx. Author. [[pdf]]().

## Blog


