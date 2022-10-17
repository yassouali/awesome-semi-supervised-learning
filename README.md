
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

### Image Classification: [Here](img_classification.md)

### Semantic and Instance Segmentation: [Here](img_segmentation.md)

### Object Detection: [Here](obj_detection.md)

### Other tasks: [Here](cv_other_tasks.md)


## NLP : [Here](nlp.md)


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
