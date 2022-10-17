
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

and adding them to the corresponding markdown file in `files/`

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

- [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170). Avital Oliver, Augustus Odena, Colin Raffel, Ekin D. Cubuk, Ian J. Goodfellow. *NeurIPS 2018*
- [Semi-Supervised Learning Literature Survey](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf). Xiaojin Zhu. *2008*
- [An Overview of Deep Semi-Supervised Learning](https://arxiv.org/abs/2006.05278). Yassine Ouali, Céline Hudelot, Myriam Tami. *2020*
- [A survey on semi-supervised learning](https://link.springer.com/content/pdf/10.1007/s10994-019-05855-6.pdf). Jesper E Van Engelen, Holger H Hoos. *2020*
- [A Survey on Deep Semi-Supervised Learning](https://arxiv.org/pdf/2103.00550.pdf). Xiangli Yang, Zixing Song, Irwin King. *2021*

## Computer Vision

- Image Classification: [list of papers here](files/img_classification.md)
- Semantic and Instance Segmentation: [list of papers here](files/img_segmentation.md)
- Object Detection: [list of papers here](files/obj_detection.md)
- Other tasks: [list of papers here](files/cv_other_tasks.md)

Note that for Image and Object segmentation tasks, we also include weakly-supervised
learning methods, that uses weak labels (eg, image classes) for detection and segmentation.

## NLP
#### [list of papers here](files/nlp.md)

## Generative Models & Tasks
#### [list of papers here](files/generative_models.md)

## Graph Based SSL
#### [list of papers here](files/graph_ssl.md)

## Theory
#### [list of papers here](files/theory.md)

## Reinforcement Learning, Meta-Learning & Robotics
#### [list of papers here](files/reinforcement_learning.md)

## Regression
#### [list of papers here](files/regression.md)

## Other
#### [list of papers here](files/other_papers.md)


## Talks
- Semi-Supervised Learning and Unsupervised Distribution Alignment. [[youtube]](https://www.youtube.com/watch?v=PXOhi6m09bA). *CS294-158-SP20 UC Berkeley.* 
- Semi-Supervised Learning and Unsupervised Distribution Alignment. [[youtube]](https://www.youtube.com/watch?v=j_-JaMPnhr0). *Pydata, Andreas Merentitis, Carmine Paolino, Vaibhav Singh.*
- Overview of Unsupervised & Semi-supervised learning. [[youtube]](https://www.youtube.com/watch?v=tnpXLK_AS_U). *AISC, Shazia Akbar.* 
- Semi-Supervised Learning. [[youtube]](https://www.youtube.com/watch?v=OMRlnKupsXM) [[slides]](https://www.cs.cmu.edu/%7Etom/10701_sp11/slides/LabUnlab-3-17-2011.pdf). *CMU Machine Learning 10-701, Tom M. Mitchell.* 

## Thesis
- Fundamental limitations of semi-supervised learnin. *Tyler Tian Lu.* [[pdf]](https://uwspace.uwaterloo.ca/bitstream/handle/10012/4387/lumastersthesis_electronic.pdf).
- Semi-Supervised Learning with Graphs. *Xiaojin Zhu.* [[pdf]](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf).
- Semi-Supervised Learning for Natural Language. *Percy Liang.* [[pdf]](https://www-cs.stanford.edu/~pliang/papers/meng-thesis.pdf).

## Blogs
- Learning with not Enough Data Part 1: Semi-Supervised Learning. *Lilian Weng.* [[link]](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/).
- An overview of proxy-label approaches for semi-supervised learning. *Sebastian Ruder.* [[link]](https://ruder.io/semi-supervised/index.html).
- The Illustrated FixMatch for Semi-Supervised Learning. *Amit Chaudhary.* [[link]](https://amitness.com/2020/03/fixmatch-semi-supervised/)
- An Overview of Deep Semi-Supervised Learning. *Yassine Ouali* [[link]](https://yassouali.github.io/ml-blog/deep-semi-supervised/)
- Semi-Supervised Learning in Computer Vision. *Amit Chaudhary* [[link]](https://amitness.com/2020/07/semi-supervised-learning/)
