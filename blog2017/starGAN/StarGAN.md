# StarGAN-多领域图像翻译

Pix2Pix模型解决了有Pair对数据的图像翻译问题；CycleGAN解决了Unpaired数据下的图像翻译问题。但无论是Pix2Pix还是CycleGAN，都是解决了一对一的问题，即一个领域到另一个领域的转换。当有很多领域要转换了，对于每一个领域转换，都需要重新训练一个模型去解决。这样的行为太低效了。本文所介绍的StarGAN就是将多领域转换用统一框架实现的算法。

下图是StarGAN的效果，在同一种模型下，可以做多个图像翻译任务，比如更换头发颜色，更换表情，更换年龄等。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/1.png)

# 引入

如果只能训练一对一的图像翻译模型，会导致两个问题：

- 训练低效，每次训练耗时很大。
- 训练效果有限，因为一个领域转换单独训练的话就不能利用其它领域的数据来增大泛化能力。

为了解决多对多的图像翻译问题，StarGAN出现了。

# 模型框架

StarGAN，顾名思义，就是星形网络结构，在StarGAN中，生成网络G被实现成星形。如下图所示，左侧为普通的Pix2Pix模型要训练多对多模型时的做法，而右侧则是StarGAN的做法，可以看到，StarGAN仅仅需要一个G来学习所有领域对之间的转换。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/2.png)

那么，是什么让G有这样的能力呢？

# 网络结构

要想让G拥有学习多个领域转换的能力，需要对生成网络G和判别网络D做如下改动。

- 在G的输入中添加目标领域信息，即把图片翻译到哪个领域这个信息告诉生成模型。
- D除了具有判断图片是否真实的功能外，还要有判断图片属于哪个类别的能力。这样可以保证G中同样的输入图像，随着目标领域的不同生成不同的效果
- 除了上述两样以外，还需要保证图像翻译过程中图像内容要保存，只改变领域差异的那部分。图像重建可以完整这一部分，图像重建即将图像翻译从领域A翻译到领域B，再翻译回来，不会发生变化。

D的训练和G的训练如下所示。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/3.png)

# 目标函数

首先是GAN的通用函数，判断输出图像是否真实

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/4.png)

其次是类别损失，该损失被分成两个，训练D的时候，使用真实图像在原始领域进行，训练G的时候，使用生成的图像在目标领域进行。

训练D的损失：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/5.png)

训练G的损失：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/6.png)

再次则是重建函数，重建函数与CycleGAN中的正向函数类似。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/7.png)

汇总后则是

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/8.png)

# 多数据集训练

在多数据集下训练StarGAN存在一个问题，那就是数据集之间的类别可能是不相交的，但内容可能是相交的。比如CelebA数据集合RaFD数据集，前者拥有很多肤色，年龄之类的类别。而后者拥有的是表情的类别。但前者的图像很多也是有表情的，这就导致前一类的图像在后一类的标记是不可知的。

为了解决这个问题，在模型输入中加入了Mask，即如果来源于数据集B，那么将数据集A中的标记全部设为0.

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/9.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/10.png)

# 效果图

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/starGAN/11.png)

更多请参考[原始论文](https://arxiv.org/abs/1711.09020).

# Reference

- [1]. [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
- [2]. [Pix2Pix图像翻译](http://blog.csdn.net/stdcoutzyx/article/details/78820728)
- [3]. [CycleGAN-Unpaired图像翻译](http://blog.csdn.net/stdcoutzyx/article/details/78823249)

