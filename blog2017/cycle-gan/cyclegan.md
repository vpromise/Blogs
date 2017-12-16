# CycleGAN-无pair图像翻译

在[Pix2Pix]((http://blog.csdn.net/stdcoutzyx/article/details/78820728))中，输入图像数据都是成对的。但在现实生活中，两个不同领域的图像很难有成对的。莫奈的画很好，但莫奈永远也画不出21世纪的样子，所以我们不可能获得21世纪莫奈风格的图像。那么要想让21世纪的图像变成莫奈风格，就必须用到无pair数据。


![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/1.png)

在语言翻译中，常用的一种假设是Cycle一致性，即X语言翻译到Y语言在翻译回X语言，应该和初始的表达是一致的。而一言以蔽之，本文介绍的CycleGAN算法就是将这种Cycle一致性思维引入到图像翻译任务上来。

# 算法模型

本文的问题和算法框架与[Pix2Pix]((http://blog.csdn.net/stdcoutzyx/article/details/78820728))一致。如果对Pix2Pix不了解的话，出门左转到[Pix2Pix]((http://blog.csdn.net/stdcoutzyx/article/details/78820728))进行阅读。

其实之前有很多Unpaired Image Translation的算法出现。但是相对于它们，CycleGAN的优势在于：

- 不依赖基于特定任务的预定义的输入和输出的相似度计算方法。
- 不需要假设输入和输出在同一个低维embedding空间。

## 公式变量定义

- X,Y：两个领域
- {x}, {y}: 两个领域的图像集合
- 映射G: X->Y
- 映射F: Y->X
- D<sub>X</sub>: 用于区分{x}和{F(y)}
- D<sub>Y</sub>: 用于区分{y}和{G(x)}

## 损失函数

首先是GAN Loss，对于映射G来说，GAN loss是：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/2.png)

对于映射F有一个类似的公式。

其次是Cycle GAN的核心， Cycle一致性Loss。所谓的Cycle一致性，就是要保证

- 前向一致: x->G(x)->F(G(x))≈x
- 后向一致: y->F(y)->G(F(y))≈y

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/7.png)

表示成公式为：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/3.png)

最后汇总的损失函数是：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/4.png)

## Why works

为什么Cycle一致性能够起作用？

对于Unpaired问题来说，只是用普通GAN的话可以学到的模型有很多种。种类数目为领域X和领域Y之间的随机映射数目。所以只是用普通GAN损失函数无法保证输入x能够得到对应领域的y。而Cycle一致性的出现，降低了随机映射的数目。从而保证得到的输出不再是随机的。

## 训练细节

CycleGAN算法仍然是GAN算法，所以需要定义G模型和D模型。在CycleGAN中，G的模型结构与[Perceptual Loss](http://blog.csdn.net/stdcoutzyx/article/details/54025243)中一致，且使用instance normalization。

模型D则是70x70的PatchGAN中的D。这样的D在[Pix2Pix图像翻译](http://blog.csdn.net/stdcoutzyx/article/details/78820728)模型中也有使用，能够使用较少的参数和应用到更大的图像上。

为了使GAN的训练更加稳定，使用平方损失而不是Log似然。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/6.png)

为了减少震荡，使用历史生成图片而不是最新生成图片来进行D的训练。具体为缓存50张历史生成图像。

其他细节：

- λ=10
- Adam 优化算法
- learning rate前100个epoch为0.0002，在下100个epoch上线性下降。

# 实验

实验检测手法仍然是AMT和FCN-score，与[Pix2Pix图像翻译](http://blog.csdn.net/stdcoutzyx/article/details/78820728)相似。

## 对比baseline算法

- CoGAN
- Pixel Loss + GAN
- Feaure Loss + GAN
- BiGAN
- Pix2Pix(在Pair数据上训练）

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/8.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/9.png)

## Loss function实验

对比普通GAN损失，GAN+前向一致性损失，GAN+后向一致性损失，CycleGAN

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/cycle-gan/10.png)

# 应用

- Collection Style Transfer
- Object Tranfiguration
- Season Transfer
- Photo Generation From Paintings
- Photo Enhancement

更多效果图请参考[Paper](https://arxiv.org/abs/1703.10593)

# 总结

使用了普通的GAN和两个方向上的一致性损失解决了Unpair的图像翻译问题。虽然效果比Pix2Pix要差，但已经达到了能看的地步。

但Pix2Pix系列的GAN算法整体上来说，在色彩和纹理上能够达到较好的效果，在有几何学变换的时候，领域转换的效果不好。比如猫狗转换问题。

另外，Unpaired问题与Paired问题存在一个Gap，生成的图像总体上相差太远。为了解决这个混淆性，弱监督或者半监督可能会帮助提高效果。


# Reference

- [1]. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [2]. [Pix2Pix图像翻译](http://blog.csdn.net/stdcoutzyx/article/details/78820728)
- [3]. [Perceptual Loss](http://blog.csdn.net/stdcoutzyx/article/details/54025243)