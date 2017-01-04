# 感知损失(Perceptual Losses)

本文是参考文献[1]的笔记。该论文是Li Fei-Fei名下的论文。

# 引入

最近新出的[图像风格转换](http://blog.csdn.net/stdcoutzyx/article/details/53771471)算法，虽然效果好，但对于每一张要生成的图片，都需要初始化，然后保持CNN的参数不变，反向传播更新图像，得到最后的结果。性能问题堪忧。

但是图像风格转换算法的成功，在生成图像领域，产生了一个非常重要的idea，那就是可以将卷积神经网络提取出的feature，作为目标函数的一部分，通过比较待生成的图片经过CNN的feature值与目标图片经过CNN的feature值，使得待生成的图片与目标图片在语义上更加相似(相对于Pixel级别的损失函数)。

图像风格转换算法将图片生成以生成的方式进行处理，如风格转换，是从一张噪音图（相当于白板）中得到一张结果图，具有图片A的内容和图片B的风格。而Perceptual Losses则是将生成问题看做是变换问题。即生成图像是从内容图中变化得到。

图像风格转换是针对待生成的图像进行求导，CNN的反向传播由于参数众多，是非常慢的，同样利用卷积神经网络的feature产生的loss，训练了一个神经网络，将内容图片输入进去，可以直接输出转换风格后的图像。而将低分辨率的图像输入进去，可以得到高分辨率的图像。因为只进行一次网络的前向计算，速度非常快，可以达到实时的效果。

# 架构

下面这个网络图是论文的精华所在。图中将网络分为Transform网络和Loss网络两种，在使用中，Transform网络用来对图像进行转换，它的参数是变化的，而Loss网络，则保持参数不变，Transform的结果图，风格图和内容图都通过Loss Net得到每一层的feature激活值，并以之进行Loss计算。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/1.png)

在风格转换上，输入x=y<sub>c</sub>是内容图片。而在图片高清化上，x是低分辨率图片，内容图片是高分辨率图片，风格图片未曾使用。

# 网络细节

网络细节的设计大体遵循DCGAN中的设计思路：

- 不使用pooling层，而是使用strided和fractionally strided卷积来做downsampling和upsampling，
- 使用了五个residual blocks
- 除了输出层之外的所有的非residual blocks后面都跟着spatial batch normalization和ReLU的非线性激活函数。
- 输出层使用一个scaled tanh来保证输出值在[0, 255]内。
- 第一个和最后一个卷积层使用9×9的核，其他卷积层使用3×3的核。

确切的网络参数值请参考文献2。

## 输入输出

- 对于风格转换来说，输入和输出的大小都是256×256×3。
- 对于图片清晰化来说，输出是288×288×3，而输入是288/f×288/f×3，f是压缩比，因为Transform Net是全卷积的，所以可以支持任意的尺寸。

## Downsampling and Upsampling

- 对于图片清晰化来说，当upsampling factor是f的时候，使用后面接着log<sub>2</sub>f个stride为1/2卷积层的residual blocks.
	- fractionally-strided卷积允许网络自己学习一个upsampling函数出来。
- 对于风格转换来说，使用2个stride=2的卷积层来做downsample，每个卷积层后面跟着若干个residual blocks。然后跟着两个stride=1/2的卷积层来做upsample。虽然输入和输出相同，但是这样有两个优点：
	- 提高性能，减少了参数
	- 大视野，风格转换会导致物体变形，因而，结果图像中的每个像素对应着的初始图像中的视野越大越好。

## Residual Connections

残差连接可以帮助网络学习到identify function，而生成模型也要求结果图像和生成图像共享某些结构，因而，残差连接对生成模型正好对应得上。

# 损失函数

同[图像风格转换](http://blog.csdn.net/stdcoutzyx/article/details/53771471)算法类似，论文定义了两种损失函数。其中，损失网络都使用在ImageNet上训练好的VGG net，使用φ来表示损失网络。

## Feature Reconstruction Loss

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/2.png)

- j表示网络的第j层。
- C<sub>j</sub>H<sub>j</sub>W<sub>j</sub>表示第j层的feature_map的size

使用不同层的重建效果如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/5.png)

## Style Reconstruction Loss

对于风格重建的损失函数，首先要先计算Gram矩阵，

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/3.png)

产生的feature_map的大小为C<sub>j</sub>H<sub>j</sub>W<sub>j</sub>，可以看成是C<sub>j</sub>个特征，这些特征两两之间的内积的计算方式如上。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/4.png)

两张图片，在loss网络的每一层都求出Gram矩阵，然后对应层之间计算欧式距离，最后将不同层的欧氏距离相加，得到最后的风格损失。

不同层的风格重建效果如下：
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/6.png)

## Simple Loss Function

- Pixel Loss，像素级的欧氏距离。
- Total Variation Regularization，是之前feature inversion和super resolution工作中使用的损失，具体还需参考论文的参考论文[6,20,48,49]

# Loss对比

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/7.png)

在图像风格转换任务上，针对不同分辨率的图像，Loss值在Perceptual Loss(ours)和[图像风格转换](http://blog.csdn.net/stdcoutzyx/article/details/53771471)([10])以及内容图片上的。

可以看到，使用Perceptual Loss相当于原始算法迭代50到100次。

而就时间来看：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/8.png)

可以提升几百倍，在GPU上0.0015s可以达到相当的效果，在CPU上更具实用性。

# 效果图

## 风格转换

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/9.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/10.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/11.png)

虽然风格转换是在256的图片上训练的，但也可以应用到其他size上，比如512的

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/12.png)

## 图片超清

4倍清晰度提升：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/13.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/14.png)

8倍清晰度提升：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/perceptual_losses/15.png)

# 总结

贡献当然就是图像风格转换的实用化：

- 速度三个量级的提升。
- fully convolutional network可以应用于各种各样的尺寸。


# 参考文献

1. Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
2. Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material