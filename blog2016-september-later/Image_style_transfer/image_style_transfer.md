# 图像风格转换(Image style transfer)

图像风格转换是最近新兴起的一种基于深度学习的技术，它的出现一方面是占了卷积神经网络的天时，卷积神经网络所带来的对图像特征的高层特征的抽取使得风格和内容的分离成为了可能。另一方面则可能是作者的灵感，内容的表示是卷积神经网络所擅长，但风格却不是，如何保持内容而转换风格则是本文所要讲述的。

本篇属于论文阅读笔记系列。论文即[1].

# 引入

风格转换属于纹理转换问题，纹理转换问题在之前采用的是一些非参方法，通过一些专有的固定的方法来渲染。

传统的方法的问题在于只能提取底层特征而非高层抽象特征。随着CNN的日渐成熟，终于，这个领域被渗透了进来。

> 最近的很多应用型的研究成果都是将CNN渗透进各个领域，从而，在普遍意义上完成一次技术的升级。


# 方法

可以进行风格转换的基本就是将内容和风格区分开来，接下来我们来看CNN如何做到这一点。

## 内容提取

和之前类似，内容就是采用CNN的某一层或者某几层来表示，一般来说，层级越高，表示就越抽象。这里，需要有几个形式化的表达：

- M<sub>l</sub>: 第l层的feature map的大小
- N<sub>l</sub>: 第l层的filter的数目
- F<sup>l</sup>: 图像在第l层的特征表示，是一个矩阵，矩阵大小为M<sub>l</sub> * N<sub>l</sub>.
- F<sup>l</sup><sub>ij</sub>: 第l层第i个filter上位置j处的激活值。
- p: 原始内容图片
- x: 生成图片
- P<sup>l</sup>: 原始图片在CNN中第l层的表示
- F<sup>l</sup>: 生成图片在CNN中第l层的表示

因而，我们就得到了内容的loss。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/1.png)

求导即为：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/2.png)

有了这个公式之后要怎么做呢？ 使用现在公布的训练好的某些CNN网络，随机初始化一个输入图片大小的噪声图像x，然后保持CNN参数不变，将原始图片P和x输入进网络，然后对x求导，这样，x就会在内容上越来越趋近于P。

## 风格提取

而风格的转换则是这篇论文的神来之笔，论文使用相关矩阵来表示图像的风格。当然，风格的抽取仍然是以层为单位的。

- a: 初始风格图片
- A<sup>l</sup>: 风格图片某一层的风格特征表示。
- G<sup>l</sup>: 生成图片某一层的风格特征表示，大小为N<sub>l</sub> * N<sub>l</sub>

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/3.png)

其中，G<sup>l</sup><sub>ij</sub>的值是l层第i个feature map和第j个feature map的内积。

从而，我们得到了风格损失函数。

单独某层的损失函数：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/4.png)

各层综合的损失函数：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/5.png)

求偏导：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/6.png)

与内容表示类似，如果我们用随机初始化的x，保持CNN参数不变，将风格图片A和x输入进网络，然后对x求导，x就会在风格上趋近于A。

## 内容重建与风格重建

不考虑风格转换，只单独的考虑内容或者风格，可以看到如图所示：

图的上半部分是风格重建，由图可见，越用高层的特征，风格重建的就越粗粒度化。下半部分是内容重建，由图可见，越是底层的特征，重建的效果就越精细，越不容易变形。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/7.png)

## 风格转换

有了内容与风格，风格转换就呼之欲出了，即两种loss的加权。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/9.png)

也可如图示：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/8.png)

即同时将三张图片(a, p, x)输入进三个相同的网络，对a求出风格特征，对p求出内容特征，然后对x求导，这样，得到的x就有a的风格和p的内容。

# 实验

实验使用的是训练好的19层VGG，并通过调整权重使得每一层的激活值得均值为0。权重的调整并不会影响VGG的输出。在试验中没有使用全连接层。

实验调整了一些参数，相对其他论文而言，本论文的参数其实并不多，有：

- loss加权的权重之比
- 层级的选择
- 初始化的方法。

## loss权重之比

比例越大，内容就越强势。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/14.png)

## 层级的选择

固定住风格的层级，变动内容的层级，可以看到，内容层级越低，结果图片中的内容就越明显。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/15.png)

## 初始化方法的选择

- A: 从内容图片初始化
- B: 从风格图片初始化
- C: 随机初始化

可以看到，初始化的不同似乎对最后结果影响不大。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/16.png)

# 效果

一张图片对应到各种风格：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/10.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/11.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/12.png)

照片风格转换：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/Image_style_transfer/13.png)

# 讨论

- 速度，因为每张图片的生成都要求导很多遍，因而高清图片的生成非常慢。
- 会引入噪声，在风格转换上不明显，但风格和内容都是照片的情况下，就变得非常明显了。但这个问题估计可以很容易解决。
- 风格转换的边界非常不明显，人类也无法量化一张图片中哪一些属于风格，哪一些属于内容。
- 风格转换的成功为生物学中人类视觉原理的研究提供了一条可以切入的点。

> 最后一个优势我觉得充分体现作者的视野。

# 思考

说了那么多，为何相关矩阵可以提取风格？我百思不得其解。

- 直观上看，风格肯定是一种遍布整个图片的共性，而相关性矩阵，我认为正是把这些共性抽象的加以提取。在这个思路下，或许可以探讨其他可以提取共性的方式。比如
	- 多项式方式
	- 两两相乘变成三三相乘。
- 是不是可以通过控制给相关性矩阵加mask的方式来探讨各种feature map的真实作用。




# 参考文献

[1]. Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.
