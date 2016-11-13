# 对抗生成网络（Generative Adversarial Net)

好久没有更新博客了，但似乎我每次更新博客的时候都这么说（泪）。最近对生活有了一些新的体会，工作上面，新的环境总算是适应了，知道了如何摆正工作和生活之间的关系，如何能在有效率工作的同时还能继续做自己感兴趣的事情。感情上也终于找到了肯对我好的那个她，感觉生活动力充实了不少。心理上，我依然故我般的做那个简单的技术boy，生活态度偏理想化但可以直面现实……

突然想到这是一片技术博客，还是不多说自己的事情了，说一说甚嚣尘上的对抗网络吧。

# 引入

## Discriminative Model的繁荣发展

最近，深度学习在很多领域的突破性进展想必不用我多说了。但大家似乎发现了这样的一个现实，即深度学习取得突破性进展的地方貌似都是discriminative的模型。

> 所谓的discriminative可以简单的认为是分类问题，比如给一张图片，判断这张图片上有什么动物；再比如给定一段语音，判断这段语音所对应的文字。

在discriminative的模型上，有很多行之有效的方法，如反向传播，dropout，piecewise linear units等技术。

## Generative Model

其实，这篇论文很早之前就看了，但我对生成模型在AI里的地位一直不能特别直观的感受。最近才慢慢的理解。

从细节上来看，生成模型可以做一些无中生有的事情。比如图片的高清化，遮住图片的一部分去修复，再或者画了一幅人脸的肖像轮廓，将其渲染成栩栩如生的照片等等。

再提高一层，生成模型的终极是创造，通过发现数据里的规律来生产一些东西，这就和真正的人工智能对应起来了。想想一个人，他可以通过看，听，闻去感知这世界，这是所谓的Discriminative，他也可以说，画，想一些新的事情，这就是创造。所以，生成模型我认为是AI在识别任务发展相当成熟之后的AI发展的又一个阶段。

## 借东风

但是现在，生成模型还没有体会到深度学习的利好，在Discriminative模型上，成果如雨后春笋，但在生成模型上，却并非如此。原因如下：

- 在最大似然估计及相关策略上，很多概率计算的模拟非常难
- 将piecewise linear units用在生成模型上比较难

那么，是不是生成模型就借不了深度学习发展的东风了呢？我只能说，有的时候，不得不曲线救国。

# 对抗网络

## 基本思想

假设有一种概率分布M，它相对于我们是一个黑盒子。为了了解这个黑盒子中的东西是什么，我们构建了两个东西G和D，G是另一种我们完全知道的概率分布，D用来区分一个事件是由黑盒子中那个不知道的东西产生的还是由我们自己设的G产生的。

不断的调整G和D，直到D不能把事件区分出来为止。在调整过程中，需要：

- 优化G，使它尽可能的让D混淆。
- 优化D，使它尽可能的能区分出假冒的东西。

当D无法区分出事件的来源的时候，可以认为，G和M是一样的。从而，我们就了解到了黑盒子中的东西。

## 简单的例子说明

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/1.png)
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/2.png)

且看上面四张图a,b,c,d. 黑色的点状线代表M所产生的一些数据，红色的线代表我们自己模拟的分布G，蓝色的线代表着分类模型D。

a图表示初始状态，b图表示，保持G不动，优化D，直到分类的准确率最高。
c图表示保持D不动，优化G，直到混淆程度最高。d图表示，多次迭代后，终于使得G能够完全你和M产生的数据，从而认为，G就是M。

## 形式化

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/3.png)

将上述例子所描述的过程公式化，得到如上公式。公式中D(x)表示x属于分布M的概率，因而，优化D的时候就是让V(D,G)最大，优化G的时候就是让V(D,G)最小。

其中，x~p<sub>data</sub>(x) 表示x取自真正的分布。
z~p<sub>z</sub>(z) 表示z取自我们模拟的分布。G表示生成模型，D表示分类模型。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/4.png)

上述即是G和D的训练过程。其中在每次迭代中，梯度下降K次来训练D，然后梯度下降一次来训练G，之所以这样做，是因为D的训练是一个非常耗时的操作，且在有限的集合上，训练次数过多容易过拟合。

# 证明

这篇论文中的思想就如上所述，但是有意思的是还有两个证明来从理论上论证了对抗网络的合理性。

## 命题一

第一个证明是，当G固定的时候，D会有唯一的最优解。真实描述如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/5.png)

证明如下：

- 首先，对V(G,D)进行变换
	![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/6.png)
- 对于任意的a,b ∈ R<sup>2</sup> \ {0, 0}, 下面的式子在a/(a+b)处达到最优。
	![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/7.png)
	
得证！

### 定理一

根据证明一，可以对V(G,D)中最大化D的步骤进行变换。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/8.png)

从而得到定理

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/9.png)

直接带入p<sub>g</sub>=p<sub>data</sub>可得-log4，当入p<sub>g</sub>!=p<sub>data</sub>时，得到

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/10.png)

## 命题二

命题二原文如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/11.png)

这个定理的证明需要用到凸函数的某个似乎是很明显的定理，即，通过凸函数的上确界的次导数可以找到函数在最大值时的导数。这个理论应用到G和D中就是在G不变时，D是拥有唯一的最优值的凸函数，因而可以得到。 但因为我对凸优化理论尚不熟悉，所以没有理解透彻这个地方。

# 实验

> 早期的训练中，D可以很轻松的分辨出来G和M中不同的样本，从而会饱和，所以用logD(G(z))来代替log(1-D(G(z)),这样可以为早期的学习提供更加好的梯度。

实验就是去拟合Guassian Parzen Windown，具体细节略过。结果如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/12.png)

# 优势和劣势

优势：

- Markov链不需要了，只需要后向传播就可以了。
- 生成网络不需要直接用样本来更新了，这是一个可能存在的优势。
- 对抗网络的表达能力更强劲，而基于Markov链的模型需要分布比较模糊才能在不同的模式间混合。

劣势：

- 对于生成模型，没有直接的表达，而是由一些参数控制。
- D需要和G同步的很好才可以。

各种生成模型的对比如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GAN/13.png)



# 参考

- Ian J. Goodfellow. Generative Adversarial Nets. 
- [深度 | OpenAI Ian Goodfellow的Quora问答：高歌猛进的机器学习人生](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650718178&idx=1&sn=6144523762955325b7567f7d69a593bd&scene=1&srcid=0821xPdRwK2wIHNzgOLXqUrw&pass_ticket=uG39FkNWWjsW38Aa2v5b3cfMhixqsJ0l1XLhNr5mivWEaLyW5R1QED0uAKHOwuGw#rd)
- [ 生成式对抗网络GAN研究进展（二）——原始GAN](http://blog.csdn.net/solomon1558/article/details/52549409)