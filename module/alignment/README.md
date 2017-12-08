## Align 2D

给出总的优化形式：
$$
\text{min}\sum_{\mathbf x} ||f(\mathbf x;\mathbf p)||^2\tag{0.0.1}
$$
这里的$f(\mathbf x;\mathbf p)$为误差函数。在本文中为图像残差。

### 1. Inverse Compositional

#### 1.1 基本模型

首先是最基本的模型$I(\mathbf x;\mathbf d)=I(\mathbf x+\mathbf d) $，给出误差函数如下，找到位移 $\mathbf d=(d_x, d_y)^T$ 使得$(0.0.1)$式子最小。
$$
f(\mathbf x;\mathbf d) = I(\mathbf x;\mathbf d) - T(\mathbf x)　\tag{1.1.1}
$$

用逆向组合法，在每次更新步骤中，我们更新一个增量$\delta\mathbf d​$，把$(0.0.1)​$式子写作：
$$
\text{min}\sum_{\mathbf x} ||I(\mathbf x;\mathbf d) - T(\mathbf x; \delta\mathbf d )||^2　\tag{1.1.2}\\
\text{update:}\quad\mathbf d \leftarrow \mathbf d-\delta\mathbf d
$$
在$\delta\mathbf d=\mathbf 0$处，对$(1.1.2)$做一阶泰勒展开，
$$
\text{min}\sum_{\mathbf x} ||I(\mathbf x;\mathbf d) - T(\mathbf x;\mathbf 0  )-\frac{\partial T(\mathbf u)}{\partial \mathbf u}\delta\mathbf d||^2 \tag{1.1.3}
$$
然后对$\delta\mathbf d$做偏导，并且令式子为$0$，则有：
$$
\delta\mathbf d = (\mathbf J^T\mathbf J)^{-1}\mathbf J[I(\mathbf x;\mathbf d)-T(\mathbf x;\mathbf 0)] \tag{1.1.4}\\
\text{with}\quad \mathbf J_i = \frac{\partial T(\mathbf x;\mathbf u)}{\partial \mathbf u}\bigg|_{\mathbf u=\mathbf 0}
$$
这里的$\mathbf J = (\mathbf J_0, \mathbf J_1,...,\mathbf J_i,...)^T$，是$N\times2$的矩阵，$N$为计算残差的像素点$\mathbf x$的个数。其实式子$(0.0.1)$可以看做是求$N\times1$的残差向量求二范数最小化。

最后，我们计算得到雅克比：
$$
\mathbf J_i = (g_x, g_y) \tag{1.1.5}
$$
这里的$g_x$和$g_y$分别是第$i$个点$\mathbf x$在模板图像$T$上的梯度。通过$(1.1.2)$的更新以及$(1.1.4)$的增量计算，不断迭代就可以求得参数$\mathbf d$。

**注意：**这里的雅克比表示是对模板图像求的，当然也可以记作对整个残差$I(\mathbf x+\mathbf d) - T(\mathbf x + \delta\mathbf d )$的雅克比，但是那样的表示方式会和现在的差一个负号（这里残差是当前图像对模板图像做差）。

#### 1.2 全局亮度偏差模型

假设当前图像相对参考帧有关全局的亮度增量$i$，我们给出模型$I(\mathbf x;\mathbf p)=I(\mathbf x+\mathbf d) +i $，这里的$\mathbf p = (dx, dy, i)^T$。我们可以把式子$(1.1.1)$改变为
$$
f(\mathbf x) =  I(\mathbf x; \mathbf p)- T(\mathbf x)　　\tag{1.2.1}
$$

同样，用逆向组合法来表示：
$$
\text{min}\sum_{\mathbf x} ||I(\mathbf x;\mathbf p)- T(\mathbf x;\delta\mathbf p)||^2　\tag{1.2.2}\\
\text{update:}\quad\mathbf p \leftarrow \mathbf p-\delta\mathbf p
$$
在$\delta\mathbf p=(0,0,0)^T​$处，用一阶泰勒展开：
$$
\text{min}\sum_{\mathbf x} ||I(\mathbf x;\mathbf p) - T(\mathbf x;\mathbf 0  )-\frac{\partial T(\mathbf x;\mathbf q)}{\partial \mathbf q}\delta\mathbf p||^2 \tag{1.2.3}
$$
对$\delta\mathbf p$求偏导数，最后化为：
$$
\delta\mathbf p = (\mathbf J^T\mathbf J)^{-1}\mathbf J[I(\mathbf x;\mathbf q)  - T(\mathbf x;\mathbf 0)]\tag{1.2.4}\\
\text{with}\quad \mathbf J_i = \frac{\partial T(\mathbf x;\mathbf q)}{\partial \mathbf q}\bigg|_{\mathbf q=\mathbf 0}
$$
然后求雅克比：
$$
\mathbf J_i = (g_x,g_y,1)\tag{1.2.5}
$$
与`1.1`中一样，这里的$g_x$和$g_y$分别是第$i$个点$\mathbf x$在模板图像$T$上的梯度。



### 2. Efficient Second-order Minimization

#### 2.1 全局亮度偏差模型

这里直接使用有一个全局亮度偏差的模型$\mathbf p = (d_x,d_y,i)^T$，我们有：
$$
f(\mathbf x;\mathbf p) =  I(\mathbf x; \mathbf p)- T(\mathbf x)　\tag{2.1.1}
$$
我们看当前帧的亮度向量$\mathbf I(\mathbf p)$，同样是用增量的形式，对$\mathbf I(\mathbf p)$做二阶泰勒展开：
$$
\mathbf I(\mathbf p+\delta\mathbf p) = \mathbf I(\mathbf p ) + \mathbf J(\mathbf p)\delta\mathbf p + \frac{1}{2}\mathbf H(\mathbf p)\delta\mathbf p^2 \tag{2.1.2}
$$
再对$\mathbf J(\mathbf p+\delta\mathbf p)$做一阶泰勒展开：
$$
\mathbf J(\mathbf p+\delta\mathbf p) = \mathbf J(\mathbf p) + \mathbf H(\mathbf p)\delta\mathbf p \tag{2.1.3}
$$
把$(2.1.3)$带入$(2.1.2)$得：
$$
\mathbf I(\mathbf p+\delta\mathbf p) = \mathbf I(\mathbf p ) + \frac{1}{2}(\mathbf J(\mathbf p)+\mathbf J(\mathbf p+\delta\mathbf p))\delta\mathbf p  \tag{2.1.4}
$$
当参数$\mathbf p$接近最优的时候，我们可以近似为$\mathbf I(\mathbf p + \delta\mathbf p)\equiv T(\mathbf 0)$。则式子$(2.1.4)$可以表示为：
$$
\mathbf F(\mathbf p) \equiv \mathbf I(\mathbf p ) -\mathbf T(\mathbf 0) = -\frac{1}{2}(\mathbf J(\mathbf p)+\mathbf J(\mathbf p+\delta\mathbf p))\delta\mathbf p  \tag{2.1.5}
$$
这里的$\mathbf J(\mathbf p)$是图像$I$在参数$(\mathbf p)$处求的雅克比，而$\mathbf J(\mathbf p+\delta\mathbf p)$则认为是在图像$T$在参数$\mathbf 0$位置求的雅克比。有：
$$
\mathbf J(\mathbf p) = (I_x, I_y,1)  \tag{2.1.6}\\
\mathbf J(\mathbf p+\delta\mathbf p) = (T_x, T_y,1)
$$
这里的$I_x$，$I_y$是图像$I$在参数$\mathbf p$位置处的梯度，$T_x$，$T_y$是图像$T$在参数$\mathbf 0$位置（原图像）的梯度。