<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

---

## Key essence of machine learning

![][1]

> 适合使用机器学习的三个关键要素：
> - 存在某些潜在的模式或者规则可以学习
> - 没有具体的定义或者规则，来编写程序，或者不容易给出完备的定义或者规则
> - 有大量的可以学习潜在模式或者规则的数据用来学习

【分析】：
- (i) 判断一个数是否为素数。不需要使用机器学习的方法来做，因为已经有明确的规则可以进行判断：若一个整数只能被1和它本身整除，则为素数，否则不是素数；
- (ii) 检测信用卡消费中的潜在欺诈行为。属于使用机器学习中异常点检测问题；
- (iii) 确定物体下落所需的时间。不需要使用机器学习，因为有物理公式可以计算；
- (iv) 确定繁忙路口的交通信号灯的最佳周期。可以统计不同路口交通信号灯的周期和拥堵程度的数据，然后使用回归分析，学习信号灯周期与拥堵程度的之间的关系；
- (v) 确定建议进行特定医疗检查的年龄。可以对不同年龄人群生病情况进行统计，进行聚类分析，年龄相似患病相似的人群聚为一类，就可以向这个年龄附近的人推荐相应的医疗检查。

<!-- more -->

## Types of learning

![][2]

> 机器学习分类:
> - **监督式学习**：可以从训练集中学到或建立一个模式（函数 / learning model），并依此模式推测新的实例。训练集是由一系列特征（通常是向量）和预期输出所组成。函数的输出可以是一个连续的值（称为回归分析），或是预测一个分类标签（称为分类）
> - **非监督式学习**：不需要人力来输入标签。它是监督式学习和强化学习等策略之外的一种选择。在监督式学习中，典型的任务是分类和回归分析，且需要人工预先做好标注。一个常见的非监督式学习是数据聚类。
> - **主动学习**：主动学习是半监督学习的一种特殊情况，其中学习算法能够交互地查询用户（或某些其他信息源），以在新的数据点获得所需的输出。
> - **强化学习**：强调如何基于环境而行动，以取得最大化的预期利益。其灵感来源于心理学中的行为主义理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。环境通常被规范为马可夫决策过程（MDP），所以许多强化学习算法在这种情况下使用动态规划技巧。强化学习和标准的监督式学习之间的区别在于，它并不需要出现正确的输入/输出对，也不需要精确校正次优化的行为。强化学习更加专注于在线规划，需要在探索（在未知的领域）和遵从（现有知识）之间找到平衡。

【分析】：
2.采用不同的策略下棋，以结果作为回馈，学习更好的下棋策略，属于强化学习

3.不给定主题对书籍进行分类，属于无监督学习，可以统计书籍的关键词等作为特征，使用聚类分析

4.人脸识别，训练数据为1000张标注为 `+1` 的人脸图像和10000张标注为 `-1` 的非人脸图像，属于监督式学习

5.有选择性的安排实验，来快速的评估抗癌药物的效果，类似主动学习，对每一种药物，由于不知道实际效果，于是进行小白鼠实验，来获得输出

## Off-Training-Set error

![][3]

【分析】：$f$ 对任何输入都输出 $+1$，$g$ 对偶数下标的样本输出 $-1$，所以只需要判断 $N+1$ 到 $N+L$ 中有多少个偶数即可，可以分不同情况讨论：
- 若 $L$ 为偶数，则测试集样本的下标为 $L$ 个连续的整数，所以肯定有 ${L\over 2}$ 个偶数；
- 若 $L$ 为奇数且 $N+1$ 也为奇数，则奇数占多数，故有 ${L-1\over 2}$ 个偶数；
- 若 $L$ 为奇数且 $N+1$ 为偶数，则偶数占多数，故有 ${L+1\over 2}$ 个偶数。
最后可整理为一个式子，即 ${1\over L}\times(\lfloor{N+L\over 2}\rfloor - \lfloor{N\over2}\rfloor)$

![][4]

【分析】：如果目标函数 $f$ 对任意的输入样本 $(x_n, y_n) \in \mathcal{D}$ 都有 $f(x_n)=y_n$，则称 $f$ 可以无噪声生成数据集 $\mathcal{D}$，现在题目问最多有多少种不同的 $f$ 可以无噪声生成 $\mathcal{D}$ ？因为是二分类问题，所以对于每一个特定的样本，$f$ 只有两种分法，要么将其分为正的，要么分成负的，现在求 $f$ 有多少种可能性，就是求在测试集上有多少种不同的分法，测试集有 $L$ 个样本，所以就有 $2^L$ 种。

![][5]

【分析】：题目意思是假设上一个问题中的那些 $f$ 都是等可能性的，请证明或证否两个确定性算法算出的 $g$ 对于 $f$ 产生的数据集的线上错误率的期望是相等的。个人理解：因为所有可能的 $f$ 都是等可能性的，所以在测试集上的数据完全无任何规律或潜在的模式，所以不管用什么算法来算出一个 $g$，虽然可以拟合训练集，但在测试集上都是相当于乱猜的，所以线上错误率的期望应该是相等的，大概 $0.5$（个人猜测^_^）

## Bin model

![][6]

【分析】：从罐子中随机抓取一个小球为橘色的概率为 $u$，现从中随机抓取 $10$ 个小球，则橘色小球个数为 $10v$ 的概率是多少

9.$u={1\over2}$，则随机抓取的$10$个小球中有$5$个是橘色的概率：
$$\dbinom{10}{5}\times({1\over 2})^5\times({1\over 2})^5={63\over 256}$$

10.$u={9\over10}$，随机抓取的$10$个小球中有$9$个是橘色的概率：
$$\dbinom{10}{9}\times({9\over 10})^9\times({1\over 10})^1=({9\over 10})^9$$

11.$u={9\over10}$，随机抓取的$10$个小球中橘色小球个数不大于$1$的概率：
$$\dbinom{10}{1}\times({9\over 10})^1\times({1\over 10})^9 + ({1\over10})^{10} = {91\over10^{10}}$$

12.由 $u=0.9,v\leq0.1$ 可得 $|u-v|\geq0.8$, 即$\epsilon=0.8$，由 $Hoeffding$ 不等式 $P(|u-v|\geq\epsilon) \leq 2e^{-2N\epsilon^2}$，代入计算得 $5.5215\times 10^{-6}$

## Multiple bins

![][7]

【分析】：
13.AC两种骰子的$1$是橘色的，因此随机抽取到$1$个骰子是橘色1的概率为 $0.5$，所以抽取到$5$个橘色1的概率为 $({1\over2})^5={1\over32}$

14.分析可知，AB两种骰子涂色方式完全相反，CD两种骰子也是完全相反，所以要想抽到的骰子中存在某一点数是都是同一个颜色，那么就不能有两种以上的骰子的组合。最简单的，只抽到一种骰子，那肯定符合题意，抽到两种骰子，那么不能是AB，也不能是CD，这样一来可能的组合为：A、B、C、D、AC、AD、BC、BD，它们的概率都可以求出来，前四种的概率都相等，为 $({1\over4})^5$，后四种的也都相等，为 $( \dbinom{5}{1} + \dbinom{5}{2} + \dbinom{5}{3} + \dbinom{5}{4} ) \times ({1\over4})^5$，所以最后的结果为：
$$4\times({1\over4})^5 + 4\times( \dbinom{5}{1} + \dbinom{5}{2} + \dbinom{5}{3} + \dbinom{5}{4} ) \times ({1\over4})^5 = {31\over64}$$

## PLA and pocket algorithm

![][9]

[[code][10]]

## Bonus
![][8]
【分析】：感知机学习算法(PLA)对于线性可分数据集收敛，证明如下：

(1) 由于训练数据是线性可分的，故存在超平面可将训练数据集完全正确分开，即存在 $w_f$，使得 $||w_f||=1$，且对任意的训练实例 $(x_n,y_n)$ 有
$$y_nw_f^Tx_n > 0$$
所以存在
$$\rho=min_n(y_nw_f^Tx_n) > 0$$
使得对任意的训练实例 $(x_n,y_n)$ 有
$$y_nw_f^Tx_n \geq \rho$$

(2) PLA 从 $w=0$ 开始，设 $w\_t$ 为第 $t-1$ 轮迭代时的权重向量，若此时实例 $(x\_n,y\_n)$ 分类错误，则进行如下更新
$$w\_t = w\_{t-1} + y\_nx\_n$$
于是有
$$w\_f^Tw\_t = w\_f^Tw\_{t-1} + y\_nw\_f^Tx\_n \geq w\_f^Tw\_{t-1} + \rho$$
由此递推可得
$$w\_f^Tw\_t \geq w\_f^Tw\_{t-1} + \rho \geq ... \geq w\_f^Tw\_0 + t\rho = t\rho$$
$$\Rightarrow ||w\_f^Tw\_t||^2 = ||w\_t||^2 \geq t^2\rho^2$$
令 $R^2 = max\_n(||x\_n||^2)$，则
$$||w\_t||^2 = ||w\_{t-1}||^2 + 2y\_nw\_{t-1}^Tx\_n + ||x\_n||^2 \leq ||w\_{t-1}||^2 + ||x\_n||^2 \leq ||w\_{t-1}||^2 + R^2$$
递推得
$$||w\_t||^2 \leq ||w\_{t-1}||^2 + R^2 \leq ... \leq ||w\_0||^2 + tR^2 = tR^2$$
所以有
$$t^2\rho^2 \leq ||w\_t||^2 \leq tR^2$$
$$\Rightarrow t \leq ({R\over \rho})^2$$
综上所述，算法迭代次数是有上界的，所以，当训练数据集线性可分时，PLA 是收敛的

现在再来回答 Problem 21 就简单了，成倍的缩小所有输入实例的 $x$，确实可以缩小 $R$，但同时也成比例的缩小了 $\rho$，而 $t$ 的上界却没变，所以这个方法不能加速 PLA

---

**【 如有错误，望不吝指正，感谢！！！邮箱：hux1aoyang@qq.com 】**


  [1]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q1.jpg
  [2]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q2-5.jpg
  [3]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q6.jpg
  [4]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q7.jpg
  [5]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q8.jpg
  [6]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q9-12.jpg
  [7]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q13-14.jpg
  [8]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q21.jpg
  [9]: http://or4kiv7u1.bkt.clouddn.com/hw1_Q15-20.jpg
  [10]: https://github.com/huxy29/learning-machine-learning/tree/master/machine-learning-foundations/hw1
