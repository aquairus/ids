#入侵检测特征分析
>基于kdd99数据集

##数据预处理
 * 挑选特征
 
	  1. TCP连接基本特征（共9种）
	  2. TCP连接的内容特征
	  3. 基于时间的网络流量统计特征
	  4. 基于主机的网络流量统计特征 
	  
 * 类型
 
 	* 离散型：
 	2 3 4 7  12 14 15 21 22
 	
 	 * 连续型：
 	rest of features

  * 洗数据
  
  	* 离散型：
  		数字化
  	* 连续型：
  		离散化

 * 样本分析  
 
 normal | total |rate|
------------ | ------------- |
97277| 494020| 0.1969|





 * 数据透视表
 * PCA
 * 开发流程
  
##预测模型&&模型检验

离散数据：

* RandomForestClassifier

	cross_val_score：0.988561251369

* GradientBoostingClassifier

	cross_val_score：0.987065361169

* AdaBoostClassifier

	cross_val_score：0.984031048032
	
连续数据：

* naive_bayes

	cross_val_score：0.939538570828
	
* LogisticRegression

	cross_val_score：0.981662747181
##检验


连续特征：
LogisticRegression

 *  roc曲线

![log_pr](./pic/log_roc)

auc=0.998036777822

 
 * pr曲线

![log_pr](./pic/log_pr)

auc=0.98742963525


离散特征：
RandomForestClassifier

 
 * roc曲线

![roc_pr](./pic/rf_roc)

auc=0.99605060404 

 *  pr曲线 

 ![rf_pr](./pic/rf_pr)


auc=0.986248939905

 * 测试集合
 
##综合
把连续数据离散化，用随机森林做预测
各项指标都在0.99以上 
##模型应用
 
 *  snort
  *  iptables

目录 | 简介 |
------------ | ------------- |
data| 数据存放|
discrete| 离散特征|
continuous| 连续特征|
final| 全部特征|

