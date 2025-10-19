# House Prices - Advanced Regression Techniques  

url：<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>  

## 环境（重要）  

本项目使用Python3.13.5与CUDA13.0
本项目包含的环境表严重冗余，请不要直接安装，以下给出需要的主要环境  

- numpy==2.2.6  
- pandas==2.3.2  
- torch==2.10.0.dev20250924+cu130  

## 项目思路  

### 1.数据处理  

此部分代码位于./code/data_read.py  

本项目题目提供的数据有./data目录下的四个文件，其中两个文件分别是训练数据集和测试数据集，另一个是提交格式实例，最重要的一个是数据集内容文档  
因此首先根据此文档分析数据内容，将需要处理的数据类别进行统计分类，例如：类别数据、连续数值数据等，并对他们使用不同的处理方法  

在此项目中选择使用pandas库进行数据读取、分析、处理，使用了模块化编写思路，将处理流程封装到函数体中，并且构建对外调用api，方便训练体直接获取所需要的训练集数据和测试集数据  

具体处理思路和方法请阅读./code/data_read.py中的代码和注释  

### 2.训练实现  

此部分代码位于./code/train_v1.py  

未完待续...

## 为什么使用Pytorch  

作为训练项目，使用Pytorch训练完整的深度学习流程  

## About  

本项目部分使用AI，包括但不限于利用AI分析错误输出、修改错误代码等  
不对本项目代码做出纯手搓保证  
此项目仅用于本人学习  
和海创团队的二轮考核  
