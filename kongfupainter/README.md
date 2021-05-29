### kongfupainter
**CycleGAN复现**

运行环境：
Ubuntu16.04
pytorch1.5.1

通过复现代码学习了以下内容：
- 数据集加载
- 网络模型定义
- 学习率更新
- 网络参数保存
- 断点训练
- 设置使用特定GPU训练
- 使用visdom进行可视化训练

下载数据集：

```
链接：https://pan.baidu.com/s/1SD5cCYfe7GMGXM77I7gCiw 
提取码：oplk 
```

将其解压缩到根目录下。

运行程序：

训练
```shell script
bash ./train.sh
```

测试
```shell script
bash ./test.sh
```