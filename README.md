# SimplePFNN

本项目为北京大学角色动画与物理仿真课程project

试图实现"基于机器学习的kinematics动画"，使用简化版本的PFNN实现（原论文模型删除地形处理部分）

**项目失败**，并未达到预期效果

## 运行

首先运行

```
python preprocess.py
```

会将```motion_material```中的动画数据预处理，存入```processed_data.npz```中

然后运行

```
python train.py
```

训练模型，会将模型存入```checkpoint```文件夹中

运行项目：

```
python task1_project.py
```

支持键盘或PS5手柄
