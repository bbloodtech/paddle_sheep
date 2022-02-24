# 写在前面
+ 本项目所有代码基于paddle的notebook（AI Stidio经典版）运行。
# 项目背景
+ 飞桨领航团AI达人创造营
# 数据集
+ [绵羊品种分类](https://aistudio.baidu.com/aistudio/datasetdetail/108109)
# 模型介绍
```python
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV2(num_classes=num_classes)
```
# 模型训练
```python
model.train(num_epochs=100,
            train_dataset=train_dataset,
            train_batch_size=64,
            eval_dataset=val_dataset,
            lr_decay_epochs=[30, 60, 90],
            save_interval_epochs=1,
            pretrain_weights='IMAGENET',
            learning_rate=0.0003,
            save_dir='output/mobilenetv2',
            use_vdl=True)
```
# 结果验证
```python
test_transforms = T.Compose(
    [T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()])

model = pdx.load_model('output/mobilenetv2/best_model')
image_name = 'work/SheepFaceImages/Poll_Dorset/000363_P.jpg'
result = model.predict(image_name,test_transforms)
print("Predict Result:", result)
```
# 总结
+ paddle作为一款国产深度学习框架，对比TF、pytouch等其他框架来说，资料相对好找，作为入门Deep Learning是一个很好的选择。
# 个人主页
+ [CSDN](https://blog.csdn.net/bblood307?type=blog)
+ [github](https://github.com/bbloodtech)
+ [gitee](https://gitee.com/bbloodtech)
