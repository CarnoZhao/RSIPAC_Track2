## RSIPAC Track2

>  CarnoZhao队 初赛审核材料

### 1. 核心代码样例展示

以Pyramid-Vision-Transformer为基础的孪生网络

```python
class MMSegSiamese(MMSegModel):
    def extract_feat_siamese(self, img):
        x0 = self.extract_feat(img[0])
        x1 = self.extract_feat(img[1])
        return [torch.abs(x0[i] - x1[i]) for i in range(len(x0))]

    def forward(self, img):
        x = self.extract_feat_siamese(img)
        out = self.decode_head.forward_test(x)
        out = resize(
            input=out,
            size=img[0].shape[2:],
            mode='bilinear',
            align_corners=self.decode_head.align_corners)
        if self.with_aux_head and self.training:
            aux_out = self.aux_head.forward_test(x)
            aux_out = resize(
                input=aux_out,
                size=img[0].shape[2:],
                mode='bilinear',
                align_corners=self.aux_head.align_corners)
            return out, aux_out
        return out
```



### 2. 技术说明

#### 2.1 算法思路

本方案整体思路是以语义分割任务作为基础，再额外增加后处理流程来适应实例分割任务。对变化检测任务，本方案采用孪生网络架构，以Pyramid-Vision-Transformer作为孪生编码器，以Domain-Adaption-Transformer作为解码器。

#### 2.2 亮点解读

本方案不直接采用实例分割框架，而是转化为语义分割任务。一方面，语义分割相对来说训练所需算力更小，所需显存更小，推理计算速度更快，另一方面，语义分割任务能更好的适应形状多样的变化检测任务，可以在不改变整体框架的基础上，对后处理阶段做个性化设置。

#### 2.3 建模算力

硬件配置：

```bash
GPU = 2x TITAN RTX 24G
CPU = 16x CPU
RAM = 40G RAM
```

软件配置：

```bash
cuda = 10.2
pytorch = 1.10.0
```

训练时间：

```bash
~2h/model
```

#### 2.4 涨分点

- 采用Transformer编-解码器结构，而非CNN
- 采用孪生网络结构
- 采用5折交叉验证和Test-Time-Augmentation

### 3. 解题思路

- 初步尝试实例分割和语义分割任务，发现语义分割任务在算力需求更小的情况下能达到更好的检测效果。
- 尝试各类网络结构，发现基于Transformer的PVT和Segformer结构在参数量较小的情况下效果优于CNN类的ConvNext和EfficientNet。
- 尝试模型融合，发现单个模型测试效果不稳定，因此采用5折交叉融合和Test-Time-Augmentation

### 4. 运行环境

#### 4.1 配置环境

```bash
# 软硬件环境
echo "支持cuda-10.2的显卡以及cuda-10.2"
# conda环境
conda create -n torch pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
# pip环境
pip install \
	scikit-learn opencv-python \
	pandas pycocotools tqdm \
	matplotlib jupyter omegaconf \
	pytorch_lightning==1.5.10 \
	lightning-bolts segmentation_models_pytorch \
	transformers albumentations timm==0.5.4 \
	torchsampler rich==12.6.0
```

#### 4.2 运行办法

```bash
# 进入附件代码文件夹
cd code

# 准备数据
ln /path/to/data ./data -s
# 数据结构应该如下：
# .
# └── data
#     ├── test
#     │   ├── A
#     │   ├── B
#     │   └── instances_test.json
#     └── train
#         ├── A
#         ├── B
#         └── label

# 设定交叉验证划分
cp -r ./weights/splits ./data/train

# 训练
config=sia_pvt_daf
for f in {0..4}; do
	sed -i -E -e "s/((fold|version|load_from).*)[0-4]([^0-9]*)$/\1$f\3/g" ./configs/${config}.yaml
	python ./Solver.py --config ./configs/${config}.yaml --gpus 2,3
done

# 预测，生成结果在./results.zip
python Submission.py
```

### 5. 联系方式

微信：18810903806