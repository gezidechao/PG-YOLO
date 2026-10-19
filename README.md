# PG-YOLO

语义先验引导的混凝土桥梁表观病害检测方法。

本次发布提供 PG-YOLO 的核心模块、模型结构配置、训练参数参考及必要的先验输入资源。

## 1. 核心模块

本项目基于 Ultralytics YOLO 框架实现。

| 论文模块名称 | 源码位置 | 代码中的名称 |
| --- | --- | --- |
| PG-GRAM | ultralytics/nn/modules/rram.py | RRAM |
| LGC-VSSBlock | ultralytics/nn/modules/vmamba.py | VSSBlock |
| DSA-AFF | ultralytics/nn/tasks.py | MCS_AFF |

代码保留了开发阶段使用的类名。

模块注册和模型解析代码包含在本仓库的 ultralytics 源码中。

## 2. 文件说明

- ultralytics/：框架与自定义模块源码。

- configs/model_self.yaml：PG-YOLO 模型结构配置。

- configs/train_self.yaml：原训练参数参考。

- priors/self/A_hat.npy：模型使用的类别先验矩阵。

- priors/self/V.npy：模型使用的类别词向量。

- third_party/selective_scan/：Selective Scan CUDA 扩展源码。

- third_party/VMamba_LICENSE：相关第三方许可证。

- pyproject.toml：框架安装配置与依赖。

- requirements-extra.txt：自定义模块的补充依赖。

- LICENSE：原框架许可证。

先验资源不包含数据集图片及逐实例标注。

训练参数文件中的数据路径、输出路径和设备编号为原运行环境设置，

实际使用时应根据自己的环境调整。

## 3. 环境说明

自定义 CUDA 算子使用 Linux、NVIDIA GPU 和 CUDA 工具链。

使用者应准备相互兼容的 PyTorch、torchvision 和 CUDA 编译环境。

框架依赖见 pyproject.toml，补充依赖见 requirements-extra.txt。

请使用本仓库提供的框架源码，以加载自定义模块。

Selective Scan 扩展的构建入口为：

third_party/selective_scan/setup.py

编译生成的 selective_scan_cuda_oflex 扩展应放置于：

ultralytics/nn/modules/selective_scan_pkg/

## 4. 模型配置

从仓库根目录使用 configs/model_self.yaml。

其中的先验路径指向仓库内的 priors/self/。

完成依赖安装与 CUDA 扩展编译后，可使用框架接口加载模型：

```python

from ultralytics import YOLO

model = YOLO("configs/model_self.yaml")

```

该配置包含5个病害类别，编号如下：

| 编号 | 类别 |
| --- | --- |
| 0 | corrosion（水侵蚀） |
| 1 | hole（孔洞） |
| 2 | rebar（露筋） |
| 3 | spall（剥落） |
| 4 | speckle（麻面） |

## 5. 数据与实验材料

本仓库不公开提供数据集图片及相应标注下载，

也不包含实验权重、训练结果和完整实验复现材料。

数据集及相关标注可根据合理请求联系论文通信作者申请获取，

具体使用应遵守相应数据来源和授权要求。

如需进一步的实验材料，可联系作者。

## 6. 许可证与致谢

本项目基于 Ultralytics 框架开发，

保留其 AGPL-3.0 许可证及源码版权声明。

Selective Scan 相关代码来自 VMamba/Mamba 相关实现，

保留其源码声明和附带许可证。

类别词向量资源基于 GloVe。

感谢相关开源项目。
