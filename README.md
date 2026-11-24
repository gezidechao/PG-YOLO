# PG-YOLO

PG-YOLO 是一种语义先验引导的混凝土桥梁表观病害检测方法。本仓库提供模型源码、自建数据集实验的模型及训练配置，以及论文公开数据集实验的复现材料。

## 核心模块

PG-YOLO 基于 Ultralytics YOLO 框架实现。下表列出论文中的模块名称及其代码位置。

| 模块 | 源码位置 | 实现名称 |
| --- | --- | --- |
| `PG-GRAM` | `ultralytics/nn/modules/rram.py` | `RRAM` |
| `LGC-VSSBlock` | `ultralytics/nn/modules/vmamba.py` | `VSSBlock` |
| `DSA-AFF` | `ultralytics/nn/tasks.py` | `MCS_AFF` |

## 安装

在 Linux 的 NVIDIA GPU 环境下运行。请先安装支持 CUDA 的 PyTorch、torchvision，以及与 PyTorch 兼容的 CUDA Toolkit（含 `nvcc`）和 C++ 编译工具。以下命令在仓库根目录运行。

```bash
python -m pip install -e . -r requirements-extra.txt -r reproduce/new_ok2/requirements.txt
(cd third_party/selective_scan && python setup.py build_ext --inplace)
cp third_party/selective_scan/selective_scan_cuda_oflex*.so ultralytics/nn/modules/selective_scan_pkg/
python -c "from ultralytics import YOLO; print('PG-YOLO import OK')"
```

## 自建数据集实验配置

模型结构见 [自建集模型配置](configs/model_self.yaml)，原训练参数见 [自建集训练配置](configs/train_self.yaml)，类别先验矩阵和词向量见 [自建集先验文件](priors/self/)。使用训练配置时，请按运行环境设置数据路径、设备编号和输出目录。

自建集的类别按编号 0–4 依次为水侵蚀、孔洞、露筋、剥落和麻面。数据获取方式见论文。

安装完成后，可从仓库根目录加载模型结构。

```python
from ultralytics import YOLO

model = YOLO("configs/model_self.yaml")
```

## 公开数据集

从本仓库的 [Releases](https://github.com/gezidechao/PG-YOLO/releases) 下载 `PG-YOLO-Public-Bridge-Damage.zip`，解压到仓库根目录下的 `datasets/`。解压后，数据配置文件应位于 `datasets/new_OK2/data.yaml`。原实验划分为训练集 3,922 张、验证集 489 张和测试集 491 张。训练集用于训练，验证集用于模型选择和检测指标统计，测试集用于可视化。

数据来自 Roboflow Universe（2026 年 4 月 2 日导出，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）。本文将原复合标签中的露筋与剥落分别标注，并将剥落归入原剥落类。类别编号为 `0 mold`、`1 spall`、`2 potholes`、`3 road bleeding`、`4 rebar`。

## 公开数据集实验复现

公开集模型配置、训练参数、先验和实例级 Copy-Paste 实现均位于 `reproduce/new_ok2/`。训练从随机初始化开始，最多 300 轮，训练批量大小为 8，输入为 1024 × 1024 像素，预热 5 轮。训练用实例级 Copy-Paste，关闭 Mixup。

从仓库根目录启动训练。

```bash
python reproduce/new_ok2/run.py train --device 0
```

训练结束时自动在验证集上评估最佳权重，并输出 P、R、mAP 和 F1。首次运行的最佳权重位于 `runs/public/pg_yolo_new_OK2/weights/best.pt`。以下命令可对该权重重新验证和统计效率。

```bash
python reproduce/new_ok2/run.py val --device 0 --weights runs/public/pg_yolo_new_OK2/weights/best.pt
python reproduce/new_ok2/run.py benchmark --device 0 --weights runs/public/pg_yolo_new_OK2/weights/best.pt
```

数据放在其他目录时，给训练或验证命令加 `--data /path/to/new_OK2/data.yaml`。已有同名运行时，请将示例中的权重路径换成该次运行实际保存的位置。

FPS 测量使用 1024 × 1024 输入、FP32、批量大小 1 及未融合模型。输入预先置于 GPU，预热 100 次后，每次对连续 500 次网络前向推理进行同步计时，取重复测量的 FPS 中位数。数据读取、预处理与后处理均不计入。参数量按模型统计，FLOPs 使用 fvcore 估算，乘加按 2 次浮点运算计，并补充注意力与选择性扫描算子的计数规则。

## 许可证与致谢

本项目基于 Ultralytics 框架开发，保留原有 [AGPL-3.0 许可证](LICENSE)及版权声明。

- [Selective Scan 源码](third_party/selective_scan/)及 [VMamba 许可证](third_party/VMamba_LICENSE)
- 类别词向量基于 GloVe。
