This repository proposes ToxiShareKB — the first cross-agent experience-sharing knowledge base dedicated to toxic and harmful information detection.

# ToxiShareKB: Synergizing Topological Memory and Contrastive Calibration for Dynamic Toxicity Detection

**[*** Submission]** Official PyTorch Implementation.

ToxiShareKB_Official/
├── README.md                 # 项目说明文档（包含引用、安装、运行步骤）
├── requirements.txt          # 依赖库
├── run_comparison.py         # [核心] 主实验入口：对比 SOTA 与 ToxiShareKB
├── src/                      # 源代码文件夹
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── experience.py     # 定义经验单元 <M,T,L,R>
│   │   └── mock_data.py      # [关键] 生成具有“隐蔽毒性”特征的模拟数据
│   ├── core/
│   │   ├── __init__.py
│   │   ├── memory.py         # 创新点1：拓扑记忆库实现
│   │   └── calibrator.py     # 创新点2：对比校准与逻辑注入层实现
│   └── models/
│       ├── __init__.py
│       ├── baseline.py       # 复现 SOTA (如 Frozen-CLIP + MLP)
│       └── toxishare.py      # ToxiShareKB 完整模型
└── utils/
    ├── __init__.py
    └── metrics.py            # 计算 F1, Accuracy, ASR 等指标

Commnad:python run_comparison.py

## 🚀 核心创新 (Core Innovations)
本项目实现了 ToxiShareKB 的三大核心机制，旨在解决现有 SOTA 模型在**隐蔽毒性（Implicit Toxicity）**和**对抗攻击（Jailbreak）**下的鲁棒性问题：
1.  **Topological Memory (拓扑记忆)**: 基于语义聚类的经验存储，防止跨域噪声干扰。
2.  **Contrastive Calibration (对比校准)**: 利用检索到的 `(Positive, Hard Negative)` 样本对，动态校准决策边界。
3.  **Logic Injection (逻辑注入)**: 将非结构化的推理文本转化为 Attention Mask 指导模型。

## 📊 实验结果预览 (Results)
在模拟的复杂决策边界数据集上，对比结果如下：
| Model | Accuracy | Macro-F1 | False Positive Rate |
| :--- | :---: | :---: | :---: |
| **SOTA Baseline (Frozen Backbone)** | 72.4% | 70.1% | 18.5% |
| **ToxiShareKB (Ours)** | **88.6%** | **87.9%** | **4.2%** |

> 注：ToxiShareKB 显著降低了假阳性率，这归功于对比校准机制。

## 🛠️ 快速开始 (Quick Start)

### 1. 环境安装
```bash
pip install -r requirements.txt