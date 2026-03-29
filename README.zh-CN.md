# scPlantFormer：面向植物单细胞组学整合与注释的 Transformer 框架

> **仓库定位**：科研代码 + 教程  
> **核心任务**：跨数据集细胞类型注释、跨模态整合、跨物种/组织/批次迁移学习  
> **适用模态**：scRNA-seq、scATAC-seq

## 1. 项目简介

scPlantFormer 旨在为植物单细胞数据提供统一的表示学习与下游分析流程，重点支持：

- 跨数据集细胞类型注释；
- 多模态（RNA/ATAC）协同分析；
- 跨物种、跨组织、跨批次的整合与迁移。

仓库由两部分构成：

- `model/`：模型与训练/推理/评估工具代码；
- `Tutorial/`：可复现实验脚本与 Notebook 示例。

## 2. 目录结构

```text
scPlantFormer/
├── README.md
├── README.zh-CN.md
├── model/
│   ├── model_config_mae.py
│   ├── data_utils_mae.py
│   ├── infer_utils_mae.py
│   ├── eval_utils_mae.py
│   └── scplantFormer_model_mae.py
├── Tutorial/
│   ├── Integration_batch.ipynb
│   ├── Integration_species.ipynb
│   ├── Integration_tissues.ipynb
│   ├── Integration_scRNA_scATAC.ipynb
│   ├── cross_dataset_cell-type_annoatation.py
│   └── inner_cell_type_annotation.py
└── Baseline_scLLM/
    └── scLLM
```

## 3. 环境依赖

建议 Python 版本：**3.9+**

```bash
pip install torch scanpy scikit-learn geomloss tqdm numpy pandas
```

建议在论文复现中固定版本并导出锁定文件（例如 `requirements.lock.txt`）。

## 4. 快速开始

1. 准备 AnnData/h5ad 格式的数据与元信息（细胞类型、批次、物种、组织标签）。
2. 选择对应任务的教程：
   - 批次整合：`Tutorial/Integration_batch.ipynb`
   - 跨物种整合：`Tutorial/Integration_species.ipynb`
   - 跨组织整合：`Tutorial/Integration_tissues.ipynb`
   - RNA-ATAC 整合：`Tutorial/Integration_scRNA_scATAC.ipynb`
   - 脚本化注释：`Tutorial/cross_dataset_cell-type_annoatation.py`、`Tutorial/inner_cell_type_annotation.py`
3. 在统一划分策略下进行训练、推理与评估。

## 5. 面向高水平期刊的实验与写作建议

建议在论文中补全以下关键要素：

- **数据透明性**：数据来源、样本量、质控与过滤策略；
- **严格数据划分**：避免 donor/批次信息泄漏；
- **充分对比基线**：传统 ML + 深度学习方法；
- **消融实验**：模型结构、损失函数、模态贡献；
- **稳健性实验**：不同随机种子、低标注、跨域迁移；
- **统计学报告**：均值±标准差、显著性检验或置信区间；
- **可解释性分析**：marker 基因、注意力/嵌入可视化。

## 6. 复现性清单

- [ ] 固定并公开随机种子；
- [ ] 报告软件版本与硬件配置；
- [ ] 公开预处理流程与参数；
- [ ] 保存 train/val/test 划分；
- [ ] 多次独立运行并报告波动范围；
- [ ] 报告失败案例或局限性。

## 7. 局限与注意事项

- 当前仓库更偏向研究原型与教程，尚未形成完整工程化发行；
- 默认未提供严格锁定的依赖文件；
- 不同数据集可能需要按本地格式调整预处理与字段映射。

## 8. 引用与学术使用

学术使用时建议：

1. 在论文中引用本仓库链接与具体 commit hash；
2. 同时引用数据集与对比方法原始文献；
3. 报告完整实验配置以支持可复核。

