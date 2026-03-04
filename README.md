# Humor-aware-retrieval

## 项目简介
该项目是一个基于深度学习的幽默感知检索（Humor-Aware Retrieval）系统，主要用于信息检索（Information Retrieval）任务，具体应用于JOKER（CLEF）评估活动的Task 1（幽默检索）。

项目结合了双编码器（Dual Encoder）和协同分类模型（Sequence Classification），实现了两阶段检索：
1. **召回阶段 (Retrieval)**：使用预训练语言模型（如 `paraphrase-multilingual-mpnet-base-v2`）构建双编码器（Dual Encoder），将查询（Query）和文档（Document）映射到向量空间，基于余弦相似度计算初始召回结果。
2. **精排阶段 (Re-ranking/Classification)**：使用 `RoBERTa` 序列分类模型（`roberta-base`）对召回阶段生成的（Query, Document）对进行相关性分类和打分，将分类模型的输出与召回阶段的分数进行加权融合，得到最终的排序结果。

## 目录结构
- `Humor_aware.py`: 模型训练和评估的核心代码，定义了基于RoBERTa微调的分类模型结构、数据集（`TextDataset`）以及训练（`train_model`）和测试（`eval_model`）循环。
- `EN_output.py`: 模型推理代码，读取英文测试数据集，使用训练好的RoBERTa模型对文本进行打分和排序，并将结果转换为标准格式输出（例如TREC run格式）。
- `test1.py`: 检索和召回模块的执行代码。它加载了双编码器模型，对文档库进行编码，计算给定的测试查询与所有文档的相似度得分，并利用阈值（如 >0.1）筛选出候选集，保存为 JSON 格式供下一阶段使用。
- `load_data.py`: 数据加载和预处理代码，定义了 `DualEncoder` 和 `MultiPositiveDataset`，以及用于对比学习（InfoNCE / 多正样本损失）的模型训练函数，支持在召回阶段微调多语言模型。
- `packerages.py`: 依赖包管理脚本，集中导入了所有需要的 Python 库（如 `torch`, `transformers`, `sentence_transformers`, `sklearn` 等），并附带了一些数据预处理的脚本逻辑。
- `set_paramters.py`: 参数配置文件。

## 核心依赖
项目主要使用以下库：
- PyTorch (`torch`)
- Hugging Face Transformers (`transformers`)
- Sentence Transformers
- scikit-learn
- tqdm
- numpy

## 模型流程
1. **获取候选文档** (`test1.py`)
   加载文档库并利用双编码器生成文档嵌入。然后读取测试Query，计算余弦相似度并根据相似度分数进行粗排和过滤，输出候选对至 `test.json`。
2. **候选对重排与融合打分** (`EN_output.py`)
   加载精排模型对第一阶段生成的候选集进行推断，获取分类 logits。模型将分类置信度与第一步的相似度分数融合（例如：`0.3 * 相似度 + 0.7 * 分类得分`），按降序排列，取 Top-1000 结果并生成标准评估文件。
