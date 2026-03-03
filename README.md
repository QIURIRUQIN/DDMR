# xPatch - 时间序列预测模型

## 项目简介

xPatch 是一个基于 Transformer 架构的时间序列预测模型，采用多尺度分解和混合专家（MoE）机制，能够有效捕捉时间序列中的长期和短期依赖关系。

## 主要特性

- **多尺度分解**：通过下采样和分解技术，同时建模不同时间尺度的趋势和季节性模式
- **混合专家模型（MoE）**：使用 LightMoE 实现专家混合机制，提高模型表达能力
- **可逆实例归一化（RevIN）**：增强模型对不同分布数据的泛化能力
- **稀疏注意力机制**：通过 TopK 选择和稀疏自注意力降低计算复杂度
- **频率共现注意力**：捕捉时间序列中的频率模式
- **多尺度周期性依赖关系（MPDR）**：专门建模周期性依赖关系

## 项目结构

```
model/
├── data_provider/          # 数据提供模块
│   ├── data_factory.py    # 数据工厂，统一数据接口
│   └── data_loader.py     # 数据加载器
├── exp/                    # 实验模块
│   ├── exp_basic.py       # 实验基类
│   └── exp_main.py        # 主实验类
├── layers/                 # 模型层组件
│   ├── decomp.py          # 时间序列分解（EMA/DEMA）
│   ├── dema.py            # 双指数移动平均
│   ├── down_sampling.py   # 下采样处理
│   ├── ema.py             # 指数移动平均
│   ├── Embed.py           # 嵌入层
│   ├── mpdr.py            # 多尺度周期性依赖关系
│   ├── revin.py           # 可逆实例归一化
│   ├── Transformer.py     # Transformer 编码器
│   └── tsmoe.py           # 时间序列混合专家模型
├── model/                  # 模型定义
│   └── my_model.py        # 主模型实现
├── utils/                  # 工具函数
│   ├── metrics.py         # 评估指标
│   ├── timefeatures.py    # 时间特征编码
│   └── tools.py           # 辅助工具
├── run.py                  # 主运行脚本
└── requirements.txt        # 依赖包列表
```

## 环境要求

- Python 3.12.3
- PyTorch 2.5.1+cu124（支持 CUDA 12.4）
- 其他依赖见 `requirements.txt`

## 安装步骤

1. 克隆或下载项目到本地

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

3. 准备数据集：
   - 将数据集放置在 `./dataset/` 目录下
   - 支持的数据集格式：ETTh1, ETTh2, ETTm1, ETTm2, Solar, PEMS 或自定义数据集

## 使用方法

### 训练模型

```bash
python run.py \
    --is_training 1 \
    --model_id test \
    --model xPatch \
    --data ETTh1 \
    --root_path ./dataset \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --batch_size 32 \
    --train_epochs 100 \
    --learning_rate 0.0001
```

### 测试模型

```bash
python run.py \
    --is_training 0 \
    --model_id test \
    --model xPatch \
    --data ETTh1 \
    --root_path ./dataset \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96
```

## 主要参数说明

### 基础配置
- `--is_training`: 训练模式（1）或测试模式（0）
- `--model_id`: 模型标识符
- `--model`: 模型名称（xPatch）
- `--data`: 数据集名称
- `--root_path`: 数据集根目录
- `--data_path`: 数据文件路径

### 预测任务配置
- `--features`: 预测任务类型
  - `M`: 多变量预测多变量
  - `S`: 单变量预测单变量
  - `MS`: 多变量预测单变量
- `--seq_len`: 输入序列长度（默认：96）
- `--label_len`: 起始标记长度（默认：48）
- `--pred_len`: 预测序列长度（默认：96）
- `--enc_in`: 编码器输入维度（默认：7）

### 模型架构参数
- `--d_model`: 降维后的维度（默认：16）
- `--d_ff`: Transformer 前馈网络维度（默认：16）
- `--stride`: 补丁步长（默认：8）
- `--down_sampling_layers`: 下采样层数（默认：3）
- `--down_sampling_window`: 下采样窗口大小（默认：2）
- `--c`: 周期参数（默认：12）
- `--hidden_dim`: 隐藏层维度（默认：128）

### 移动平均参数
- `--ma_type`: 移动平均类型（`reg`, `ema`, `dema`，默认：`ema`）
- `--alpha`: EMA 平滑因子（默认：0.3）
- `--beta`: DEMA 平滑因子（默认：0.3）

### MoE 参数
- `--num_experts`: 专家数量（默认：3）
- `--top_k`: Top-K 专家选择（默认：3）
- `--base_alpha`: 调整因子（默认：10）

### 训练参数
- `--batch_size`: 批次大小（默认：32）
- `--train_epochs`: 训练轮数（默认：100）
- `--learning_rate`: 学习率（默认：0.0001）
- `--patience`: 早停耐心值（默认：10）
- `--loss`: 损失函数（默认：mse）
- `--revin`: 是否使用 RevIN（1：是，0：否，默认：1）

### GPU 配置
- `--use_gpu`: 是否使用 GPU（默认：True）
- `--gpu`: GPU 设备 ID（默认：0）
- `--use_multi_gpu`: 是否使用多 GPU（默认：False）
- `--devices`: 多 GPU 设备 ID（默认：'0,1,2,3'）

## 支持的数据集

- **ETTh1/ETTh2**: 电力变压器温度数据集（小时级）
- **ETTm1/ETTm2**: 电力变压器温度数据集（分钟级）
- **Solar**: 太阳能发电数据集
- **PEMS**: 交通流量数据集
- **custom**: 自定义数据集

## 模型架构

模型主要包含以下组件：

1. **数据预处理**：RevIN 归一化、通道混合（可选）
2. **多尺度下采样**：生成不同时间尺度的序列
3. **分解模块**：将序列分解为趋势和季节性成分
4. **时间依赖建模**：
   - MPDR 模块建模季节性依赖
   - 线性层建模趋势依赖
5. **通道依赖建模**：Transformer 编码器捕捉通道间关系
6. **多尺度融合**：融合不同尺度的特征
7. **混合专家模型**：使用 MoE 进行最终预测

## 评估指标

模型使用以下指标进行评估：
- **MSE** (Mean Squared Error): 均方误差
- **MAE** (Mean Absolute Error): 平均绝对误差

评估结果会保存在 `result.txt` 文件中。

## 输出文件

- **模型检查点**: `./checkpoints/{setting}/checkpoint.pth`
- **测试结果**: `./test_results/{setting}/`
- **评估指标**: `result.txt`

## 注意事项

1. 确保数据集格式正确，CSV 文件应包含时间戳和特征列
2. 根据数据集调整 `--enc_in` 参数（特征维度）
3. 根据硬件配置调整 `--batch_size` 和 `--num_workers`
4. 使用 GPU 训练时，确保已正确安装 CUDA 版本的 PyTorch
5. 模型使用固定随机种子（2026）以确保可复现性

## 许可证

请查看项目许可证文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 Issue 联系。
