<h1>"The score is finally 0.83170"</h1>

<img 
  width="1336" 
  height="634" 
  alt="final—rank" 
  src="https://github.com/user-attachments/assets/02e8e3ec-3274-4f6d-9679-bd0e131288cb" 
/>

# v36 相对于 v9 的主要修改（依旧让deepseek写写吧）

## 一、特征工程 (新增 `feature.py`)

| 修改项 | v9 | v36 |
|--------|-----|-----|
| 时间特征 | 无 | 新增 `user_int_feats_201`(小时)、`202`(星期)、`203`(小时×星期交叉) |
| 活跃度特征 | 无 | 新增 `user_dense_feats_901`(log序列长度)、`902`(序列长度比值) |
| 特征注入 | 无 | `infer.py` 在线注入时序/活跃度特征，自动对齐训练开关 |

## 二、模型架构 (`model.py`)

| 修改项 | v9 | v36 |
|--------|-----|-----|
| 序列兴趣多视角 | 无 | `seq_interest_ratios` 参数，将每个序列按比例切分为多个前缀视图（默认 1.0, 0.7, 0.1） |
| User Dense 处理 | 单一 token + pair residual | 三种模式：`old`(legacy)、`new`(多token对齐)、`none`(无pair) |
| User Dense Tokenizer | 无 | `aligned` 模式：每个 dense 特征独立 token，支持 int+dense 对齐融合 |
| 特征排除 | 无 | `excluded_user_int_fids` / `excluded_user_dense_fids` 参数 |
| Dense 归一化 | 无 | `SimpleDenseNorm`：平滑裁剪 + 可学习仿射变换 |
| 独立 Dense Token | 无 | `independent_user_dense_fids` 参数（默认 61,87）作为独立 token |

## 三、训练器 (`trainer.py`)

| 修改项 | v9 | v36 |
|--------|-----|-----|
| 早停策略 | 早停触发后停止训练 | **禁用了早停**：保存所有 epoch 的 checkpoint，仅记录最佳 AUC |
| Checkpoint 保存 | 仅保存最佳模型 | 每个 epoch/step 都保存独立 checkpoint |
| 特征注入 | 无 | `infer.py` 中 `_clean_missing_columns()` 处理缺失列 |
| OOB 处理 | 直接 clip 为 0 | 对 vocab>1 的特征做 modulo 映射 |

## 四、配置参数 (新增)

```python
# 模型参数
--seq_interest_ratios "1.0,0.7,0.1"     # 序列前缀比例
--user_dense_pair_mode "new"             # 'old' | 'new' | 'none'
--user_dense_tokenizer_type "aligned"    # 'global' | 'aligned'
--aligned_user_dense_fids "62,63,64,65,66"
--independent_user_dense_fids "61,87"
--enable_user_dense_smooth_norm          # 启用 dense 平滑归一化
--user_dense_smooth_norm_fids "62,63,64,65,66"
--excluded_user_dense_fids ""            # 排除的 dense fids
--excluded_user_int_fids ""              # 排除的 int fids

# 推理参数 (infer.py 新增)
--inject_901_902                         # 在线注入活跃度特征
```

## 五、推理增强 (`infer.py`)

| 功能 | 说明 |
|------|------|
| 在线特征注入 | `process_eval_data()` 对测试集注入 201/202/203/901/902 特征 |
| 缺失列处理 | `_clean_missing_columns()` 过滤不存在的列，对应特征置零 |
| 自动对齐 | 读取 checkpoint 的 schema.json 判断是否需要 901/902 |

## 六、数据集处理 (`dataset.py`)

| 修改项 | v9 | v36 |
|--------|-----|-----|
| OOB 处理 | 直接 clip 为 0 | `vocab>1` 时用 modulo 映射到有效范围 |
| Dense 特征 | log1p 仅对 62-66 | 保持不变 |
