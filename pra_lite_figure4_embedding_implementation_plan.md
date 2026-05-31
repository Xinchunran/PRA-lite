# PRA-lite 对齐 PRAGMA Figure 4 的 Embedding/Tokenization 实施方案

本文档针对 PRA-lite 当前与 PRAGMA Figure 4 / §2.2 / §2.3 的主要差距，给出前四项的具体落地方案：

1. within-field positional embedding
2. numerical / categorical / textual tokenization
3. calendar feature
4. MLM 三路上下文与 embedding-table matching

> 不把 `time-to-last-event` 作为本文主改动项：PRA-lite 已经使用 PRAGMA 风格的 soft log-seconds 变换，争议主要在 anchor 是 `evaluation_ts` 还是 `last_event_ts`。这个建议作为单独小 PR 处理，避免和 tokenizer 版本升级混在一起。

---

## 0. 当前判断

### 0.1 已经符合的核心结构

PRA-lite 当前已经具备 PRAGMA Figure 4 的 backbone skeleton：

```text
Profile key/value/time -> Profile State Encoder -> [USR]
Event key/value/calendar -> Event Encoder -> [EVT]
[USR] + [EVT sequence] + event time -> History Encoder -> zh
```

主要模型结构已经在 `src/model/pragma_lite/model.py` 中存在：

- `KeyValueEmbedding`
- `profile_encoder`
- `event_encoder`
- `history_encoder`
- `calendar_proj`
- `time_proj`
- `zh_usr` / `zh_evt`
- `_mlm_logits(local_context, event_context, user_context)`

### 0.2 需要修的四项

| 项目 | 当前状态 | 目标状态 |
|---|---|---|
| within-field positional embedding | 模型里有 `value_pos_emb`，但 tokenizer 里多数 `value_pos=0` | 多 token value 正常展开，key 复制，`value_pos=0,1,2...` |
| numerical/categorical/textual tokenization | numeric/categorical 有简化，textual BPE/subword path 不完整 | 每个 field 先判定 value type，再走 numeric bucket / categorical lookup / textual BPE |
| calendar feature | 6 维 sin/cos + 单层 Linear | 6 维 sin/cos + two-layer MLP + 加到 `[EVT]` |
| MLM 三路上下文 | 已有三路 concat，但输出是独立 Linear | 三路 concat 后投回 `d_model`，再与共享 embedding table 做匹配，最好 weight tying |

---

## 1. 改动一：within-field positional embedding 真正生效

### 1.1 目标

论文里一个字段可以产生多个 value tokens。例如：

```text
Description = "metal plan"
```

应该被编码成：

```text
key_ids:   [K:E:Description, K:E:Description, K:E:Description]
value_ids: [T:met,           T:al,            T:plan]
value_pos: [0,               1,               2]
```

而不是当前类似：

```text
key_ids:   [K:E:Description]
value_ids: [V:E:Description=metal plan]
value_pos: [0]
```

### 1.2 需要修改的文件

优先修改：

```text
src/tokenizer/structured.py
src/tokenizer/vocab.py
```

可新增：

```text
src/tokenizer/schema.py
src/tokenizer/text_bpe.py
```

### 1.3 设计新的字段编码函数

在 `structured.py` 里把当前 `_encode_value(...) -> int` 改成更通用的：

```python
def _encode_field(
    vocab: TokenizerVocab,
    namespace: str,
    col: str,
    value: Any,
    max_value_tokens: int,
) -> list[int]:
    """Return one or more value token ids for a single field."""
    field_key = f"{namespace}:{col}"
    value_type = vocab.field_value_types.get(field_key, "categorical")

    if value_type == "numeric":
        return [_encode_numeric_value(vocab, namespace, col, value)]

    if value_type == "categorical":
        return [_encode_categorical_value(vocab, namespace, col, value)]

    if value_type == "textual":
        return _encode_textual_value(
            vocab=vocab,
            namespace=namespace,
            col=col,
            value=value,
            max_value_tokens=max_value_tokens,
        )

    return [vocab.unk_id]
```

然后事件和 profile 的 loop 从：

```python
key_ids.append(vocab.encode_token(f"K:E:{col}"))
value_ids.append(_encode_value(vocab, "E", col, value))
value_pos.append(0)
```

改成：

```python
key_id = vocab.encode_token(f"K:E:{col}")
field_value_ids = _encode_field(
    vocab=vocab,
    namespace="E",
    col=col,
    value=value,
    max_value_tokens=vocab.max_value_tokens_per_field,
)

for pos, value_id in enumerate(field_value_ids):
    key_ids.append(key_id)
    value_ids.append(value_id)
    value_pos.append(pos)
```

Profile 侧同理，只是 namespace 从 `E` 换成 `P`。

### 1.4 max_event_tokens 的截断策略

当前 event 是固定 `max_event_tokens=24`。引入 multi-value 后，单个 event 的 token 数会增加，因此需要明确截断策略。

推荐策略：

1. 按 `vocab.event_cols` 的固定字段顺序编码。
2. 每个字段最多保留 `max_value_tokens_per_field` 个 value tokens。
3. 整个 event 超过 `max_event_tokens` 后截断。
4. 记录截断率，作为 preprocessing 质量指标。

示例：

```python
if len(key_ids) >= max_event_tokens:
    break
remaining = max_event_tokens - len(key_ids)
field_value_ids = field_value_ids[:remaining]
```

### 1.5 需要新增的单元测试

新增测试文件：

```text
tests/test_structured_multivalue_tokenization.py
```

测试 1：text 字段展开

```python
def test_text_field_replicates_key_and_positions():
    encoded = encode_event_features(...)
    # Description="metal plan" 被 BPE 切成 3 个 token 的假设下：
    assert event_key_ids[desc_start:desc_start+3] == [desc_key_id] * 3
    assert event_value_pos[desc_start:desc_start+3] == [0, 1, 2]
```

测试 2：numeric/categorical 仍然是单 token

```python
def test_numeric_and_categorical_are_single_token():
    assert amount_positions == [0]
    assert currency_positions == [0]
```

测试 3：padding 不破坏 mask

```python
def test_multivalue_padding_mask():
    assert sum(event_token_mask[0]) == actual_nonpad_token_count
    assert all(pos == 0 for pos, m in zip(event_value_pos[0], event_token_mask[0]) if m == 0)
```

### 1.6 完成标准

完成后，跑一个小样本统计：

```text
value_pos > 0 的 token 占比 > 0
textual fields average tokens per field > 1
max_event_tokens 截断率可观测，最好 < 1-5%，具体看数据
```

---

## 2. 改动二：numerical / categorical / textual tokenization

### 2.1 目标

把 value encoding 从“所有非数值基本都按完整字符串 lookup”改成三路：

```text
numeric     -> percentile bucket + zero bucket
categorical -> one token per category value
textual     -> BPE/subword tokens
```

这一步是 Figure 4 embedding 复现里最关键、也最容易影响旧数据兼容性的改动。

### 2.2 字段类型判定

新增 `src/tokenizer/schema.py`：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class FieldSchema:
    namespace: str          # "P" or "E"
    name: str               # column name
    value_type: str         # "numeric", "categorical", "textual"
    cardinality: int | None = None
```

字段类型推荐规则：

```text
1. pandas dtype 是 number，或者字段名在 numeric override list 中 -> numeric
2. string 字段 unique_count <= categorical_threshold -> categorical
3. string 字段 unique_count > categorical_threshold -> textual
4. 手动 override 优先级最高，例如 MCC、currency、direction 应强制 categorical
```

建议 config：

```python
@dataclass(frozen=True)
class VocabBuildConfig:
    num_numeric_bins: int = 100
    categorical_threshold: int = 2048
    max_text_vocab_size: int = 28000
    max_value_tokens_per_field: int = 8
    numeric_zero_bucket: bool = True
    force_categorical_cols: tuple[str, ...] = ("currency", "mcc", "type", "direction")
    force_textual_cols: tuple[str, ...] = ("description", "merchant_name", "memo")
```

### 2.3 修改 TokenizerVocab

当前 `TokenizerVocab` 只保存：

```python
token_to_id
profile_cols
event_cols
numeric_binners
```

建议扩展为：

```python
class TokenizerVocab:
    def __init__(
        self,
        token_to_id: dict[str, int],
        profile_cols: list[str],
        event_cols: list[str],
        numeric_binners: dict[str, NumericBinner],
        field_value_types: dict[str, str],
        categorical_values: dict[str, list[str]],
        text_tokenizer_path: str | None = None,
        max_value_tokens_per_field: int = 8,
    ) -> None:
        ...
```

保存到 `tokenizer.json` 时增加：

```json
{
  "field_value_types": {
    "E:amount": "numeric",
    "E:currency": "categorical",
    "E:description": "textual"
  },
  "categorical_values": {
    "E:currency": ["gbp", "eur", "usd"]
  },
  "text_tokenizer_path": "text_bpe.json",
  "max_value_tokens_per_field": 8,
  "tokenizer_version": 2
}
```

### 2.4 numeric encoding

当前 `NumericBinner.bucket()` 可以继续用，但建议增加 zero bucket。

目标 token 格式：

```text
V:E:amount#ZERO
V:E:amount#B0
V:E:amount#B1
...
```

实现：

```python
def _encode_numeric_value(vocab, namespace, col, value) -> int:
    field_key = f"{namespace}:{col}"
    if value is None or pd.isna(value):
        return vocab.encode_token(f"V:{field_key}#[NA]")
    try:
        v = float(value)
    except Exception:
        return vocab.encode_token(f"V:{field_key}#[NA]")

    if vocab.numeric_zero_bucket and v == 0.0:
        return vocab.encode_token(f"V:{field_key}#ZERO")

    bucket = vocab.numeric_binners[field_key].bucket(v)
    return vocab.encode_token(f"V:{field_key}#B{bucket}")
```

注意：如果使用 zero bucket，训练 binner 的 percentile edges 时最好排除 0，否则 0 会同时影响 percentile 分布。

### 2.5 categorical encoding

目标 token 格式保持 field-specific：

```text
V:E:currency=gbp
V:E:direction=out
V:P:plan=metal
```

OOV 推荐使用 field-specific unknown，而不是全局 `[UNK]`：

```text
V:E:currency=[UNK]
V:P:plan=[UNK]
```

实现：

```python
def _encode_categorical_value(vocab, namespace, col, value) -> int:
    field_key = f"{namespace}:{col}"
    token_value = "[NA]" if value is None or pd.isna(value) else str(value).strip().lower()
    token = f"V:{field_key}={token_value}"
    if token in vocab.token_to_id:
        return vocab.token_to_id[token]
    return vocab.encode_token(f"V:{field_key}=[UNK]")
```

### 2.6 textual / BPE encoding

新增 `src/tokenizer/text_bpe.py`。推荐直接使用 Hugging Face `tokenizers`，不要自己实现 BPE。

训练阶段：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFKC, Sequence

special_tokens = ["[UNK]"]
text_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
text_tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
text_tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=max_text_vocab_size, special_tokens=special_tokens)
text_tokenizer.train_from_iterator(text_iterator, trainer=trainer)
text_tokenizer.save(str(path / "text_bpe.json"))
```

编码阶段：

```python
def _encode_textual_value(vocab, namespace, col, value, max_value_tokens):
    text = "" if value is None or pd.isna(value) else str(value)
    if not text.strip():
        return [vocab.encode_token("T:[NA]")]

    pieces = vocab.text_tokenizer.encode(text).tokens
    pieces = pieces[:max_value_tokens]
    if not pieces:
        return [vocab.encode_token("T:[UNK]")]
    return [vocab.encode_token(f"T:{piece}") for piece in pieces]
```

这里用全局 text token namespace：

```text
T:met
T:al
T:plan
```

而不是 field-specific：

```text
V:E:description=met
```

原因是论文说 key 和 value 从共享 lookup table 中取 embedding，key 已经提供字段语义，text subtoken 本身可以跨字段共享。

### 2.7 vocabulary 构建流程

建议把 tokenizer fit 分成 4 步：

```text
Step 1: infer field schema
Step 2: fit numeric binners
Step 3: collect categorical tokens
Step 4: train textual BPE and add text pieces into shared token_to_id
```

伪代码：

```python
def build_vocab_v2(profile_df, events_df, cfg):
    field_schemas = infer_field_schemas(profile_df, events_df, cfg)
    numeric_binners = fit_numeric_binners(profile_df, events_df, field_schemas, cfg)
    categorical_values = collect_categorical_values(profile_df, events_df, field_schemas, cfg)
    text_tokenizer = train_text_bpe(profile_df, events_df, field_schemas, cfg)

    token_to_id = init_special_tokens()
    add_key_tokens(token_to_id, profile_cols, event_cols)
    add_numeric_tokens(token_to_id, numeric_binners, cfg)
    add_categorical_tokens(token_to_id, categorical_values)
    add_text_bpe_tokens(token_to_id, text_tokenizer)

    return TokenizerVocab(...)
```

### 2.8 兼容性要求

这一步会改变 token ids，因此不能直接复用旧的 LMDB / parquet tokenized dataset。

必须做：

```text
1. tokenizer_version 从 1 升级到 2
2. 新 tokenizer 保存到新目录，例如 artifacts/tokenizer_v2/
3. 新 tokenized dataset 保存到新目录，例如 data/tokenized_lmdb_v2/
4. training config 明确记录 tokenizer_version 和 tokenizer_path
```

不要把 v1/v2 数据混在同一个训练 run 里。

### 2.9 完成标准

Preprocessing 后输出一份统计：

```text
field_value_type counts:
  numeric: xx
  categorical: xx
  textual: xx

textual tokenization:
  avg text pieces per textual field: xx
  p95 text pieces per textual field: xx
  truncation rate by max_value_tokens_per_field: xx%

unknown rate:
  categorical OOV rate: xx%
  textual BPE [UNK] rate: xx%

sequence length:
  avg event tokens: xx
  p95 event tokens: xx
  event truncation rate at max_event_tokens=24: xx%
```

---

## 3. 改动三：calendar feature 从单层 Linear 改成 two-layer MLP

### 3.1 目标

当前 PRA-lite 已经有：

```python
calendar_features: [sin(hour), cos(hour), sin(dow), cos(dow), sin(dom), cos(dom)]
calendar_proj = nn.Linear(6, d_model)
```

目标改成：

```text
calendar_features -> Linear(6, d_model) -> GELU -> Dropout -> Linear(d_model, d_model)
```

然后继续加到每个 event 的 `[EVT]` embedding：

```python
ze = z_evt_prime + z_calendar
```

### 3.2 修改文件

```text
src/model/pragma_lite/model.py
```

### 3.3 config 增加开关

在 `PragmaLiteConfig` 增加：

```python
calendar_mlp: bool = True
calendar_hidden_dim: int | None = None
```

`__post_init__` 中：

```python
if self.calendar_hidden_dim is None:
    self.calendar_hidden_dim = self.d_model
```

### 3.4 替换 calendar_proj

从：

```python
self.calendar_proj = nn.Linear(6, self.d_model)
```

改成：

```python
if cfg.calendar_mlp:
    self.calendar_proj = nn.Sequential(
        nn.Linear(6, cfg.calendar_hidden_dim),
        nn.GELU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.calendar_hidden_dim, self.d_model),
    )
else:
    self.calendar_proj = nn.Linear(6, self.d_model)
```

forward 不需要大改，因为调用方式仍然是：

```python
event_embeddings = event_embeddings + self.calendar_proj(calendar_features.to(event_embeddings.dtype))
```

### 3.5 测试

新增：

```text
tests/test_calendar_projection.py
```

测试：

```python
def test_calendar_proj_shape():
    cfg = PragmaLiteConfig(vocab_size=100, d_model=192, calendar_mlp=True)
    model = PragmaLiteModel(cfg)
    x = torch.randn(2, 10, 6)
    y = model.calendar_proj(x)
    assert y.shape == (2, 10, 192)
```

### 3.6 完成标准

```text
1. calendar_features 仍是 6 维。
2. model forward shape 不变。
3. 旧 checkpoint 不能直接 strict load；如果需要兼容，load_state_dict(strict=False)。
4. small batch smoke training loss 正常下降。
```

---

## 4. 改动四：MLM 三路上下文与 embedding-table matching

### 4.1 目标

当前 PRA-lite 已经做了三路上下文：

```text
local_context = Event Encoder token-level hidden
 event_context = History Encoder corresponding [EVT] hidden
 user_context  = History Encoder [USR] hidden
```

当前 `_mlm_logits` 大概是：

```python
concat([local, event, user]) -> Linear(3d, d) -> GELU -> Linear(d, vocab_size)
```

目标改成更贴近论文的 matched against embedding table：

```text
concat([local, event, user]) -> projection to d_model -> dot(shared token embedding table)
```

即 MLM 输出头和 `kv_embedding.token_emb.weight` 绑定。

### 4.2 修改文件

```text
src/model/pragma_lite/model.py
```

### 4.3 config 增加开关

在 `PragmaLiteConfig` 增加：

```python
tie_mlm_to_embedding: bool = True
```

### 4.4 替换 MLM head

当前：

```python
self.mlm_head = nn.Sequential(
    nn.Linear(self.d_model * 3, self.d_model),
    nn.GELU(),
    nn.Linear(self.d_model, self.vocab_size),
)
```

建议改成：

```python
self.mlm_fuse = nn.Sequential(
    nn.Linear(self.d_model * 3, self.d_model),
    nn.GELU(),
    nn.LayerNorm(self.d_model),
)

if cfg.tie_mlm_to_embedding:
    self.mlm_bias = nn.Parameter(torch.zeros(self.vocab_size))
    self.mlm_head = None
else:
    self.mlm_head = nn.Linear(self.d_model, self.vocab_size)
```

### 4.5 修改 `_mlm_logits`

```python
import torch.nn.functional as F


def _mlm_logits(
    self,
    local_context: torch.Tensor,
    event_context: torch.Tensor,
    user_context: torch.Tensor,
) -> torch.Tensor:
    if local_context.ndim == 4:
        user_context = user_context.unsqueeze(1).unsqueeze(2).expand_as(local_context)
    elif local_context.ndim == 3:
        user_context = user_context.unsqueeze(1).expand_as(local_context)
    else:
        raise ValueError(f"Unsupported local_context rank: {local_context.ndim}")

    fused = torch.cat([local_context, event_context, user_context], dim=-1)
    hidden = self.mlm_fuse(fused)

    if self.cfg.tie_mlm_to_embedding:
        return F.linear(hidden, self.kv_embedding.token_emb.weight, self.mlm_bias)

    return self.mlm_head(hidden)
```

### 4.6 是否同时预测 key 和 value？

建议分两阶段：

#### 阶段 A：保持当前训练标签逻辑

先只改输出头，不改 masking/data collator。这样风险最低。

```text
输入：masked event value tokens
标签：原始 value token ids
输出：vocab logits
```

#### 阶段 B：再扩展 key/value 都可 mask

论文说 masked input tokens，严格上 key/value 都在共享 vocabulary 中，可以都预测。但工程上建议先别混在本 PR。

### 4.7 label smoothing 与 ignore_index

保留当前 cross entropy + label smoothing 逻辑。

需要确保：

```text
1. 非 mask 位置 label = -100
2. [UNK] 替换位置如果按论文作为 input dropout，不进入 MLM loss
3. padding 位置 label = -100
```

### 4.8 测试

新增：

```text
tests/test_mlm_weight_tying.py
```

测试 1：shape

```python
def test_mlm_logits_shape():
    logits = model(..., return_mlm_logits=True)
    assert logits.shape == (batch_size, num_events, num_tokens, vocab_size)
```

测试 2：确实使用共享 embedding table

```python
def test_mlm_uses_shared_embedding_weight():
    assert model.cfg.tie_mlm_to_embedding
    assert model.kv_embedding.token_emb.weight.shape[0] == model.vocab_size
```

测试 3：反向传播正常

```python
def test_mlm_backward_with_tied_embeddings():
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    loss.backward()
    assert model.kv_embedding.token_emb.weight.grad is not None
```

### 4.9 完成标准

```text
1. return_mlm_logits=True 输出 shape 不变。
2. 参数量略微下降，因为移除了独立 Linear(d, vocab_size)。
3. 训练 loss 初始值正常，不出现 NaN。
4. embedding table gradient 正常。
```

---

## 5. 推荐落地顺序

不要一次性全改。推荐分成 4 个 PR / commit 组。

### PR 1：低风险模型结构改动

包含：

```text
1. calendar_proj: Linear -> two-layer MLP
2. mlm_head: independent Linear -> tied embedding table
```

原因：

```text
- 不改变 tokenized dataset。
- 不需要重跑 preprocessing。
- 最容易用现有 batch 做 smoke test。
```

验证：

```bash
pytest tests/test_calendar_projection.py tests/test_mlm_weight_tying.py
python -m scripts.smoke_train --max_steps 20
```

### PR 2：TokenizerVocab v2 schema

包含：

```text
1. 增加 field_value_types
2. 增加 categorical_values
3. 增加 text_tokenizer_path
4. 增加 tokenizer_version=2
5. 增加 max_value_tokens_per_field
```

原因：

```text
- 先让 vocab 能保存/加载新信息。
- 但暂时不改变 encode_record 输出。
```

验证：

```bash
pytest tests/test_vocab_v2_roundtrip.py
```

### PR 3：multi-value field encoding + textual BPE

包含：

```text
1. _encode_value -> _encode_field
2. key replication
3. value_pos = 0,1,2...
4. train_text_bpe / load_text_bpe
5. textual fields BPE encoding
```

原因：

```text
- 这是最大改动，需要单独 review。
- 会改变 tokenized dataset，必须重跑 preprocessing。
```

验证：

```bash
pytest tests/test_structured_multivalue_tokenization.py
python -m scripts.inspect_tokenizer_v2 --tokenizer artifacts/tokenizer_v2
```

### PR 4：重新预处理 + 小规模训练验证

包含：

```text
1. 新建 tokenizer_v2
2. 新建 tokenized_lmdb_v2
3. 跑 1k-10k batch smoke pretrain
4. 对比 old tokenizer vs new tokenizer 的 loss 曲线和 throughput
```

验证指标：

```text
- event truncation rate
- [UNK] rate
- value_pos > 0 token ratio
- throughput tokens/sec
- MLM loss curve
- downstream probe sanity check
```

---

## 6. 推荐配置

建议先用 conservative 配置，避免 sequence length 暴涨。

```python
VocabBuildConfig(
    num_numeric_bins=100,
    categorical_threshold=2048,
    max_text_vocab_size=28000,
    max_value_tokens_per_field=4,
    numeric_zero_bucket=True,
)
```

模型配置：

```python
PragmaLiteConfig(
    d_model=192,
    n_heads=3,
    d_ffn=768,
    profile_layers=1,
    event_layers=5,
    history_layers=2,
    calendar_mlp=True,
    tie_mlm_to_embedding=True,
)
```

注意：`max_value_tokens_per_field=4` 是第一版保守选择。如果 truncation rate 很低，可以提升到 8。

---

## 7. 风险与注意事项

### 7.1 旧 checkpoint 不完全兼容

以下改动会影响 checkpoint loading：

```text
calendar_proj 从 Linear 变 Sequential
mlm_head 从 Sequential 变 mlm_fuse + tied embedding
vocab_size 可能变化
```

建议：

```python
model.load_state_dict(ckpt, strict=False)
```

但如果 tokenizer 变成 v2，最干净的做法是重新 pretrain。

### 7.2 tokenized dataset 必须重建

只要引入 textual BPE 和 multi-value field encoding，旧 tokenized dataset 就不能继续用。

必须新建：

```text
artifacts/tokenizer_v2/
data/tokenized_lmdb_v2/
runs/pretrain_v2_strict_pragma_s/
```

### 7.3 max_event_tokens 可能需要重新调

原来的 `max_event_tokens=24` 在 one-token-per-field 时够用；引入 BPE 后可能截断更多。

建议先统计：

```text
max_event_tokens = 24, 32, 48
```

三个版本的 truncation rate 和 throughput，再决定正式值。

### 7.4 field-specific categorical OOV 比全局 [UNK] 更稳

建议 categorical 不要直接落到 `[UNK]`，而是落到：

```text
V:E:currency=[UNK]
V:P:plan=[UNK]
```

这样模型知道“哪个字段未知”，不会把所有未知 categorical 混成一个语义。

---

## 8. 最小可执行任务清单

### Day 1

- [ ] 新增 `calendar_mlp` config
- [ ] 替换 `calendar_proj`
- [ ] 新增 `tie_mlm_to_embedding` config
- [ ] 替换 `_mlm_logits`
- [ ] 跑模型 forward/backward smoke test

### Day 2

- [ ] 扩展 `TokenizerVocab` 保存/加载 v2 schema
- [ ] 新增 `FieldSchema` / `VocabBuildConfig`
- [ ] 完成 vocab roundtrip test

### Day 3-4

- [ ] 实现 `_encode_field`
- [ ] 实现 key replication + value_pos range
- [ ] 实现 numeric zero bucket
- [ ] 实现 categorical field-specific OOV
- [ ] 实现 textual BPE encode
- [ ] 跑 tokenizer unit tests

### Day 5

- [ ] 重建一个小样本 tokenizer_v2
- [ ] 重建小样本 tokenized dataset
- [ ] 跑 1k-10k step pretrain smoke
- [ ] 输出 tokenization report

---

## 9. 本次改动完成后的预期符合度

如果按上面四项完成，PRA-lite 对 Figure 4 embedding/tokenization 的符合度会从当前大约 74-78% 提升到约 85-90%。

剩余主要差距会变成：

```text
1. time-to-last-event anchor 是否严格使用 last included event
2. sequence packing / varlen attention 是否对齐论文工程实现
3. profile lifelong event schema 是否足够接近原论文
4. masking strategy 是否同时覆盖 token-level / event-level / key-level masking
```

其中 `time-to-last-event anchor` 已经可以通过 preprocessing 配置切换：

```python
last_event_ts = max(event_timestamps_in_history)
event_time = soft_log_seconds(last_event_ts - event_ts)
```

此外，training 侧目前已经补上了两级效率改进：

```text
1. event-count bucketed batching
2. token-budget dynamic batch sampler
```

当前状态是：

- record 级语义保持不变
- batch 内会裁剪到局部最大 `profile/event/event_token` 长度
- 仍然是 rectangular tensor + 常规 attention
- 尚未实现 PRAGMA 论文里的 event-token varlen packing / varlen attention kernel

也就是说，PRA-lite 已经不再是“固定 batch_size + 全局固定 shape 直接 stack”的最原始路径，但仍未完全复现论文强调的 varlen batching 工程实现。
