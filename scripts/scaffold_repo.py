from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
REGULAR_TRACK_KEYS = [
    "01_transformer_fundamentals",
    "02_tokenization",
    "03_inference_optimization",
    "04_training_finetuning",
    "05_rag_systems",
    "06_agents_tool_use",
    "07_ai_systems_engineering",
]


@dataclass(frozen=True)
class TrackSpec:
    key: str
    title_en: str
    title_zh: str
    desc_en: str
    desc_zh: str


@dataclass(frozen=True)
class ProblemSpec:
    pid: str
    order: int
    track: str
    slug: str
    title_en: str
    title_zh: str
    difficulty: int
    round_name: str
    estimated_minutes: int
    company_tags: tuple[str, ...]
    high_frequency: bool
    requirements_en: tuple[str, ...]
    requirements_zh: tuple[str, ...]
    followups_en: tuple[str, ...]
    followups_zh: tuple[str, ...]
    starter: dict[str, object]
    test_kind: str


@dataclass(frozen=True)
class MockPackSpec:
    pid: str
    slug: str
    title_en: str
    title_zh: str
    duration_minutes: int
    problems: tuple[str, ...]
    goals_en: tuple[str, ...]
    goals_zh: tuple[str, ...]


TRACKS = {
    "01_transformer_fundamentals": TrackSpec(
        key="01_transformer_fundamentals",
        title_en="Transformer Fundamentals",
        title_zh="Transformer 基础实现",
        desc_en="Hand-write the core Transformer building blocks that appear in model fundamentals rounds.",
        desc_zh="手写 Transformer 的核心模块，覆盖模型基础轮最常见实现题。",
    ),
    "02_tokenization": TrackSpec(
        key="02_tokenization",
        title_en="Tokenization",
        title_zh="Tokenization",
        desc_en="Practice subword algorithms, reversible encoding, and input packing.",
        desc_zh="训练子词算法、可逆编码与输入打包相关题型。",
    ),
    "03_inference_optimization": TrackSpec(
        key="03_inference_optimization",
        title_en="Inference Optimization",
        title_zh="推理优化",
        desc_en="Cover generation, sampling, batching, streaming, and cache reuse.",
        desc_zh="覆盖生成、采样、批处理、流式输出与缓存复用。",
    ),
    "04_training_finetuning": TrackSpec(
        key="04_training_finetuning",
        title_en="Training and Finetuning",
        title_zh="训练与微调",
        desc_en="Focus on training loops, loss functions, and lightweight adaptation.",
        desc_zh="聚焦训练循环、损失函数与轻量微调实现。",
    ),
    "05_rag_systems": TrackSpec(
        key="05_rag_systems",
        title_en="RAG Systems",
        title_zh="RAG 系统",
        desc_en="Practice retrieval pipelines, ranking, chunking, and search quality controls.",
        desc_zh="训练检索链路、排序、切分与召回质量控制。",
    ),
    "06_agents_tool_use": TrackSpec(
        key="06_agents_tool_use",
        title_en="Agents and Tool Use",
        title_zh="Agent 与 Tool Use",
        desc_en="Build the control loop that powers planner, memory, and tool-calling questions.",
        desc_zh="构建 Planner、Memory、Tool Calling 等常见 Agent 控制循环。",
    ),
    "07_ai_systems_engineering": TrackSpec(
        key="07_ai_systems_engineering",
        title_en="AI Systems Engineering",
        title_zh="AI 系统工程",
        desc_en="Cover scheduling, rate limiting, caching, and service APIs for production LLM systems.",
        desc_zh="覆盖生产级 LLM 系统中的调度、限流、缓存与服务接口。",
    ),
    "90_mock_interviews": TrackSpec(
        key="90_mock_interviews",
        title_en="Mock Interviews",
        title_zh="Mock Interviews",
        desc_en="Run timed mixed rounds that simulate real AI / LLM interview loops.",
        desc_zh="按真实 AI / LLM 面试轮次做限时混合模拟。",
    ),
}


CONCEPT_CARDS = {
    "01_transformer_fundamentals": {
        "en": [
            ("Why scale the attention scores?", "It stabilizes gradients by preventing large dot products from saturating softmax."),
            ("What is the role of the residual path?", "It preserves the identity signal and helps optimization through deep stacks."),
            ("Why does RoPE work well for decoding?", "It encodes relative position information directly into Q and K without extra embeddings."),
            ("Why is causal masking mandatory in decoder-only LMs?", "It prevents leakage from future tokens during both training and inference."),
        ],
        "zh": [
            ("为什么 attention score 要除以 sqrt(d_k)？", "为了防止点积过大导致 softmax 饱和，从而稳定梯度。"),
            ("Residual path 的核心作用是什么？", "保留恒等映射路径，让深层网络更容易优化。"),
            ("RoPE 为什么适合 decoder 推理？", "它把相对位置信息直接编码到 Q/K 中，不需要额外位置表。"),
            ("为什么 decoder-only LM 必须使用 causal mask？", "否则训练和推理都会看到未来 token，目标被泄漏。"),
        ],
    },
    "02_tokenization": {
        "en": [
            ("Why are subword tokenizers dominant?", "They balance vocabulary size, OOV robustness, and compressibility."),
            ("What is the practical difference between BPE and WordPiece?", "BPE optimizes merge frequency, while WordPiece often chooses merges by likelihood gain."),
            ("Why must encode/decode be reversible in a practice repo?", "A non-reversible tokenizer hides bugs in whitespace and special-token handling."),
            ("What is the point of the attention mask after padding?", "It lets the model ignore padded positions during attention and loss computation."),
        ],
        "zh": [
            ("为什么主流 tokenizer 都是 subword？", "它在词表大小、未登录词鲁棒性和压缩率之间取得了平衡。"),
            ("BPE 和 WordPiece 的实用差异是什么？", "BPE 更偏频次合并，WordPiece 更偏基于似然提升选择合并。"),
            ("为什么练习仓库里 encode/decode 必须可逆？", "否则空格、特殊符号和边界处理的 bug 很难暴露。"),
            ("padding 后为什么还需要 attention mask？", "让模型在注意力和 loss 计算时忽略补齐位置。"),
        ],
    },
    "03_inference_optimization": {
        "en": [
            ("What does KV cache save?", "It avoids recomputing historical keys and values for every generated token."),
            ("Why use top-p instead of only top-k?", "Top-p adapts the candidate set size to the shape of the distribution."),
            ("What problem does dynamic batching solve?", "It improves throughput by merging concurrent requests with similar decoding steps."),
            ("Why is prefix caching valuable for chat workloads?", "Many requests share a long prompt prefix, so the cache removes duplicated prefill cost."),
        ],
        "zh": [
            ("KV cache 主要节省了什么？", "避免每个新 token 都重复计算历史位置的 K/V。"),
            ("为什么需要 top-p 而不只用 top-k？", "top-p 会随分布形状自适应候选集合大小。"),
            ("dynamic batching 解决了什么问题？", "把并发请求合并解码，提升吞吐并减少空转。"),
            ("prefix caching 对对话场景为什么很有价值？", "大量请求共享长前缀，缓存后可省掉重复 prefill 成本。"),
        ],
    },
    "04_training_finetuning": {
        "en": [
            ("Why do we shift labels in causal LM training?", "Each token predicts the next token, so inputs and targets are offset by one position."),
            ("What does gradient clipping protect against?", "It prevents unstable updates when gradients spike."),
            ("Why is LoRA parameter-efficient?", "It learns low-rank adapters instead of updating the full dense weight."),
            ("When do you need gradient accumulation?", "When the desired effective batch size exceeds memory limits."),
        ],
        "zh": [
            ("为什么 causal LM 训练要 shift labels？", "每个位置都预测下一个 token，所以输入和目标要错开一位。"),
            ("gradient clipping 的主要作用是什么？", "防止梯度突然爆炸导致更新不稳定。"),
            ("LoRA 为什么参数高效？", "它只学习低秩增量，而不是更新整块权重。"),
            ("什么时候需要 gradient accumulation？", "当显存不够但又需要更大的等效 batch size 时。"),
        ],
    },
    "05_rag_systems": {
        "en": [
            ("Why does chunk overlap matter?", "It preserves context around chunk boundaries and reduces retrieval fragmentation."),
            ("What is MMR optimizing?", "It trades off relevance against redundancy when selecting documents."),
            ("Why add a reranker after vector search?", "Embedding recall is broad but often noisy, while rerankers improve ordering precision."),
            ("What does multi-query retrieval fix?", "It broadens recall when a single query phrasing misses relevant documents."),
        ],
        "zh": [
            ("为什么 chunk overlap 很重要？", "它保留边界上下文，减少关键信息被切断。"),
            ("MMR 优化的目标是什么？", "在相关性和多样性之间做平衡。"),
            ("为什么向量召回后还需要 reranker？", "向量召回覆盖广但排序不一定精确，reranker 用来提升排序质量。"),
            ("multi-query retrieval 主要修复什么问题？", "避免单一 query 表达覆盖不足导致漏召回。"),
        ],
    },
    "06_agents_tool_use": {
        "en": [
            ("What is the minimum loop for tool calling?", "Model decides tool -> tool executes -> observation returns to the model."),
            ("Why parse function schemas strictly?", "Loose parsing leads to brittle tool execution and silent failures."),
            ("What belongs in short-term memory?", "Only the local task state that should influence the next tool or reasoning step."),
            ("What makes ReAct debuggable?", "Thought, action, and observation are explicit and inspectable."),
        ],
        "zh": [
            ("tool calling 的最小闭环是什么？", "模型决策工具 -> 工具执行 -> 结果回注给模型。"),
            ("为什么 function schema 必须严格解析？", "解析宽松会让工具调用脆弱且容易静默失败。"),
            ("short-term memory 应该保存什么？", "只保存会影响下一步决策的局部任务状态。"),
            ("ReAct 为什么容易调试？", "Thought、Action、Observation 三段显式可观测。"),
        ],
    },
    "07_ai_systems_engineering": {
        "en": [
            ("What is the tradeoff in model-parallel scheduling?", "Higher utilization vs. more communication and placement complexity."),
            ("Why rate limit by tokens instead of requests?", "Request cost varies widely with prompt and generation lengths."),
            ("Why cache prompts and embeddings separately?", "They have different invalidation rules, cost profiles, and hit patterns."),
            ("What should a minimal LLM service API expose?", "A stable generation endpoint with clear input, config, and output shape."),
        ],
        "zh": [
            ("模型并行调度的核心 tradeoff 是什么？", "更高利用率与更复杂通信、放置策略之间的平衡。"),
            ("为什么要按 token 而不是请求数限流？", "不同请求的 prompt 和生成长度差异很大。"),
            ("为什么 prompt cache 和 embedding cache 要分开？", "它们的失效规则、成本结构和命中模式不同。"),
            ("最小可用 LLM 服务 API 应暴露什么？", "稳定的生成接口，以及清晰的输入、配置和输出格式。"),
        ],
    },
    "90_mock_interviews": {
        "en": [
            ("What matters most in a mock interview?", "A realistic time box plus a strict debrief on tradeoffs and edge cases."),
            ("How should you review a failed pack?", "Find the first incorrect assumption, not just the final bug."),
            ("What is the point of a scoring rubric?", "It separates coding speed from communication, testing, and system judgment."),
            ("How should you sequence a pack?", "Solve the medium-confidence problem first to build momentum, then tackle the hardest one."),
        ],
        "zh": [
            ("模拟面试最重要的是什么？", "真实限时和严格复盘，比单纯做对更重要。"),
            ("刷完一套 mock 后应该怎么复盘？", "先找第一个错误假设，而不是只盯最终 bug。"),
            ("为什么要有评分标准？", "把编码速度和表达、测试、系统判断分开评估。"),
            ("一套 mock 里如何安排做题顺序？", "先做中等把握题建立节奏，再处理最难题。"),
        ],
    },
}


PROBLEMS: list[ProblemSpec] = [
    ProblemSpec(
        pid="P001",
        order=1,
        track="01_transformer_fundamentals",
        slug="scaled_dot_product_attention",
        title_en="Implement scaled dot-product attention",
        title_zh="实现 scaled dot-product attention",
        difficulty=2,
        round_name="core",
        estimated_minutes=20,
        company_tags=("bytedance", "openai", "anthropic", "domestic_bigtech"),
        high_frequency=True,
        requirements_en=("Take Q, K, V as inputs.", "Support an optional mask.", "Return the attention output tensor."),
        requirements_zh=("输入 Q、K、V。", "支持可选 mask。", "返回 attention output。"),
        followups_en=("How would you avoid numerical instability in softmax?", "What changes for cross-attention?", "How do mask shapes broadcast across batches?"),
        followups_zh=("softmax 数值不稳定时怎么处理？", "如果改成 cross-attention，需要改什么？", "mask 在 batch 维上如何广播？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def scaled_dot_product_attention(q, k, v, mask=None):"},
        test_kind="scaled_dot_product_attention",
    ),
    ProblemSpec(
        pid="P002",
        order=2,
        track="01_transformer_fundamentals",
        slug="multi_head_attention",
        title_en="Implement multi-head attention",
        title_zh="实现 multi-head attention",
        difficulty=3,
        round_name="core",
        estimated_minutes=35,
        company_tags=("bytedance", "openai", "anthropic"),
        high_frequency=True,
        requirements_en=("Split hidden states into multiple heads.", "Concatenate the heads after attention.", "Apply an output projection."),
        requirements_zh=("支持多头拆分。", "attention 后 concat heads。", "做最终 linear projection。"),
        followups_en=("How do you validate hidden_size % num_heads?", "Why keep projection weights inside the module?", "Where would you place dropout?"),
        followups_zh=("hidden_size 不能整除 num_heads 时怎么办？", "为什么投影权重通常放在模块内部？", "dropout 通常插在哪里？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "MultiHeadAttention",
            "init_signature": "self, hidden_size: int, num_heads: int, bias: bool = True",
            "forward_signature": "self, x, mask=None",
        },
        test_kind="multi_head_attention",
    ),
    ProblemSpec(
        pid="P003",
        order=3,
        track="01_transformer_fundamentals",
        slug="self_attention_forward",
        title_en="Implement self-attention forward",
        title_zh="实现 self-attention forward",
        difficulty=3,
        round_name="core",
        estimated_minutes=25,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Accept x with shape [batch, seq_len, hidden].", "Project x into Q, K, V.", "Return the attention output."),
        requirements_zh=("输入 x 形状为 [batch, seq_len, hidden]。", "将 x 投影成 Q、K、V。", "返回 attention output。"),
        followups_en=("What is the difference from problem P002?", "Would you expose qkv projection separately?", "How do you handle attention masks?"),
        followups_zh=("它和 P002 的区别是什么？", "是否需要暴露 qkv projection？", "attention mask 应如何接入？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "SelfAttention",
            "init_signature": "self, hidden_size: int",
            "forward_signature": "self, x, mask=None",
        },
        test_kind="self_attention_forward",
    ),
    ProblemSpec(
        pid="P004",
        order=4,
        track="01_transformer_fundamentals",
        slug="transformer_block",
        title_en="Implement a Transformer block",
        title_zh="实现 Transformer block",
        difficulty=4,
        round_name="core",
        estimated_minutes=40,
        company_tags=("bytedance", "openai"),
        high_frequency=False,
        requirements_en=("Compose self-attention, residual connections, layer norm, and feed-forward layers.", "Preserve the input hidden size.", "Expose a forward pass."),
        requirements_zh=("组合 self-attention、residual、layernorm、feed-forward。", "保持输入输出 hidden size 一致。", "提供 forward。"),
        followups_en=("Would you choose pre-norm or post-norm?", "Where should dropout go?", "How would you benchmark activation memory?"),
        followups_zh=("你会选 pre-norm 还是 post-norm？", "dropout 通常放在哪里？", "如何估算激活显存？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "TransformerBlock",
            "init_signature": "self, hidden_size: int, num_heads: int, intermediate_size: int",
            "forward_signature": "self, x, mask=None",
        },
        test_kind="transformer_block",
    ),
    ProblemSpec(
        pid="P005",
        order=5,
        track="01_transformer_fundamentals",
        slug="positionwise_feedforward",
        title_en="Implement a position-wise feedforward layer",
        title_zh="实现 position-wise feedforward",
        difficulty=2,
        round_name="core",
        estimated_minutes=20,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Implement FFN(x) = max(0, xW1 + b1)W2 + b2.", "Process each position independently.", "Preserve batch and sequence dimensions."),
        requirements_zh=("实现 FFN(x) = max(0, xW1 + b1)W2 + b2。", "按位置独立计算。", "保持 batch 和 seq 维度。"),
        followups_en=("Why is the intermediate size typically larger?", "What changes with SwiGLU?", "Where does dropout fit in the FFN block?"),
        followups_zh=("为什么中间层通常更宽？", "如果换成 SwiGLU 要改什么？", "FFN 中 dropout 放哪里？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "PositionwiseFeedForward",
            "init_signature": "self, hidden_size: int, intermediate_size: int",
            "forward_signature": "self, x",
        },
        test_kind="positionwise_feedforward",
    ),
    ProblemSpec(
        pid="P006",
        order=6,
        track="01_transformer_fundamentals",
        slug="rope_position_embedding",
        title_en="Implement RoPE position embedding",
        title_zh="实现 RoPE position embedding",
        difficulty=3,
        round_name="core",
        estimated_minutes=30,
        company_tags=("openai", "anthropic"),
        high_frequency=False,
        requirements_en=("Implement rotary embedding.", "Apply the rotation to Q and K.", "Preserve the original tensor shape."),
        requirements_zh=("实现 rotary embedding。", "应用到 Q 和 K 上。", "保持原始 shape 不变。"),
        followups_en=("Why rotate half dimensions in pairs?", "How does RoPE support relative positions?", "What changes for grouped-query attention?"),
        followups_zh=("为什么要成对旋转维度？", "RoPE 如何体现相对位置信息？", "如果是 grouped-query attention 需要改什么？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def apply_rope(q, k, position_ids=None, base: float = 10000.0):"},
        test_kind="rope",
    ),
    ProblemSpec(
        pid="P007",
        order=7,
        track="01_transformer_fundamentals",
        slug="causal_mask",
        title_en="Implement a causal mask",
        title_zh="实现 causal mask",
        difficulty=1,
        round_name="core",
        estimated_minutes=10,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Generate a lower-triangular mask of shape [seq_len, seq_len].", "Mask future positions.", "Allow selecting the fill value."),
        requirements_zh=("生成 [seq_len, seq_len] 的下三角 mask。", "屏蔽未来位置。", "允许指定 mask fill value。"),
        followups_en=("Should the mask use booleans or additive logits?", "How do you broadcast it over heads?", "What changes for prefix-LM masking?"),
        followups_zh=("mask 应该用 bool 还是 additive logits？", "如何扩展到多头维度？", "如果是 prefix-LM mask 要怎么改？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def causal_mask(seq_len: int, fill_value: float = float('-inf')):"},
        test_kind="causal_mask",
    ),
    ProblemSpec(
        pid="P008",
        order=8,
        track="01_transformer_fundamentals",
        slug="layer_norm",
        title_en="Implement LayerNorm without torch.nn.LayerNorm",
        title_zh="不调用现成 API 实现 LayerNorm",
        difficulty=2,
        round_name="core",
        estimated_minutes=20,
        company_tags=("bytedance", "openai"),
        high_frequency=False,
        requirements_en=("Normalize over the last hidden dimension.", "Apply learnable gamma and beta.", "Do not call the built-in LayerNorm module."),
        requirements_zh=("沿最后一维做归一化。", "支持可学习的 gamma 和 beta。", "不能调用内置 LayerNorm。"),
        followups_en=("Why add epsilon inside the sqrt?", "How does RMSNorm differ?", "Which dimension should be normalized?"),
        followups_zh=("为什么 epsilon 要放在开方内部？", "RMSNorm 和 LayerNorm 的区别是什么？", "应沿哪一维归一化？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def layer_norm(x, gamma, beta, eps: float = 1e-5):"},
        test_kind="layer_norm",
    ),
    ProblemSpec(
        pid="P009",
        order=9,
        track="01_transformer_fundamentals",
        slug="transformer_decoder_layer",
        title_en="Implement a Transformer decoder layer",
        title_zh="实现 Transformer decoder layer",
        difficulty=4,
        round_name="core",
        estimated_minutes=40,
        company_tags=("openai", "anthropic", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Include causal attention and an FFN.", "Support a causal mask.", "Expose a forward pass over decoder hidden states."),
        requirements_zh=("包含 causal attention 和 FFN。", "支持 causal mask。", "提供 decoder hidden states 的 forward。"),
        followups_en=("Would you fuse QKV projection?", "Where does KV cache plug in later?", "How does decoder-only differ from encoder-decoder blocks?"),
        followups_zh=("QKV projection 是否应该融合？", "后续 KV cache 应该接在哪里？", "decoder-only 和 encoder-decoder block 的区别是什么？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "TransformerDecoderLayer",
            "init_signature": "self, hidden_size: int, num_heads: int, intermediate_size: int",
            "forward_signature": "self, x, mask=None",
        },
        test_kind="transformer_decoder_layer",
    ),
    ProblemSpec(
        pid="P010",
        order=10,
        track="01_transformer_fundamentals",
        slug="mini_transformer_lm",
        title_en="Implement a mini Transformer language model from scratch",
        title_zh="从零实现 mini Transformer language model",
        difficulty=4,
        round_name="core",
        estimated_minutes=55,
        company_tags=("bytedance", "openai", "anthropic"),
        high_frequency=True,
        requirements_en=("Support token embeddings.", "Add at least one attention block.", "Return vocabulary logits through an LM head."),
        requirements_zh=("支持 token embedding。", "至少包含一层 attention block。", "通过 LM head 返回词表 logits。"),
        followups_en=("How would you tie token embedding and LM head weights?", "What changes when adding RoPE?", "How do you compute the causal LM loss on top?"),
        followups_zh=("如何做 embedding 和 LM head 权重共享？", "如果加入 RoPE 要改哪里？", "其上如何计算 causal LM loss？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "MiniTransformerLM",
            "init_signature": "self, vocab_size: int, hidden_size: int, num_heads: int, intermediate_size: int",
            "forward_signature": "self, input_ids",
        },
        test_kind="mini_transformer_lm",
    ),
    ProblemSpec(
        pid="P011",
        order=11,
        track="02_tokenization",
        slug="bpe_tokenizer",
        title_en="Implement a BPE tokenizer",
        title_zh="实现 BPE tokenizer",
        difficulty=3,
        round_name="tokenization",
        estimated_minutes=35,
        company_tags=("bytedance", "openai", "domestic_bigtech"),
        high_frequency=True,
        requirements_en=("Train merge pairs from a corpus.", "Update the vocabulary after each merge.", "Expose encode and decode."),
        requirements_zh=("从语料训练 merge pair。", "每次 merge 后更新词表。", "提供 encode 和 decode。"),
        followups_en=("How do you represent end-of-word markers?", "What happens when merges conflict with whitespace?", "How would you store merges efficiently?"),
        followups_zh=("词尾标记通常怎么表示？", "merge 和空格边界冲突时怎么办？", "如何高效存储 merges？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "BPETokenizer",
            "methods": (
                ("__init__", "self, vocab_size: int"),
                ("train", "self, corpus"),
                ("encode", "self, text: str"),
                ("decode", "self, tokens"),
            ),
        },
        test_kind="bpe_tokenizer",
    ),
    ProblemSpec(
        pid="P012",
        order=12,
        track="02_tokenization",
        slug="wordpiece_tokenizer",
        title_en="Implement a WordPiece tokenizer",
        title_zh="实现 WordPiece tokenizer",
        difficulty=3,
        round_name="tokenization",
        estimated_minutes=35,
        company_tags=("bytedance", "openai"),
        high_frequency=False,
        requirements_en=("Support greedy longest-match segmentation.", "Expose encode and decode.", "Handle unknown pieces explicitly."),
        requirements_zh=("支持 greedy longest match。", "提供 encode 和 decode。", "显式处理 unknown piece。"),
        followups_en=("How does the continuation prefix work?", "What is the OOV fallback policy?", "How does training differ from BPE?"),
        followups_zh=("continuation prefix 的作用是什么？", "OOV 的回退策略是什么？", "它和 BPE 的训练区别是什么？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "WordPieceTokenizer",
            "methods": (
                ("__init__", "self, vocab"),
                ("encode", "self, text: str"),
                ("decode", "self, tokens"),
            ),
        },
        test_kind="wordpiece_tokenizer",
    ),
    ProblemSpec(
        pid="P013",
        order=13,
        track="02_tokenization",
        slug="train_subword_vocab",
        title_en="Implement subword tokenizer training",
        title_zh="实现 subword tokenizer 训练",
        difficulty=2,
        round_name="tokenization",
        estimated_minutes=25,
        company_tags=("domestic_bigtech",),
        high_frequency=False,
        requirements_en=("Input a corpus.", "Return a vocabulary of the requested size.", "Keep the training loop deterministic."),
        requirements_zh=("输入 corpus。", "输出指定大小的 vocab。", "保持训练过程可复现。"),
        followups_en=("What statistics do you maintain while training?", "How do you stop when no merge improves quality?", "Should special tokens count toward vocab size?"),
        followups_zh=("训练时要维护哪些统计量？", "如果没有 merge 继续提升质量，何时停止？", "special tokens 是否计入 vocab size？"),
        starter={"kind": "function", "imports": (), "signature": "def train_subword_vocab(corpus, vocab_size: int):"},
        test_kind="train_subword_vocab",
    ),
    ProblemSpec(
        pid="P014",
        order=14,
        track="02_tokenization",
        slug="tokenize_detokenize",
        title_en="Implement tokenize + detokenize",
        title_zh="实现 tokenize + detokenize",
        difficulty=2,
        round_name="tokenization",
        estimated_minutes=20,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Ensure decode(encode(text)) == text.", "Support fitting on a small corpus.", "Return integer token ids."),
        requirements_zh=("保证 decode(encode(text)) == text。", "支持在小语料上 fit。", "输出整数 token id。"),
        followups_en=("How do you preserve whitespace exactly?", "What special tokens would you reserve?", "How do you serialize the vocab?"),
        followups_zh=("如何精确保留空格？", "你会预留哪些 special tokens？", "词表如何序列化？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "RoundTripTokenizer",
            "methods": (
                ("__init__", "self"),
                ("fit", "self, corpus"),
                ("encode", "self, text: str"),
                ("decode", "self, token_ids"),
            ),
        },
        test_kind="tokenize_detokenize",
    ),
    ProblemSpec(
        pid="P015",
        order=15,
        track="02_tokenization",
        slug="padding_and_attention_mask",
        title_en="Implement padding and attention mask creation",
        title_zh="实现 padding + attention mask",
        difficulty=1,
        round_name="tokenization",
        estimated_minutes=15,
        company_tags=("domestic_bigtech",),
        high_frequency=False,
        requirements_en=("Pad sequences to a shared length.", "Return an attention mask.", "Keep padding token configurable."),
        requirements_zh=("把序列 pad 到统一长度。", "返回 attention mask。", "padding token 可配置。"),
        followups_en=("Should the mask use 1 for valid tokens or 0?", "How does left-padding change decoding?", "What if the input is already rectangular?"),
        followups_zh=("mask 中有效 token 应该用 1 还是 0？", "left-padding 对解码有什么影响？", "如果输入本来就是等长的怎么办？"),
        starter={"kind": "function", "imports": (), "signature": "def pad_sequences(sequences, pad_token_id: int = 0):"},
        test_kind="padding_and_attention_mask",
    ),
    ProblemSpec(
        pid="P016",
        order=16,
        track="03_inference_optimization",
        slug="kv_cache",
        title_en="Implement a KV cache",
        title_zh="实现 KV cache",
        difficulty=2,
        round_name="serving",
        estimated_minutes=25,
        company_tags=("openai", "anthropic", "bytedance"),
        high_frequency=True,
        requirements_en=("Store keys and values.", "Support incremental appends.", "Return the concatenated cache state."),
        requirements_zh=("保存 K/V。", "支持增量追加。", "返回拼接后的 cache 状态。"),
        followups_en=("What is the cache shape for multi-head attention?", "How do you reset between requests?", "How does paged attention change this interface?"),
        followups_zh=("多头 attention 的 cache shape 应该是什么？", "请求结束后如何 reset？", "如果是 paged attention，这个接口怎么变？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "KVCache",
            "methods": (
                ("__init__", "self"),
                ("append", "self, key, value"),
                ("get", "self"),
                ("reset", "self"),
            ),
        },
        test_kind="kv_cache",
    ),
    ProblemSpec(
        pid="P017",
        order=17,
        track="03_inference_optimization",
        slug="greedy_decoding",
        title_en="Implement greedy decoding",
        title_zh="实现 greedy decoding",
        difficulty=1,
        round_name="serving",
        estimated_minutes=15,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Take logits as input.", "Return the argmax token id.", "Handle batched or flat logits."),
        requirements_zh=("输入 logits。", "返回 argmax token id。", "兼容单条或 batch logits。"),
        followups_en=("Why is greedy decoding deterministic?", "When does greedy decoding fail badly?", "How do you stop at EOS?"),
        followups_zh=("greedy decoding 为什么是确定性的？", "它在哪些场景效果很差？", "如何在 EOS 处停止？"),
        starter={"kind": "function", "imports": ("import math",), "signature": "def greedy_decode(logits):"},
        test_kind="greedy_decoding",
    ),
    ProblemSpec(
        pid="P018",
        order=18,
        track="03_inference_optimization",
        slug="top_k_sampling",
        title_en="Implement top-k sampling",
        title_zh="实现 top-k sampling",
        difficulty=1,
        round_name="serving",
        estimated_minutes=15,
        company_tags=("bytedance", "openai"),
        high_frequency=True,
        requirements_en=("Take logits and k.", "Restrict sampling to the top-k tokens.", "Return one sampled token id."),
        requirements_zh=("输入 logits 和 k。", "只在 top-k 候选中采样。", "返回一个 token id。"),
        followups_en=("How do you handle k larger than vocab size?", "Should you renormalize probabilities?", "What if logits contain ties?"),
        followups_zh=("如果 k 大于词表大小怎么办？", "是否需要重新归一化概率？", "如果 logits 有并列值怎么办？"),
        starter={"kind": "function", "imports": ("import random",), "signature": "def top_k_sample(logits, k: int):"},
        test_kind="top_k_sampling",
    ),
    ProblemSpec(
        pid="P019",
        order=19,
        track="03_inference_optimization",
        slug="top_p_sampling",
        title_en="Implement top-p sampling",
        title_zh="实现 top-p sampling",
        difficulty=2,
        round_name="serving",
        estimated_minutes=20,
        company_tags=("openai", "anthropic"),
        high_frequency=True,
        requirements_en=("Use nucleus sampling.", "Build the smallest candidate set whose cumulative probability exceeds p.", "Return one sampled token id."),
        requirements_zh=("实现 nucleus sampling。", "构造累计概率超过 p 的最小候选集合。", "返回一个 token id。"),
        followups_en=("Why sort by probability first?", "How do you guard against p <= 0?", "What happens on a very peaked distribution?"),
        followups_zh=("为什么必须先按概率排序？", "p <= 0 时怎么处理？", "如果分布极尖锐，会发生什么？"),
        starter={"kind": "function", "imports": ("import random",), "signature": "def top_p_sample(logits, p: float):"},
        test_kind="top_p_sampling",
    ),
    ProblemSpec(
        pid="P020",
        order=20,
        track="03_inference_optimization",
        slug="temperature_sampling",
        title_en="Implement temperature sampling",
        title_zh="实现 temperature sampling",
        difficulty=1,
        round_name="serving",
        estimated_minutes=15,
        company_tags=("domestic_bigtech",),
        high_frequency=False,
        requirements_en=("Scale logits by temperature.", "Sample from the adjusted distribution.", "Support temperature > 0."),
        requirements_zh=("按 temperature 缩放 logits。", "从调整后的分布中采样。", "支持 temperature > 0。"),
        followups_en=("Why does lower temperature sharpen the distribution?", "What should happen at temperature=0?", "How do you combine this with top-k?"),
        followups_zh=("为什么低 temperature 会让分布更尖锐？", "temperature=0 时应该怎么处理？", "如何和 top-k 组合？"),
        starter={"kind": "function", "imports": ("import random",), "signature": "def temperature_sample(logits, temperature: float = 1.0):"},
        test_kind="temperature_sampling",
    ),
    ProblemSpec(
        pid="P021",
        order=21,
        track="03_inference_optimization",
        slug="beam_search",
        title_en="Implement beam search",
        title_zh="实现 beam search",
        difficulty=3,
        round_name="serving",
        estimated_minutes=35,
        company_tags=("openai", "bytedance"),
        high_frequency=True,
        requirements_en=("Support beam size.", "Accumulate log probabilities.", "Return the highest-scoring sequence."),
        requirements_zh=("支持 beam size。", "累计 log probability。", "返回得分最高的序列。"),
        followups_en=("How do you stop early on EOS?", "Should you length-normalize scores?", "What is the complexity vs greedy decoding?"),
        followups_zh=("遇到 EOS 如何提前停止？", "是否要做长度归一化？", "和 greedy decoding 相比复杂度如何？"),
        starter={"kind": "function", "imports": ("import math",), "signature": "def beam_search(step_fn, start_tokens, beam_size: int, max_new_tokens: int, eos_token_id=None):"},
        test_kind="beam_search",
    ),
    ProblemSpec(
        pid="P022",
        order=22,
        track="03_inference_optimization",
        slug="logits_processor",
        title_en="Implement a logits processor with repetition penalty",
        title_zh="实现支持 repetition penalty 的 logits processor",
        difficulty=2,
        round_name="serving",
        estimated_minutes=20,
        company_tags=("anthropic", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Accept logits and generated token ids.", "Apply repetition penalty to repeated ids.", "Return adjusted logits."),
        requirements_zh=("输入 logits 和已生成 token ids。", "对重复 token 应用 repetition penalty。", "返回调整后的 logits。"),
        followups_en=("Should negative logits be multiplied or divided?", "How do you combine multiple logits processors?", "Should EOS receive the penalty?"),
        followups_zh=("负 logits 应该乘还是除？", "多个 logits processor 如何组合？", "EOS 要不要被惩罚？"),
        starter={"kind": "function", "imports": (), "signature": "def apply_repetition_penalty(logits, generated_ids, penalty: float):"},
        test_kind="logits_processor",
    ),
    ProblemSpec(
        pid="P023",
        order=23,
        track="03_inference_optimization",
        slug="dynamic_batching",
        title_en="Implement dynamic batching",
        title_zh="实现 dynamic batching",
        difficulty=3,
        round_name="serving",
        estimated_minutes=35,
        company_tags=("openai", "anthropic"),
        high_frequency=False,
        requirements_en=("Merge concurrent generation requests.", "Support enqueue and flush.", "Preserve request ordering in the emitted batch."),
        requirements_zh=("合并并发生成请求。", "支持 enqueue 和 flush。", "输出 batch 时保留请求顺序。"),
        followups_en=("How would you batch by decode step?", "What is the latency-throughput tradeoff?", "How do you handle finished sequences inside a batch?"),
        followups_zh=("如何按 decode step 分桶？", "延迟和吞吐的 tradeoff 是什么？", "batch 内已经完成的序列如何处理？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "DynamicBatcher",
            "methods": (
                ("__init__", "self, max_batch_size: int"),
                ("enqueue", "self, request"),
                ("flush", "self"),
            ),
        },
        test_kind="dynamic_batching",
    ),
    ProblemSpec(
        pid="P024",
        order=24,
        track="03_inference_optimization",
        slug="prefix_caching",
        title_en="Implement prefix caching",
        title_zh="实现 prefix caching",
        difficulty=3,
        round_name="serving",
        estimated_minutes=30,
        company_tags=("anthropic", "openai"),
        high_frequency=False,
        requirements_en=("Reuse cached state for shared prompt prefixes.", "Allow inserting and looking up prefixes.", "Return cache hits with the longest matching prefix."),
        requirements_zh=("复用共享 prompt 前缀的缓存。", "支持插入和查找前缀。", "命中时返回最长匹配前缀。"),
        followups_en=("How would you hash token prefixes?", "What invalidates a prefix cache entry?", "How do you bound memory growth?"),
        followups_zh=("token 前缀通常如何做 hash？", "prefix cache 何时失效？", "如何控制内存增长？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "PrefixCache",
            "methods": (
                ("__init__", "self"),
                ("add_prefix", "self, prefix_tokens, value"),
                ("lookup", "self, prefix_tokens"),
            ),
        },
        test_kind="prefix_caching",
    ),
    ProblemSpec(
        pid="P025",
        order=25,
        track="03_inference_optimization",
        slug="streaming_generation",
        title_en="Implement streaming generation",
        title_zh="实现 streaming generation",
        difficulty=2,
        round_name="serving",
        estimated_minutes=25,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Yield tokens one by one.", "Stop at max_new_tokens or EOS.", "Expose a generator-style interface."),
        requirements_zh=("逐 token 输出。", "在 max_new_tokens 或 EOS 处停止。", "提供生成器式接口。"),
        followups_en=("How would you surface partial text to a client?", "What is backpressure in streaming APIs?", "How do you test streamed outputs deterministically?"),
        followups_zh=("如何把 partial text 返回给客户端？", "流式 API 的 backpressure 是什么？", "如何稳定测试 streaming 输出？"),
        starter={"kind": "function", "imports": (), "signature": "def stream_generate(step_fn, prompt_ids, max_new_tokens: int, eos_token_id=None):"},
        test_kind="streaming_generation",
    ),
    ProblemSpec(
        pid="P026",
        order=26,
        track="04_training_finetuning",
        slug="cross_entropy_loss",
        title_en="Implement cross entropy loss",
        title_zh="实现 cross entropy loss",
        difficulty=2,
        round_name="training",
        estimated_minutes=20,
        company_tags=("bytedance", "openai"),
        high_frequency=False,
        requirements_en=("Compute cross entropy from logits and labels.", "Average over the batch.", "Avoid using torch.nn.CrossEntropyLoss directly."),
        requirements_zh=("从 logits 和 labels 计算 cross entropy。", "对 batch 求平均。", "不能直接调用现成 CrossEntropyLoss。"),
        followups_en=("Why subtract the max logit before exponentiating?", "How do you support ignore_index?", "What is label smoothing?"),
        followups_zh=("为什么 exp 前要减去最大 logit？", "如何支持 ignore_index？", "什么是 label smoothing？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def cross_entropy_loss(logits, targets):"},
        test_kind="cross_entropy_loss",
    ),
    ProblemSpec(
        pid="P027",
        order=27,
        track="04_training_finetuning",
        slug="causal_lm_training_loss",
        title_en="Implement causal LM training loss",
        title_zh="实现 causal LM training loss",
        difficulty=2,
        round_name="training",
        estimated_minutes=20,
        company_tags=("domestic_bigtech", "bytedance"),
        high_frequency=False,
        requirements_en=("Shift labels by one position.", "Ignore masked labels.", "Return a scalar loss."),
        requirements_zh=("将 labels 向右错一位。", "忽略被 mask 的标签。", "返回标量 loss。"),
        followups_en=("Should the last position contribute to loss?", "Where does ignore_index apply?", "How do you flatten logits efficiently?"),
        followups_zh=("最后一个位置是否参与 loss？", "ignore_index 应作用在哪一步？", "如何高效展平 logits？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def causal_lm_loss(logits, labels, ignore_index: int = -100):"},
        test_kind="causal_lm_training_loss",
    ),
    ProblemSpec(
        pid="P028",
        order=28,
        track="04_training_finetuning",
        slug="gradient_clipping",
        title_en="Implement gradient clipping",
        title_zh="实现 gradient clipping",
        difficulty=1,
        round_name="training",
        estimated_minutes=15,
        company_tags=("domestic_bigtech",),
        high_frequency=False,
        requirements_en=("Clip gradients to a max norm.", "Handle parameters without gradients.", "Return the pre-clipped norm."),
        requirements_zh=("按 max norm 截断梯度。", "兼容没有梯度的参数。", "返回截断前的梯度范数。"),
        followups_en=("Why clip by global norm instead of per-tensor norm?", "What happens when the norm is already below the threshold?", "Where do you place clipping in the training step?"),
        followups_zh=("为什么通常按全局 norm 而不是逐 tensor clip？", "如果 norm 已经低于阈值会怎样？", "clip 应放在训练循环的哪一步？"),
        starter={"kind": "function", "imports": ("import math",), "signature": "def clip_gradients(parameters, max_norm: float):"},
        test_kind="gradient_clipping",
    ),
    ProblemSpec(
        pid="P029",
        order=29,
        track="04_training_finetuning",
        slug="lora_forward",
        title_en="Implement LoRA forward",
        title_zh="实现 LoRA forward",
        difficulty=3,
        round_name="training",
        estimated_minutes=30,
        company_tags=("bytedance", "openai", "anthropic"),
        high_frequency=True,
        requirements_en=("Implement W' = W + BA.", "Expose a forward path over the adapted weight.", "Support scaling by alpha / rank."),
        requirements_zh=("实现 W' = W + BA。", "在 forward 中使用增量权重。", "支持 alpha / rank 缩放。"),
        followups_en=("Why is the adapter initialized asymmetrically?", "How do you freeze the base weight?", "Where does dropout fit into LoRA?"),
        followups_zh=("为什么 LoRA 常用不对称初始化？", "base weight 如何冻结？", "LoRA dropout 应该放在哪里？"),
        starter={
            "kind": "nn_module",
            "imports": ("import torch", "from torch import nn"),
            "class_name": "LoRALinear",
            "init_signature": "self, in_features: int, out_features: int, rank: int, alpha: float = 1.0, bias: bool = False",
            "forward_signature": "self, x",
        },
        test_kind="lora_forward",
    ),
    ProblemSpec(
        pid="P030",
        order=30,
        track="04_training_finetuning",
        slug="lora_merge",
        title_en="Implement LoRA merge",
        title_zh="实现 LoRA merge",
        difficulty=2,
        round_name="training",
        estimated_minutes=20,
        company_tags=("anthropic", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Merge LoRA weights into the base weight.", "Apply alpha / rank scaling.", "Return the merged dense weight."),
        requirements_zh=("把 LoRA 权重合并到 base weight。", "应用 alpha / rank 缩放。", "返回合并后的 dense weight。"),
        followups_en=("When is merge useful in serving?", "Should merge be reversible?", "How do you validate shape compatibility?"),
        followups_zh=("merge 对部署有什么帮助？", "merge 需要可逆吗？", "如何验证 shape 是否兼容？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def merge_lora_weights(base_weight, lora_a, lora_b, alpha: float = 1.0):"},
        test_kind="lora_merge",
    ),
    ProblemSpec(
        pid="P031",
        order=31,
        track="04_training_finetuning",
        slug="parameter_freezing",
        title_en="Implement parameter freezing",
        title_zh="实现 parameter freezing",
        difficulty=1,
        round_name="training",
        estimated_minutes=15,
        company_tags=("domestic_bigtech",),
        high_frequency=False,
        requirements_en=("Freeze parameters that do not match the provided patterns.", "Return the trainable parameter names.", "Leave matching parameters trainable."),
        requirements_zh=("冻结不匹配给定模式的参数。", "返回可训练参数名。", "匹配到的参数保持可训练。"),
        followups_en=("Would you match exact names or prefixes?", "Why freeze embeddings or lower layers?", "How do you verify optimizer coverage afterward?"),
        followups_zh=("你会匹配精确名字还是前缀？", "为什么会冻结 embedding 或底层层？", "如何验证 optimizer 覆盖了正确参数？"),
        starter={"kind": "function", "imports": (), "signature": "def freeze_parameters(model, trainable_patterns):"},
        test_kind="parameter_freezing",
    ),
    ProblemSpec(
        pid="P032",
        order=32,
        track="04_training_finetuning",
        slug="sft_training_loop",
        title_en="Implement one SFT training step",
        title_zh="实现 SFT 训练循环",
        difficulty=3,
        round_name="training",
        estimated_minutes=30,
        company_tags=("bytedance", "openai"),
        high_frequency=False,
        requirements_en=("Run forward, loss, backward, and optimizer step.", "Zero gradients safely.", "Return the scalar loss value."),
        requirements_zh=("包含 forward、loss、backward、optimizer step。", "正确清理梯度。", "返回标量 loss。"),
        followups_en=("Where should gradient clipping happen?", "What would you log for debugging?", "How do you switch between train and eval modes?"),
        followups_zh=("gradient clipping 应该放哪？", "为了调试你会记录什么？", "train/eval 模式如何切换？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def sft_train_step(model, batch, optimizer):"},
        test_kind="sft_training_loop",
    ),
    ProblemSpec(
        pid="P033",
        order=33,
        track="04_training_finetuning",
        slug="gradient_accumulation",
        title_en="Implement gradient accumulation",
        title_zh="实现 gradient accumulation",
        difficulty=2,
        round_name="training",
        estimated_minutes=25,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Accumulate gradients across multiple micro-batches.", "Step the optimizer every accumulation_steps.", "Return the number of optimizer steps taken."),
        requirements_zh=("跨多个 micro-batch 累积梯度。", "每 accumulation_steps 次做一次 optimizer step。", "返回 optimizer step 次数。"),
        followups_en=("Should you divide the loss before backward?", "How do you handle a remainder batch count?", "How does this interact with gradient clipping?"),
        followups_zh=("backward 前是否需要缩放 loss？", "不能整除的尾 batch 如何处理？", "它和 gradient clipping 如何协作？"),
        starter={"kind": "function", "imports": ("import torch",), "signature": "def run_gradient_accumulation(model, batches, optimizer, accumulation_steps: int):"},
        test_kind="gradient_accumulation",
    ),
    ProblemSpec(
        pid="P034",
        order=34,
        track="05_rag_systems",
        slug="document_chunking",
        title_en="Implement document chunking",
        title_zh="实现 document chunking",
        difficulty=1,
        round_name="retrieval",
        estimated_minutes=15,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Support chunk_size and overlap.", "Return chunks in order.", "Do not drop the tail content."),
        requirements_zh=("支持 chunk_size 和 overlap。", "按顺序返回 chunk。", "不能丢掉结尾内容。"),
        followups_en=("What is a good unit: characters, tokens, or sentences?", "How do you keep overlap from duplicating too much content?", "When should you chunk by structure instead?"),
        followups_zh=("chunk 的单位该是字符、token 还是句子？", "如何避免 overlap 带来过多重复？", "什么时候应该按文档结构切分？"),
        starter={"kind": "function", "imports": (), "signature": "def chunk_document(text: str, chunk_size: int, overlap: int = 0):"},
        test_kind="document_chunking",
    ),
    ProblemSpec(
        pid="P035",
        order=35,
        track="05_rag_systems",
        slug="embedding_vector_store",
        title_en="Implement embedding + vector store",
        title_zh="实现 embedding + vector store",
        difficulty=3,
        round_name="retrieval",
        estimated_minutes=35,
        company_tags=("openai", "anthropic"),
        high_frequency=False,
        requirements_en=("Expose add_document and search.", "Store embeddings in memory.", "Return top-k results."),
        requirements_zh=("提供 add_document 和 search。", "在内存中存 embedding。", "返回 top-k 结果。"),
        followups_en=("How do you inject an embedder for testability?", "What metadata should you store per vector?", "How do you batch embedding calls?"),
        followups_zh=("如何注入 embedder 以便测试？", "每个向量需要存什么 metadata？", "embedding 调用如何批处理？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "InMemoryVectorStore",
            "methods": (
                ("__init__", "self, embedder"),
                ("add_document", "self, doc_id: str, text: str"),
                ("search", "self, query: str, top_k: int = 3"),
            ),
        },
        test_kind="embedding_vector_store",
    ),
    ProblemSpec(
        pid="P036",
        order=36,
        track="05_rag_systems",
        slug="cosine_similarity_search",
        title_en="Implement cosine similarity search",
        title_zh="实现 cosine similarity search",
        difficulty=2,
        round_name="retrieval",
        estimated_minutes=20,
        company_tags=("openai", "domestic_bigtech"),
        high_frequency=True,
        requirements_en=("Compare a query embedding with document embeddings.", "Return the most similar document ids.", "Sort results by score descending."),
        requirements_zh=("比较 query embedding 和文档 embedding。", "返回最相似的文档 id。", "按分数降序排序。"),
        followups_en=("Why normalize before cosine similarity?", "How do you handle zero vectors?", "What is the complexity of brute-force search?"),
        followups_zh=("为什么要先归一化再算 cosine similarity？", "零向量怎么处理？", "暴力检索复杂度是多少？"),
        starter={"kind": "function", "imports": ("import math",), "signature": "def cosine_similarity_search(query_embedding, doc_embeddings, top_k: int = 3):"},
        test_kind="cosine_similarity_search",
    ),
    ProblemSpec(
        pid="P037",
        order=37,
        track="05_rag_systems",
        slug="simple_rag_pipeline",
        title_en="Implement a simple RAG pipeline",
        title_zh="实现简单 RAG pipeline",
        difficulty=4,
        round_name="retrieval",
        estimated_minutes=40,
        company_tags=("bytedance", "openai", "anthropic"),
        high_frequency=True,
        requirements_en=("Run query -> retrieve -> prompt -> LLM.", "Return the final generated answer.", "Keep the components injectable."),
        requirements_zh=("实现 query -> retrieve -> prompt -> LLM。", "返回最终答案。", "各组件应支持注入替换。"),
        followups_en=("Where should citations be inserted?", "How do you handle empty retrieval results?", "What prompt fields matter most?"),
        followups_zh=("引用信息应该插在哪里？", "检索为空时怎么处理？", "prompt 中最关键的字段是什么？"),
        starter={"kind": "function", "imports": (), "signature": "def run_simple_rag(query: str, retriever, llm, prompt_template: str):"},
        test_kind="simple_rag_pipeline",
    ),
    ProblemSpec(
        pid="P038",
        order=38,
        track="05_rag_systems",
        slug="mmr_retrieval",
        title_en="Implement MMR retrieval",
        title_zh="实现 MMR retrieval",
        difficulty=3,
        round_name="retrieval",
        estimated_minutes=30,
        company_tags=("openai", "anthropic"),
        high_frequency=False,
        requirements_en=("Implement maximal marginal relevance selection.", "Balance similarity to the query and diversity across docs.", "Return selected indices."),
        requirements_zh=("实现 maximal marginal relevance。", "平衡 query 相关性和文档多样性。", "返回被选中的索引。"),
        followups_en=("How do you choose lambda?", "Should the first selected document be pure relevance?", "How do you scale MMR on large candidate sets?"),
        followups_zh=("lambda 应该怎么选？", "第一个文档是否应只看相关性？", "大候选集合下如何扩展 MMR？"),
        starter={"kind": "function", "imports": ("import math",), "signature": "def mmr_select(query_embedding, doc_embeddings, lambda_mult: float = 0.5, top_k: int = 3):"},
        test_kind="mmr_retrieval",
    ),
    ProblemSpec(
        pid="P039",
        order=39,
        track="05_rag_systems",
        slug="reranker",
        title_en="Implement a reranker",
        title_zh="实现 reranker",
        difficulty=2,
        round_name="retrieval",
        estimated_minutes=20,
        company_tags=("domestic_bigtech", "bytedance"),
        high_frequency=False,
        requirements_en=("Take a query and documents.", "Score each document.", "Return documents sorted by score."),
        requirements_zh=("输入 query 和 docs。", "给每个文档打分。", "返回排序后的 docs。"),
        followups_en=("Should the scorer return calibrated probabilities?", "How do you preserve the original doc ids?", "How do you use a cross-encoder here?"),
        followups_zh=("scorer 需要返回校准概率吗？", "如何保留原始文档 id？", "cross-encoder 在这里怎么接？"),
        starter={"kind": "function", "imports": (), "signature": "def rerank_documents(query: str, docs, scorer):"},
        test_kind="reranker",
    ),
    ProblemSpec(
        pid="P040",
        order=40,
        track="05_rag_systems",
        slug="multi_query_retrieval",
        title_en="Implement multi-query retrieval",
        title_zh="实现 multi-query retrieval",
        difficulty=3,
        round_name="retrieval",
        estimated_minutes=30,
        company_tags=("openai", "anthropic", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Generate multiple query rewrites.", "Retrieve for each rewrite.", "Merge the results and deduplicate."),
        requirements_zh=("生成多个 query 变体。", "对每个 query 做检索。", "合并结果并去重。"),
        followups_en=("How many rewrites are enough?", "Where do you deduplicate: by id or by text?", "How do you score merged candidates?"),
        followups_zh=("生成多少个 query 变体合适？", "去重应该按 id 还是按文本？", "合并候选后如何重新打分？"),
        starter={"kind": "function", "imports": (), "signature": "def multi_query_retrieve(query: str, query_generator, retriever, top_k: int = 3):"},
        test_kind="multi_query_retrieval",
    ),
    ProblemSpec(
        pid="P041",
        order=41,
        track="06_agents_tool_use",
        slug="tool_calling_agent",
        title_en="Implement a tool-calling agent",
        title_zh="实现 tool calling agent",
        difficulty=4,
        round_name="agent",
        estimated_minutes=40,
        company_tags=("openai", "anthropic", "bytedance"),
        high_frequency=False,
        requirements_en=("Run the loop LLM -> tool -> result -> LLM.", "Support named tools.", "Return the final answer string."),
        requirements_zh=("实现 LLM -> tool -> result -> LLM 闭环。", "支持具名工具。", "返回最终答案字符串。"),
        followups_en=("Where do you cap tool recursion?", "How do you validate tool arguments?", "How would you log every tool call?"),
        followups_zh=("如何限制工具递归调用次数？", "如何校验工具参数？", "怎样记录每次 tool call？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "ToolCallingAgent",
            "methods": (
                ("__init__", "self, llm, tools"),
                ("run", "self, user_query: str"),
            ),
        },
        test_kind="tool_calling_agent",
    ),
    ProblemSpec(
        pid="P042",
        order=42,
        track="06_agents_tool_use",
        slug="function_schema_parsing",
        title_en="Implement function schema parsing",
        title_zh="实现 function schema 解析",
        difficulty=2,
        round_name="agent",
        estimated_minutes=20,
        company_tags=("openai", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Parse a JSON schema-like function definition.", "Extract required fields.", "Return a normalized schema summary."),
        requirements_zh=("解析 JSON schema 风格的函数定义。", "抽取 required 字段。", "返回标准化后的 schema 摘要。"),
        followups_en=("How strict should validation be?", "How do nested objects change the parser?", "Which fields are required for tool execution?"),
        followups_zh=("校验应该多严格？", "如果有嵌套对象，解析器要怎么改？", "哪些字段对 tool execution 是必须的？"),
        starter={"kind": "function", "imports": (), "signature": "def parse_function_schema(schema):"},
        test_kind="function_schema_parsing",
    ),
    ProblemSpec(
        pid="P043",
        order=43,
        track="06_agents_tool_use",
        slug="agent_planner",
        title_en="Implement an agent planner",
        title_zh="实现 agent planner",
        difficulty=3,
        round_name="agent",
        estimated_minutes=25,
        company_tags=("anthropic", "bytedance"),
        high_frequency=False,
        requirements_en=("Break a task into ordered steps.", "Use available tool names as hints.", "Return a step list."),
        requirements_zh=("把任务拆成有序步骤。", "利用 available tools 作为提示。", "返回步骤列表。"),
        followups_en=("When should a planner skip tool use entirely?", "How do you prevent over-planning?", "Should the plan be editable after observations?"),
        followups_zh=("什么时候 planner 应直接跳过工具？", "如何避免过度规划？", "观察到新信息后计划是否应可修改？"),
        starter={"kind": "function", "imports": (), "signature": "def plan_steps(task: str, available_tools):"},
        test_kind="agent_planner",
    ),
    ProblemSpec(
        pid="P044",
        order=44,
        track="06_agents_tool_use",
        slug="memory_system",
        title_en="Implement a short-term memory system",
        title_zh="实现 memory system",
        difficulty=2,
        round_name="agent",
        estimated_minutes=20,
        company_tags=("domestic_bigtech",),
        high_frequency=False,
        requirements_en=("Store recent messages or observations.", "Support append and recent retrieval.", "Limit the memory window size."),
        requirements_zh=("保存最近消息或 observation。", "支持 append 和 recent retrieval。", "限制短期记忆窗口大小。"),
        followups_en=("Should memory keep tool outputs verbatim?", "How do you summarize when the window overflows?", "How is short-term memory different from retrieval memory?"),
        followups_zh=("tool 输出要不要原样保留？", "窗口溢出时如何做摘要？", "短期记忆和检索记忆有什么区别？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "ShortTermMemory",
            "methods": (
                ("__init__", "self, max_items: int = 5"),
                ("append", "self, item"),
                ("recent", "self"),
            ),
        },
        test_kind="memory_system",
    ),
    ProblemSpec(
        pid="P045",
        order=45,
        track="06_agents_tool_use",
        slug="react_agent_loop",
        title_en="Implement a ReAct agent loop",
        title_zh="实现 ReAct agent loop",
        difficulty=4,
        round_name="agent",
        estimated_minutes=40,
        company_tags=("openai", "anthropic"),
        high_frequency=False,
        requirements_en=("Iterate through Thought / Action / Observation.", "Call tools when the model requests one.", "Stop at a final answer."),
        requirements_zh=("实现 Thought / Action / Observation 循环。", "模型请求工具时执行 tool。", "在最终答案处停止。"),
        followups_en=("How do you cap the number of ReAct steps?", "What traces should be logged?", "How do you recover from malformed tool calls?"),
        followups_zh=("如何限制 ReAct 最大步数？", "应记录哪些 trace？", "工具调用格式错误时如何恢复？"),
        starter={"kind": "function", "imports": (), "signature": "def react_loop(question: str, llm, tools, max_steps: int = 5):"},
        test_kind="react_agent_loop",
    ),
    ProblemSpec(
        pid="P046",
        order=46,
        track="07_ai_systems_engineering",
        slug="model_parallel_scheduler",
        title_en="Implement model-parallel inference scheduling",
        title_zh="实现模型并行推理调度",
        difficulty=4,
        round_name="systems",
        estimated_minutes=40,
        company_tags=("openai", "anthropic"),
        high_frequency=False,
        requirements_en=("Assign requests to model shards.", "Track which shard is responsible for each stage.", "Return an executable stage plan."),
        requirements_zh=("把请求分配到模型分片。", "跟踪每个阶段负责的 shard。", "返回可执行的 stage plan。"),
        followups_en=("How do you minimize cross-device hops?", "What changes for pipeline vs tensor parallelism?", "How would you queue overloaded shards?"),
        followups_zh=("如何减少跨设备 hops？", "pipeline parallel 和 tensor parallel 有什么区别？", "分片过载时如何排队？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "ModelParallelScheduler",
            "methods": (
                ("__init__", "self, shard_ids"),
                ("schedule", "self, request_id: str, stages"),
            ),
        },
        test_kind="model_parallel_scheduler",
    ),
    ProblemSpec(
        pid="P047",
        order=47,
        track="07_ai_systems_engineering",
        slug="token_rate_limiter",
        title_en="Implement a token rate limiter",
        title_zh="实现 token rate limiter",
        difficulty=2,
        round_name="systems",
        estimated_minutes=20,
        company_tags=("bytedance", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Limit throughput by tokens per second.", "Track allowance over time.", "Expose an allow check."),
        requirements_zh=("按 tokens/sec 限流。", "跟踪随时间恢复的额度。", "提供 allow 检查。"),
        followups_en=("Why rate limit tokens instead of requests?", "Should prompt and output tokens share one budget?", "How do you handle bursts?"),
        followups_zh=("为什么按 token 而不是请求数限流？", "prompt 和 output token 是否共用预算？", "如何支持突发流量？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "TokenRateLimiter",
            "methods": (
                ("__init__", "self, tokens_per_second: float, burst: float | None = None"),
                ("allow", "self, token_count: int, now: float"),
            ),
        },
        test_kind="token_rate_limiter",
    ),
    ProblemSpec(
        pid="P048",
        order=48,
        track="07_ai_systems_engineering",
        slug="llm_cache_system",
        title_en="Implement an LLM cache system",
        title_zh="实现 LLM 缓存系统",
        difficulty=2,
        round_name="systems",
        estimated_minutes=20,
        company_tags=("openai", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Use a prompt hash as the key.", "Support get and put.", "Expose cache hit semantics clearly."),
        requirements_zh=("以 prompt hash 作为 key。", "支持 get 和 put。", "清晰表达 cache hit 语义。"),
        followups_en=("What should the cache key include besides the prompt?", "How do you handle model/version invalidation?", "Would you cache streamed responses?"),
        followups_zh=("除了 prompt，cache key 还应包含什么？", "模型版本变化后如何失效？", "streaming response 要不要缓存？"),
        starter={
            "kind": "class",
            "imports": ("import hashlib",),
            "class_name": "PromptCache",
            "methods": (
                ("__init__", "self"),
                ("make_key", "self, prompt: str"),
                ("get", "self, prompt: str"),
                ("put", "self, prompt: str, value"),
            ),
        },
        test_kind="llm_cache_system",
    ),
    ProblemSpec(
        pid="P049",
        order=49,
        track="07_ai_systems_engineering",
        slug="embedding_cache",
        title_en="Implement an embedding cache",
        title_zh="实现 embedding 缓存",
        difficulty=2,
        round_name="systems",
        estimated_minutes=20,
        company_tags=("anthropic", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Avoid recomputing identical embeddings.", "Support a user-provided embedder function.", "Cache by input text."),
        requirements_zh=("避免重复计算 embedding。", "支持注入 embedder 函数。", "按输入文本做缓存。"),
        followups_en=("How is this different from prompt caching?", "How would you bound cache size?", "When do you evict entries?"),
        followups_zh=("它和 prompt cache 的区别是什么？", "如何控制缓存大小？", "何时驱逐条目？"),
        starter={
            "kind": "class",
            "imports": (),
            "class_name": "EmbeddingCache",
            "methods": (
                ("__init__", "self, embedder"),
                ("get_embedding", "self, text: str"),
            ),
        },
        test_kind="embedding_cache",
    ),
    ProblemSpec(
        pid="P050",
        order=50,
        track="07_ai_systems_engineering",
        slug="llm_service_api",
        title_en="Implement a simple LLM service API",
        title_zh="实现简单 LLM 服务 API",
        difficulty=3,
        round_name="systems",
        estimated_minutes=30,
        company_tags=("openai", "domestic_bigtech"),
        high_frequency=False,
        requirements_en=("Expose POST /generate.", "Accept a prompt.", "Return generated text in JSON."),
        requirements_zh=("提供 POST /generate。", "接收 prompt。", "以 JSON 返回 generated text。"),
        followups_en=("What generation config should the endpoint expose?", "How do you stream tokens later without breaking the contract?", "Where should auth and rate limiting sit?"),
        followups_zh=("接口应暴露哪些生成参数？", "后续如何在不破坏协议的前提下支持流式返回？", "鉴权和限流应放在哪层？"),
        starter={"kind": "api_factory", "imports": ("from fastapi import FastAPI", "from pydantic import BaseModel"), "signature": "def create_app(generator):"},
        test_kind="llm_service_api",
    ),
]


MOCK_PACKS = [
    MockPackSpec(
        pid="M001",
        slug="core_model_round",
        title_en="Core Model Round",
        title_zh="模型基础轮",
        duration_minutes=75,
        problems=("P001", "P002", "P008", "P010"),
        goals_en=("Finish one core implementation cleanly.", "Explain normalization, masking, and shape reasoning.", "Leave 10 minutes for debrief."),
        goals_zh=("完整做出一题核心实现。", "能清楚解释归一化、mask 和 shape 推导。", "最后预留 10 分钟做复盘。"),
    ),
    MockPackSpec(
        pid="M002",
        slug="generation_serving_round",
        title_en="Generation and Serving Round",
        title_zh="推理优化轮",
        duration_minutes=75,
        problems=("P016", "P018", "P019", "P021"),
        goals_en=("Compare decoding strategies.", "Explain cache and latency tradeoffs.", "Show one deterministic test per function."),
        goals_zh=("比较不同 decoding 策略。", "能解释 cache 和延迟 tradeoff。", "每题至少给出一个可验证测试。"),
    ),
    MockPackSpec(
        pid="M003",
        slug="training_finetuning_round",
        title_en="Training and Finetuning Round",
        title_zh="训练微调轮",
        duration_minutes=70,
        problems=("P026", "P027", "P029", "P033"),
        goals_en=("Explain loss shapes and optimizer flow.", "Reason about PEFT tradeoffs.", "Show stable training loop habits."),
        goals_zh=("说清 loss shape 和 optimizer 流程。", "能讲清 PEFT 的 tradeoff。", "体现稳定训练循环习惯。"),
    ),
    MockPackSpec(
        pid="M004",
        slug="rag_agent_round",
        title_en="RAG and Agent Round",
        title_zh="RAG / Agent 轮",
        duration_minutes=90,
        problems=("P036", "P037", "P041", "P045"),
        goals_en=("Explain component boundaries clearly.", "Handle empty retrieval or malformed tool calls.", "Discuss observability and failure modes."),
        goals_zh=("清楚表达组件边界。", "能处理空召回或错误 tool call。", "会讨论可观测性和失败模式。"),
    ),
    MockPackSpec(
        pid="M005",
        slug="systems_engineering_round",
        title_en="AI Systems Engineering Round",
        title_zh="AI 系统工程轮",
        duration_minutes=80,
        problems=("P046", "P047", "P048", "P050"),
        goals_en=("Explain throughput, latency, and correctness tradeoffs.", "Design stable interfaces.", "Discuss rollout and safeguards."),
        goals_zh=("能解释吞吐、延迟和正确性 tradeoff。", "接口设计稳定。", "能讨论上线和保护机制。"),
    ),
]


def project_title(locale: str) -> str:
    if locale == "zh-cn":
        return "大模型面试复习仓库"
    return "LLM Interview Prep Repo"


def render_root_readme(locale: str) -> str:
    if locale == "zh-cn":
        return dedent(
            """\
            # 大模型面试复习仓库

            这是中文练习分支，按真实 AI / LLM 岗位面试轮次组织大模型手撕题、知识卡片、公共测试和模拟面试。

            ## 训练路径

            1. 先读 `docs/how_to_practice.md`
            2. 按 `docs/roadmaps/zero_to_one_8week.md` 或 `docs/roadmaps/top10_7day_sprint.md` 开练
            3. 每个题目按 `problem.md -> starter.py -> test_public.py -> followups.md` 的顺序刷

            ## 仓库结构

            - `docs/`: 训练说明、路线图、高频题单、轮次说明
            - `src/llm_prep/`: 共享接口与类型定义
            - `tracks/`: 7 个核心主题 + 5 套 Mock Interviews

            ## 分支说明

            - `main`: 技术中枢分支，只放共享代码骨架和同步规则
            - `codex/en-us`: 英文练习分支
            - `codex/zh-cn-solutions`: 中文答案分支
            - `codex/en-us-solutions`: 英文答案分支
            """
        )

    return dedent(
        """\
        # LLM Interview Prep Repo

        This is the English practice branch. The repository is organized around real AI / LLM interview rounds, with coding drills, flashcards, public tests, and timed mock packs.

        ## Practice Flow

        1. Start with `docs/how_to_practice.md`
        2. Pick either `docs/roadmaps/zero_to_one_8week.md` or `docs/roadmaps/top10_7day_sprint.md`
        3. Work each problem in the order `problem.md -> starter.py -> test_public.py -> followups.md`

        ## Repository Layout

        - `docs/`: study guides, roadmaps, top-frequency lists, and interview round mappings
        - `src/llm_prep/`: shared interfaces and types
        - `tracks/`: 7 core topic tracks plus 5 mock interview packs

        ## Branches

        - `main`: technical hub branch for shared code and sync rules
        - `codex/zh-cn`: Chinese practice branch
        - `codex/zh-cn-solutions`: Chinese solutions branch
        - `codex/en-us-solutions`: English solutions branch
        """
    )


def render_how_to_practice(locale: str) -> str:
    if locale == "zh-cn":
        return dedent(
            """\
            # 如何刷这个仓库

            ## 推荐节奏

            1. 先按 track 读 `concepts/overview.md`
            2. 再做对应题目的 `starter.py`
            3. 写完后运行该题的 `test_public.py`
            4. 最后看 `followups.md`，补上面试追问

            ## 一次刷题的最小闭环

            - 先自己写 20-40 分钟
            - 卡住时只允许回看知识卡片，不直接看答案
            - 用 public tests 检查边界
            - 最后用 3 分钟口头复述：题意、接口、复杂度、边界条件

            ## 分支使用方式

            - 题面练习请在 `codex/zh-cn` 或 `codex/en-us`
            - 参考答案只在 solutions 分支查看
            - `main` 只作为代码和测试同步基线
            """
        )

    return dedent(
        """\
        # How to Practice This Repo

        ## Recommended Loop

        1. Read the track-level `concepts/overview.md`
        2. Implement the prompt in `starter.py`
        3. Run the local `test_public.py`
        4. Finish with `followups.md` to rehearse the interview discussion

        ## Minimum Practice Cycle

        - Spend 20-40 minutes coding before you look anything up
        - If you are stuck, review the flashcards before looking at solutions
        - Use the public tests to check edge cases
        - End every session with a 3-minute verbal summary: goal, API, complexity, edge cases

        ## Branch Usage

        - Practice on `codex/zh-cn` or `codex/en-us`
        - Check solutions only on the paired solutions branch
        - Treat `main` as the shared sync baseline for code and tests
        """
    )


def render_interview_rounds(locale: str) -> str:
    if locale == "zh-cn":
        return dedent(
            """\
            # 面试轮次映射

            | 轮次 | 对应 Track | 重点 |
            | --- | --- | --- |
            | 模型基础轮 | `01_transformer_fundamentals`, `02_tokenization` | attention、RoPE、LayerNorm、Tokenizer |
            | 推理 / Serving 轮 | `03_inference_optimization`, `07_ai_systems_engineering` | sampling、KV cache、batching、API、缓存 |
            | 训练轮 | `04_training_finetuning` | loss、LoRA、训练循环、梯度策略 |
            | RAG / 检索轮 | `05_rag_systems` | chunking、召回、排序、pipeline |
            | Agent / Tool Use 轮 | `06_agents_tool_use` | planner、schema、memory、ReAct |
            | 综合模拟 | `90_mock_interviews` | 限时混合题 + 复盘 |
            """
        )

    return dedent(
        """\
        # Interview Round Mapping

        | Round | Tracks | Focus |
        | --- | --- | --- |
        | Core model round | `01_transformer_fundamentals`, `02_tokenization` | attention, RoPE, LayerNorm, tokenizers |
        | Generation / serving round | `03_inference_optimization`, `07_ai_systems_engineering` | sampling, KV cache, batching, APIs, caches |
        | Training round | `04_training_finetuning` | losses, LoRA, training loops, gradient strategies |
        | Retrieval round | `05_rag_systems` | chunking, retrieval, ranking, pipelines |
        | Agent / tool-use round | `06_agents_tool_use` | planner, schema parsing, memory, ReAct |
        | Mixed mock round | `90_mock_interviews` | timed blended practice and debrief |
        """
    )


def render_top10(locale: str) -> str:
    rows = [
        ("P001", "Scaled dot-product attention", "scaled dot-product attention"),
        ("P002", "Multi-head attention", "multi-head attention"),
        ("P010", "Mini Transformer language model", "mini Transformer language model"),
        ("P011", "BPE tokenizer", "BPE tokenizer"),
        ("P016", "KV cache", "KV cache"),
        ("P018", "Top-k sampling", "top-k sampling"),
        ("P019", "Top-p sampling", "top-p sampling"),
        ("P021", "Beam search", "beam search"),
        ("P029", "LoRA forward", "LoRA forward"),
        ("P036", "Cosine similarity search", "cosine similarity search"),
        ("P037", "Simple RAG pipeline", "简单 RAG pipeline"),
    ]
    if locale == "zh-cn":
        lines = ["# 高频 Top10（含一个 bonus）", "", "这组题覆盖最常见手撕能力。建议先刷完再扩展到完整 50 题。", ""]
        for pid, _, zh_title in rows:
            lines.append(f"- `{pid}`: {zh_title}")
        return "\n".join(lines) + "\n"
    lines = ["# High-Frequency Top 10 (plus one bonus)", "", "These are the fastest route to the most common hand-written interview patterns.", ""]
    for pid, en_title, _ in rows:
        lines.append(f"- `{pid}`: {en_title}")
    return "\n".join(lines) + "\n"


def render_zero_to_one(locale: str) -> str:
    if locale == "zh-cn":
        return dedent(
            """\
            # 8 周从 0 到 1 路线图

            1. 第 1 周：P001-P005，补 attention / FFN / block 基础
            2. 第 2 周：P006-P010，串起 RoPE、decoder、mini LM
            3. 第 3 周：P011-P015，系统过一遍 tokenizer
            4. 第 4 周：P016-P021，集中攻克采样与搜索
            5. 第 5 周：P022-P025，补齐 serving 工程题
            6. 第 6 周：P026-P033，训练 / 微调一口气打通
            7. 第 7 周：P034-P045，完整过 RAG + Agent
            8. 第 8 周：P046-P050 + M001-M005，刷系统工程与 Mock Interviews
            """
        )

    return dedent(
        """\
        # 8-Week Zero-to-One Roadmap

        1. Week 1: P001-P005 for attention, FFN, and block fundamentals
        2. Week 2: P006-P010 to connect RoPE, decoder layers, and a mini LM
        3. Week 3: P011-P015 for tokenizer fundamentals
        4. Week 4: P016-P021 for sampling and search
        5. Week 5: P022-P025 for serving-focused implementation drills
        6. Week 6: P026-P033 for training and finetuning
        7. Week 7: P034-P045 for RAG and agent loops
        8. Week 8: P046-P050 plus M001-M005 for systems and mock rounds
        """
    )


def render_top10_sprint(locale: str) -> str:
    if locale == "zh-cn":
        return dedent(
            """\
            # 7 天高频冲刺

            1. Day 1: P001 + P002
            2. Day 2: P010
            3. Day 3: P011
            4. Day 4: P016 + P018 + P019
            5. Day 5: P021
            6. Day 6: P029
            7. Day 7: P036 + P037 + 一套 M004 复盘
            """
        )

    return dedent(
        """\
        # 7-Day High-Frequency Sprint

        1. Day 1: P001 + P002
        2. Day 2: P010
        3. Day 3: P011
        4. Day 4: P016 + P018 + P019
        5. Day 5: P021
        6. Day 6: P029
        7. Day 7: P036 + P037 + one M004 debrief
        """
    )


def track_title(locale: str, track: TrackSpec) -> str:
    return track.title_zh if locale == "zh-cn" else track.title_en


def problem_title(locale: str, problem: ProblemSpec) -> str:
    return problem.title_zh if locale == "zh-cn" else problem.title_en


def render_track_readme(locale: str, track_key: str) -> str:
    track = TRACKS[track_key]
    title = track_title(locale, track)
    desc = track.desc_zh if locale == "zh-cn" else track.desc_en
    related = [p for p in PROBLEMS if p.track == track_key]
    lines = [f"# {title}", "", desc, "", "## Problems", ""]
    for problem in related:
        lines.append(f"- `{problem.pid}` {problem_title(locale, problem)}")
    if track_key == "90_mock_interviews":
        lines.extend(["", "## Mock Packs", ""])
        for pack in MOCK_PACKS:
            pack_title = pack.title_zh if locale == "zh-cn" else pack.title_en
            lines.append(f"- `{pack.pid}` {pack_title}")
    return "\n".join(lines) + "\n"


def render_concepts(locale: str, track_key: str) -> str:
    cards = CONCEPT_CARDS[track_key]["zh" if locale == "zh-cn" else "en"]
    title = "知识卡片" if locale == "zh-cn" else "Flashcards"
    lines = [f"# {title}", ""]
    for question, answer in cards:
        lines.append(f"## {question}")
        lines.append("")
        lines.append(answer)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_problem_md(locale: str, problem: ProblemSpec) -> str:
    title = problem_title(locale, problem)
    requirements = problem.requirements_zh if locale == "zh-cn" else problem.requirements_en
    lines = [f"# {problem.pid} {title}", ""]
    if locale == "zh-cn":
        lines.extend(
            [
                "## 题目目标",
                "",
                "围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。",
                "",
                "## 要求",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Goal",
                "",
                "Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.",
                "",
                "## Requirements",
                "",
            ]
        )
    for item in requirements:
        lines.append(f"- {item}")
    if locale == "zh-cn":
        lines.extend(
            [
                "",
                "## 练习建议",
                "",
                "- 先只把接口和 shape 跑通",
                "- 再补边界条件和异常输入",
                "- 最后口头说明时间复杂度与工程 tradeoff",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Practice Hint",
                "",
                "- First make the interface and shapes correct",
                "- Then add edge-case handling",
                "- Finish by explaining complexity and engineering tradeoffs out loud",
            ]
        )
    return "\n".join(lines) + "\n"


def render_followups(locale: str, problem: ProblemSpec) -> str:
    followups = problem.followups_zh if locale == "zh-cn" else problem.followups_en
    title = "面试追问" if locale == "zh-cn" else "Interview Follow-ups"
    lines = [f"# {title}", ""]
    for item in followups:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_meta(locale: str, problem: ProblemSpec) -> str:
    display_title = problem_title(locale, problem)
    lines = [
        f"id: {problem.pid}",
        f"title: {problem.slug}",
        f"display_title: \"{display_title}\"",
        f"track: {problem.track}",
        f"difficulty: {problem.difficulty}",
        f"round: {problem.round_name}",
        "company_tags:",
    ]
    for tag in problem.company_tags:
        lines.append(f"  - {tag}")
    lines.extend(
        [
            f"estimated_minutes: {problem.estimated_minutes}",
            f"high_frequency: {'true' if problem.high_frequency else 'false'}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_function_stub(spec: ProblemSpec) -> str:
    imports = "\n".join(spec.starter.get("imports", ()))
    prefix = f"{imports}\n\n" if imports else ""
    signature = spec.starter["signature"]
    return (
        f"{prefix}{signature}\n"
        f"    \"\"\"Starter stub for {spec.pid}.\"\"\"\n"
        f"    raise NotImplementedError(\"Implement {spec.pid} in this starter.\")\n"
    )


def render_nn_module_stub(spec: ProblemSpec) -> str:
    imports = "\n".join(spec.starter.get("imports", ()))
    class_name = spec.starter["class_name"]
    init_signature = spec.starter["init_signature"]
    forward_signature = spec.starter["forward_signature"]
    return dedent(
        f"""\
        {imports}


        class {class_name}(nn.Module):
            def __init__({init_signature}):
                super().__init__()
                raise NotImplementedError("Implement {spec.pid} in this starter.")

            def forward({forward_signature}):
                raise NotImplementedError("Implement {spec.pid} in this starter.")
        """
    )


def render_class_stub(spec: ProblemSpec) -> str:
    imports = "\n".join(spec.starter.get("imports", ()))
    class_name = spec.starter["class_name"]
    methods: tuple[tuple[str, str], ...] = spec.starter["methods"]  # type: ignore[assignment]
    lines = []
    if imports:
        lines.extend([imports, "", ""])
    lines.append(f"class {class_name}:")
    if not methods:
        lines.append("    pass")
    for method_name, signature in methods:
        lines.append(f"    def {method_name}({signature}):")
        if method_name == "__init__":
            lines.append("        raise NotImplementedError(\"Initialize this class in the starter.\")")
        else:
            lines.append(f"        raise NotImplementedError(\"Implement {spec.pid}.{method_name}().\")")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_api_factory_stub(spec: ProblemSpec) -> str:
    imports = "\n".join(spec.starter.get("imports", ()))
    signature = spec.starter["signature"]
    return dedent(
        f"""\
        {imports}


        class GenerateRequest(BaseModel):
            prompt: str


        {signature}
            raise NotImplementedError("Implement {spec.pid} in this starter.")
        """
    )


def render_starter(spec: ProblemSpec) -> str:
    kind = spec.starter["kind"]
    if kind == "function":
        return render_function_stub(spec)
    if kind == "nn_module":
        return render_nn_module_stub(spec)
    if kind == "class":
        return render_class_stub(spec)
    if kind == "api_factory":
        return render_api_factory_stub(spec)
    raise ValueError(f"Unsupported starter kind: {kind}")


def base_marks(spec: ProblemSpec) -> str:
    track_marker = {
        "01_transformer_fundamentals": "transformer",
        "02_tokenization": "tokenization",
        "03_inference_optimization": "inference",
        "04_training_finetuning": "training",
        "05_rag_systems": "rag",
        "06_agents_tool_use": "agent",
        "07_ai_systems_engineering": "systems",
    }[spec.track]
    lines = [
        "@pytest.mark.practice",
        f"@pytest.mark.{track_marker}",
        f"@pytest.mark.difficulty_{spec.difficulty}",
    ]
    if spec.high_frequency:
        lines.append("@pytest.mark.top10")
    prefix = " " * 12
    return "\n".join(f"{prefix}{line}" for line in lines)


def render_torch_shape_test(spec: ProblemSpec, class_name: str, ctor: str, call: str, shape: str) -> str:
    return dedent(
        f"""\
        from pathlib import Path

        import pytest

        torch = pytest.importorskip("torch")

        from llm_prep.utils.testing import load_local_starter, skip_not_implemented

        starter = load_local_starter(Path(__file__).with_name("starter.py"))


        {base_marks(spec)}
        def test_{spec.slug}_shape() -> None:
            try:
                module = starter.{class_name}({ctor})
                output = module({call})
            except Exception as exc:  # pragma: no cover - practice starter path
                skip_not_implemented(exc)
            assert tuple(output.shape) == {shape}
        """
    )


def render_function_test(spec: ProblemSpec) -> str:
    k = spec.test_kind
    if k == "scaled_dot_product_attention":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_scaled_dot_product_attention_supports_mask() -> None:
                q = torch.tensor([[[1.0, 0.0]]])
                k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
                v = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
                mask = torch.tensor([[0.0, float("-inf")]])
                try:
                    output = starter.scaled_dot_product_attention(q, k, v, mask=mask)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert tuple(output.shape) == (1, 1, 2)
                assert torch.allclose(output[0, 0], torch.tensor([2.0, 0.0]), atol=1e-4)
            """
        )
    if k == "multi_head_attention":
        return render_torch_shape_test(spec, "MultiHeadAttention", "hidden_size=8, num_heads=2", "torch.randn(2, 4, 8)", "(2, 4, 8)")
    if k == "self_attention_forward":
        return render_torch_shape_test(spec, "SelfAttention", "hidden_size=8", "torch.randn(2, 4, 8)", "(2, 4, 8)")
    if k == "transformer_block":
        return render_torch_shape_test(spec, "TransformerBlock", "hidden_size=8, num_heads=2, intermediate_size=16", "torch.randn(2, 4, 8)", "(2, 4, 8)")
    if k == "positionwise_feedforward":
        return render_torch_shape_test(spec, "PositionwiseFeedForward", "hidden_size=8, intermediate_size=16", "torch.randn(2, 4, 8)", "(2, 4, 8)")
    if k == "rope":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_apply_rope_preserves_shape() -> None:
                q = torch.randn(2, 3, 8)
                k = torch.randn(2, 3, 8)
                try:
                    q_out, k_out = starter.apply_rope(q, k)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert tuple(q_out.shape) == tuple(q.shape)
                assert tuple(k_out.shape) == tuple(k.shape)
            """
        )
    if k == "causal_mask":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_causal_mask_is_lower_triangular() -> None:
                try:
                    mask = starter.causal_mask(4)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert tuple(mask.shape) == (4, 4)
                assert torch.isfinite(torch.diag(mask)).all()
                assert mask[0, 3] < 0
            """
        )
    if k == "layer_norm":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_layer_norm_normalizes_last_dimension() -> None:
                x = torch.tensor([[1.0, 2.0, 3.0]])
                gamma = torch.ones(3)
                beta = torch.zeros(3)
                try:
                    out = starter.layer_norm(x, gamma, beta)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert torch.allclose(out.mean(dim=-1), torch.zeros(1), atol=1e-4)
            """
        )
    if k == "transformer_decoder_layer":
        return render_torch_shape_test(spec, "TransformerDecoderLayer", "hidden_size=8, num_heads=2, intermediate_size=16", "torch.randn(2, 4, 8)", "(2, 4, 8)")
    if k == "mini_transformer_lm":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_mini_transformer_lm_returns_vocab_logits() -> None:
                try:
                    model = starter.MiniTransformerLM(vocab_size=32, hidden_size=16, num_heads=4, intermediate_size=32)
                    logits = model(torch.randint(0, 32, (2, 5)))
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert tuple(logits.shape) == (2, 5, 32)
            """
        )
    if k == "bpe_tokenizer":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_bpe_tokenizer_round_trip_on_training_corpus() -> None:
                tokenizer = starter.BPETokenizer(vocab_size=20)
                try:
                    tokenizer.train(["low lower newest"])
                    tokens = tokenizer.encode("low")
                    text = tokenizer.decode(tokens)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert isinstance(tokens, list)
                assert isinstance(text, str)
            """
        )
    if k == "wordpiece_tokenizer":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_wordpiece_tokenizer_uses_greedy_matching() -> None:
                vocab = {{"[UNK]": 0, "play": 1, "##ing": 2}}
                tokenizer = starter.WordPieceTokenizer(vocab=vocab)
                try:
                    tokens = tokenizer.encode("playing")
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert isinstance(tokens, list)
            """
        )
    if k == "train_subword_vocab":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_train_subword_vocab_returns_dict() -> None:
                try:
                    vocab = starter.train_subword_vocab(["a aa aaa"], vocab_size=8)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert isinstance(vocab, dict)
                assert len(vocab) <= 8
            """
        )
    if k == "tokenize_detokenize":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_round_trip_tokenizer_is_reversible() -> None:
                tokenizer = starter.RoundTripTokenizer()
                try:
                    tokenizer.fit(["hello world"])
                    token_ids = tokenizer.encode("hello world")
                    text = tokenizer.decode(token_ids)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert text == "hello world"
            """
        )
    if k == "padding_and_attention_mask":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_pad_sequences_returns_padded_batch_and_mask() -> None:
                try:
                    padded, mask = starter.pad_sequences([[1, 2, 3], [4]], pad_token_id=0)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(padded) == 2
                assert len(mask[0]) == 3
                assert mask[1][-1] == 0
            """
        )
    if k == "kv_cache":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_kv_cache_appends_incrementally() -> None:
                cache = starter.KVCache()
                try:
                    cache.append([1], [10])
                    cache.append([2], [20])
                    keys, values = cache.get()
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(keys) == 2
                assert len(values) == 2
            """
        )
    if k == "greedy_decoding":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_greedy_decode_returns_argmax_index() -> None:
                try:
                    token_id = starter.greedy_decode([0.1, 0.9, 0.2])
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert token_id == 1
            """
        )
    if k == "top_k_sampling":
        return dedent(
            f"""\
            import random
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_top_k_sample_only_returns_from_top_k() -> None:
                random.seed(0)
                try:
                    token_id = starter.top_k_sample([0.1, 0.9, 0.8, 0.0], k=2)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert token_id in {{1, 2}}
            """
        )
    if k == "top_p_sampling":
        return dedent(
            f"""\
            import random
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_top_p_sample_returns_valid_index() -> None:
                random.seed(0)
                try:
                    token_id = starter.top_p_sample([3.0, 2.0, 0.1], p=0.8)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert token_id in {{0, 1, 2}}
            """
        )
    if k == "temperature_sampling":
        return dedent(
            f"""\
            import random
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_temperature_sample_returns_valid_index() -> None:
                random.seed(0)
                try:
                    token_id = starter.temperature_sample([1.0, 2.0, 3.0], temperature=0.7)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert token_id in {{0, 1, 2}}
            """
        )
    if k == "beam_search":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            def step_fn(tokens):
                last = tokens[-1]
                if last == 2:
                    return [(-0.1, 2)]
                return [(-0.1, last + 1), (-0.5, 2)]


            {base_marks(spec)}
            def test_beam_search_returns_sequence() -> None:
                try:
                    sequence = starter.beam_search(step_fn, start_tokens=[0], beam_size=2, max_new_tokens=3, eos_token_id=2)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert isinstance(sequence, list)
                assert sequence[0] == 0
            """
        )
    if k == "logits_processor":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_repetition_penalty_changes_seen_tokens() -> None:
                logits = [1.0, 2.0, 3.0]
                try:
                    adjusted = starter.apply_repetition_penalty(logits, [1], penalty=1.2)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(adjusted) == 3
                assert adjusted[1] != logits[1]
            """
        )
    if k == "dynamic_batching":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_dynamic_batcher_flushes_in_enqueue_order() -> None:
                batcher = starter.DynamicBatcher(max_batch_size=4)
                try:
                    batcher.enqueue({{"id": "a"}})
                    batcher.enqueue({{"id": "b"}})
                    batch = batcher.flush()
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert [item["id"] for item in batch] == ["a", "b"]
            """
        )
    if k == "prefix_caching":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_prefix_cache_returns_longest_match() -> None:
                cache = starter.PrefixCache()
                try:
                    cache.add_prefix((1, 2), "short")
                    cache.add_prefix((1, 2, 3), "long")
                    value = cache.lookup((1, 2, 3, 4))
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert value == "long"
            """
        )
    if k == "streaming_generation":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            def step_fn(tokens):
                return len(tokens) + 10


            {base_marks(spec)}
            def test_stream_generate_yields_tokens() -> None:
                try:
                    output = list(starter.stream_generate(step_fn, [1, 2], max_new_tokens=3))
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(output) == 3
            """
        )
    if k == "cross_entropy_loss":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_cross_entropy_loss_returns_scalar() -> None:
                logits = torch.tensor([[2.0, 1.0]])
                targets = torch.tensor([0])
                try:
                    loss = starter.cross_entropy_loss(logits, targets)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert loss.ndim == 0
            """
        )
    if k == "causal_lm_training_loss":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_causal_lm_loss_returns_scalar() -> None:
                logits = torch.randn(2, 4, 5)
                labels = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
                try:
                    loss = starter.causal_lm_loss(logits, labels)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert loss.ndim == 0
            """
        )
    if k == "gradient_clipping":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class ToyModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(2))


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_clip_gradients_returns_preclip_norm() -> None:
                model = ToyModel()
                model.weight.grad = torch.tensor([3.0, 4.0])
                try:
                    norm = starter.clip_gradients(model.parameters(), max_norm=1.0)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert norm >= 5.0
            """
        )
    if k == "lora_forward":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_lora_linear_returns_projected_shape() -> None:
                try:
                    layer = starter.LoRALinear(in_features=4, out_features=3, rank=2)
                    output = layer(torch.randn(2, 4))
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert tuple(output.shape) == (2, 3)
            """
        )
    if k == "lora_merge":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_merge_lora_weights_preserves_dense_shape() -> None:
                base_weight = torch.zeros(3, 4)
                lora_a = torch.ones(2, 4)
                lora_b = torch.ones(3, 2)
                try:
                    merged = starter.merge_lora_weights(base_weight, lora_a, lora_b, alpha=2.0)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert tuple(merged.shape) == (3, 4)
            """
        )
    if k == "parameter_freezing":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class ToyModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.encoder = torch.nn.Linear(2, 2)
                    self.head = torch.nn.Linear(2, 1)


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_freeze_parameters_keeps_matching_names_trainable() -> None:
                model = ToyModel()
                try:
                    trainable = starter.freeze_parameters(model, ["head"])
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert any("head" in name for name in trainable)
            """
        )
    if k == "sft_training_loop":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class ToyModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(4, 2)

                def forward(self, x):
                    return self.linear(x)


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_sft_train_step_returns_float_loss() -> None:
                model = ToyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                batch = {{"inputs": torch.randn(2, 4), "labels": torch.tensor([0, 1])}}
                try:
                    loss = starter.sft_train_step(model, batch, optimizer)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert isinstance(loss, float)
            """
        )
    if k == "gradient_accumulation":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            torch = pytest.importorskip("torch")

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class ToyModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(4, 2)

                def forward(self, x):
                    return self.linear(x)


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_gradient_accumulation_returns_step_count() -> None:
                model = ToyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                batches = [
                    {{"inputs": torch.randn(2, 4), "labels": torch.tensor([0, 1])}},
                    {{"inputs": torch.randn(2, 4), "labels": torch.tensor([1, 0])}},
                ]
                try:
                    step_count = starter.run_gradient_accumulation(model, batches, optimizer, accumulation_steps=2)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert step_count == 1
            """
        )
    if k == "document_chunking":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_chunk_document_respects_overlap() -> None:
                try:
                    chunks = starter.chunk_document("abcdefghij", chunk_size=4, overlap=1)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert chunks[0] == "abcd"
                assert chunks[1].startswith("d")
            """
        )
    if k == "embedding_vector_store":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            def embedder(text):
                return [float(len(text)), 1.0]


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_vector_store_adds_and_searches_documents() -> None:
                store = starter.InMemoryVectorStore(embedder=embedder)
                try:
                    store.add_document("doc-1", "short")
                    store.add_document("doc-2", "longer text")
                    results = store.search("short", top_k=1)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(results) == 1
            """
        )
    if k == "cosine_similarity_search":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_cosine_similarity_search_returns_best_id() -> None:
                try:
                    results = starter.cosine_similarity_search([1.0, 0.0], {{"a": [1.0, 0.0], "b": [0.0, 1.0]}}, top_k=1)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                top_id = results[0][0] if isinstance(results[0], tuple) else results[0]["doc_id"]
                assert top_id == "a"
            """
        )
    if k == "simple_rag_pipeline":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class DummyRetriever:
                def search(self, query, top_k=3):
                    return [{{"doc_id": "doc-1", "text": "retrieved context"}}]


            def dummy_llm(prompt):
                return f"answer::{{prompt}}"


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_simple_rag_pipeline_returns_final_answer() -> None:
                try:
                    answer = starter.run_simple_rag("what is rag", DummyRetriever(), dummy_llm, "Question: {{query}}\\nContext: {{context}}")
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert answer.startswith("answer::")
            """
        )
    if k == "mmr_retrieval":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_mmr_select_returns_requested_count() -> None:
                doc_embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
                try:
                    selected = starter.mmr_select([1.0, 0.0], doc_embeddings, lambda_mult=0.5, top_k=2)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(selected) == 2
            """
        )
    if k == "reranker":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            def scorer(query, doc):
                return len(set(query.split()) & set(doc.split()))


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_rerank_documents_sorts_descending() -> None:
                docs = ["llm systems", "systems", "tokens"]
                try:
                    ordered = starter.rerank_documents("llm systems", docs, scorer)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert ordered[0] == "llm systems"
            """
        )
    if k == "multi_query_retrieval":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            def query_generator(query):
                return [query, query + " rewrite"]


            class DummyRetriever:
                def search(self, query, top_k=3):
                    suffix = "1" if "rewrite" not in query else "2"
                    return [{{"doc_id": f"doc-{{suffix}}", "text": query}}]


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_multi_query_retrieve_deduplicates_results() -> None:
                try:
                    results = starter.multi_query_retrieve("query", query_generator, DummyRetriever(), top_k=3)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(results) >= 1
            """
        )
    if k == "tool_calling_agent":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented
            from llm_prep.types import ToolCall


            class EchoTool:
                name = "echo"
                description = "Echoes the provided text."

                def call(self, arguments):
                    return arguments["text"]


            class DummyLLM:
                def __init__(self):
                    self.calls = 0

                def generate(self, messages, tools=None):
                    self.calls += 1
                    if self.calls == 1:
                        return ToolCall(name="echo", arguments={{"text": "tool result"}})
                    return "final answer"


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_tool_calling_agent_runs_tool_then_returns_answer() -> None:
                agent = starter.ToolCallingAgent(llm=DummyLLM(), tools=[EchoTool()])
                try:
                    answer = agent.run("say hi")
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert answer == "final answer"
            """
        )
    if k == "function_schema_parsing":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_parse_function_schema_extracts_required_fields() -> None:
                schema = {{
                    "name": "search",
                    "parameters": {{
                        "type": "object",
                        "properties": {{"query": {{"type": "string"}}}},
                        "required": ["query"],
                    }},
                }}
                try:
                    parsed = starter.parse_function_schema(schema)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert parsed["name"] == "search"
                assert "query" in parsed["required"]
            """
        )
    if k == "agent_planner":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_plan_steps_returns_ordered_list() -> None:
                try:
                    steps = starter.plan_steps("answer a support ticket", ["retrieve_docs", "draft_reply"])
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert isinstance(steps, list)
                assert len(steps) >= 1
            """
        )
    if k == "memory_system":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_short_term_memory_keeps_recent_window() -> None:
                memory = starter.ShortTermMemory(max_items=2)
                try:
                    memory.append("a")
                    memory.append("b")
                    memory.append("c")
                    recent = memory.recent()
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert recent == ["b", "c"]
            """
        )
    if k == "react_agent_loop":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class SearchTool:
                name = "search"

                def call(self, arguments):
                    return f"obs::{{arguments['query']}}"


            class DummyLLM:
                def __init__(self):
                    self.calls = 0

                def __call__(self, prompt):
                    self.calls += 1
                    if self.calls == 1:
                        return {{"action": "search", "arguments": {{"query": "llm"}}}}
                    return {{"final_answer": "done"}}


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_react_loop_returns_final_answer() -> None:
                try:
                    answer = starter.react_loop("what is llm", DummyLLM(), [SearchTool()], max_steps=3)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert answer == "done"
            """
        )
    if k == "model_parallel_scheduler":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_model_parallel_scheduler_returns_stage_plan() -> None:
                scheduler = starter.ModelParallelScheduler(shard_ids=["gpu0", "gpu1"])
                try:
                    plan = scheduler.schedule("req-1", ["embed", "decode"])
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert len(plan) == 2
            """
        )
    if k == "token_rate_limiter":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_token_rate_limiter_respects_budget() -> None:
                limiter = starter.TokenRateLimiter(tokens_per_second=10.0, burst=10.0)
                try:
                    first = limiter.allow(6, now=0.0)
                    second = limiter.allow(6, now=0.0)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert first is True
                assert second is False
            """
        )
    if k == "llm_cache_system":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_prompt_cache_hits_by_prompt() -> None:
                cache = starter.PromptCache()
                try:
                    cache.put("hello", {{"text": "world"}})
                    value = cache.get("hello")
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert value == {{"text": "world"}}
            """
        )
    if k == "embedding_cache":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            from llm_prep.utils.testing import load_local_starter, skip_not_implemented


            class CountingEmbedder:
                def __init__(self):
                    self.calls = 0

                def __call__(self, text):
                    self.calls += 1
                    return [float(len(text))]


            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            {base_marks(spec)}
            def test_embedding_cache_avoids_duplicate_work() -> None:
                embedder = CountingEmbedder()
                cache = starter.EmbeddingCache(embedder)
                try:
                    first = cache.get_embedding("hello")
                    second = cache.get_embedding("hello")
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                assert first == second
                assert embedder.calls == 1
            """
        )
    if k == "llm_service_api":
        return dedent(
            f"""\
            from pathlib import Path

            import pytest

            fastapi = pytest.importorskip("fastapi")
            _ = fastapi

            from fastapi.testclient import TestClient
            from llm_prep.utils.testing import load_local_starter, skip_not_implemented

            starter = load_local_starter(Path(__file__).with_name("starter.py"))


            def generator(prompt):
                return f"generated::{{prompt}}"


            {base_marks(spec)}
            def test_create_app_exposes_generate_endpoint() -> None:
                try:
                    app = starter.create_app(generator)
                except Exception as exc:  # pragma: no cover
                    skip_not_implemented(exc)
                client = TestClient(app)
                response = client.post("/generate", json={{"prompt": "hello"}})
                assert response.status_code == 200
                assert response.json()["generated_text"] == "generated::hello"
            """
        )
    raise ValueError(f"Unsupported test kind: {k}")


def render_mock_problem(locale: str, pack: MockPackSpec) -> str:
    title = pack.title_zh if locale == "zh-cn" else pack.title_en
    goals = pack.goals_zh if locale == "zh-cn" else pack.goals_en
    intro = "在真实限时下完成下列题目，并按评分标准复盘。" if locale == "zh-cn" else "Solve the following pack under a realistic time box, then debrief with the rubric."
    lines = [f"# {pack.pid} {title}", "", intro, "", "## Pack", ""]
    for pid in pack.problems:
        lines.append(f"- `{pid}`")
    lines.extend(["", "## Goals", ""])
    for goal in goals:
        lines.append(f"- {goal}")
    return "\n".join(lines) + "\n"


def render_mock_followups(locale: str, pack: MockPackSpec) -> str:
    if locale == "zh-cn":
        return dedent(
            f"""\
            # 评分与复盘

            - 总时长：{pack.duration_minutes} 分钟
            - 编码正确性：40%
            - 表达与推导：30%
            - 测试与边界：20%
            - 工程判断：10%
            """
        )
    return dedent(
        f"""\
        # Scoring and Debrief

        - Total time: {pack.duration_minutes} minutes
        - Coding correctness: 40%
        - Communication and reasoning: 30%
        - Testing and edge cases: 20%
        - Engineering judgment: 10%
        """
    )


def render_mock_starter(pack: MockPackSpec) -> str:
    return dedent(
        f"""\
        MOCK_PACK = {{
            "id": "{pack.pid}",
            "duration_minutes": {pack.duration_minutes},
            "problems": {list(pack.problems)},
        }}


        def get_mock_pack():
            return MOCK_PACK
        """
    )


def render_mock_test(pack: MockPackSpec) -> str:
    return dedent(
        f"""\
        from pathlib import Path

        import pytest

        from llm_prep.utils.testing import load_local_starter

        starter = load_local_starter(Path(__file__).with_name("starter.py"))


        @pytest.mark.smoke
        @pytest.mark.mock
        def test_{pack.slug}_pack_shape() -> None:
            pack = starter.get_mock_pack()
            assert pack["id"] == "{pack.pid}"
            assert len(pack["problems"]) == {len(pack.problems)}
        """
    )


def render_mock_meta(locale: str, pack: MockPackSpec) -> str:
    title = pack.title_zh if locale == "zh-cn" else pack.title_en
    return dedent(
        f"""\
        id: {pack.pid}
        title: {pack.slug}
        display_title: "{title}"
        track: 90_mock_interviews
        difficulty: 4
        round: mock
        company_tags:
          - mixed_loop
        estimated_minutes: {pack.duration_minutes}
        high_frequency: false
        """
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_test_source(content: str) -> str:
    lines = content.splitlines()
    adjusted = [line.lstrip() if line.lstrip().startswith("@pytest.mark") else line for line in lines]
    return "\n".join(adjusted).lstrip() + "\n"


def normalize_starter_source(content: str) -> str:
    lines = content.splitlines()
    adjusted = [line[8:] if line.startswith(" " * 8) else line for line in lines]
    return "\n".join(adjusted).lstrip() + "\n"


def generate_docs(locale: str) -> None:
    if locale in {"zh-cn", "en-us"}:
        write(ROOT / "README.md", render_root_readme(locale))
    write(ROOT / "docs/how_to_practice.md", render_how_to_practice(locale))
    write(ROOT / "docs/interview_rounds.md", render_interview_rounds(locale))
    write(ROOT / "docs/high_frequency_top10.md", render_top10(locale))
    write(ROOT / "docs/roadmaps/zero_to_one_8week.md", render_zero_to_one(locale))
    write(ROOT / "docs/roadmaps/top10_7day_sprint.md", render_top10_sprint(locale))


def generate_tracks(locale: str) -> None:
    for track_key in REGULAR_TRACK_KEYS:
        track_dir = ROOT / "tracks" / track_key
        write(track_dir / "README.md", render_track_readme(locale, track_key))
        write(track_dir / "concepts/overview.md", render_concepts(locale, track_key))
        for problem in [p for p in PROBLEMS if p.track == track_key]:
            problem_dir = track_dir / "problems" / f"{problem.pid}_{problem.slug}"
            write(problem_dir / "problem.md", render_problem_md(locale, problem))
            write(problem_dir / "followups.md", render_followups(locale, problem))
            starter_source = render_starter(problem)
            if problem.starter["kind"] in {"nn_module", "api_factory"}:
                starter_source = normalize_starter_source(starter_source)
            write(problem_dir / "starter.py", starter_source)
            write(problem_dir / "test_public.py", normalize_test_source(render_function_test(problem)))
            write(problem_dir / "meta.yaml", render_meta(locale, problem))


def generate_mocks(locale: str) -> None:
    mock_dir = ROOT / "tracks/90_mock_interviews"
    write(mock_dir / "README.md", render_track_readme(locale, "90_mock_interviews"))
    write(mock_dir / "concepts/overview.md", render_concepts(locale, "90_mock_interviews"))
    for pack in MOCK_PACKS:
        pack_dir = mock_dir / "problems" / f"{pack.pid}_{pack.slug}"
        write(pack_dir / "problem.md", render_mock_problem(locale, pack))
        write(pack_dir / "followups.md", render_mock_followups(locale, pack))
        write(pack_dir / "starter.py", render_mock_starter(pack))
        write(pack_dir / "test_public.py", normalize_test_source(render_mock_test(pack)))
        write(pack_dir / "meta.yaml", render_mock_meta(locale, pack))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold the bilingual LLM interview prep repository.")
    parser.add_argument("--locale", choices=["main", "zh-cn", "en-us"], required=True)
    args = parser.parse_args()

    locale = "en-us" if args.locale == "main" else args.locale
    generate_docs(locale)
    generate_tracks(locale)
    generate_mocks(locale)


if __name__ == "__main__":
    main()
