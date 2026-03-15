# 知识卡片

## 为什么主流 tokenizer 都是 subword？

它在词表大小、未登录词鲁棒性和压缩率之间取得了平衡。

## BPE 和 WordPiece 的实用差异是什么？

BPE 更偏频次合并，WordPiece 更偏基于似然提升选择合并。

## 为什么练习仓库里 encode/decode 必须可逆？

否则空格、特殊符号和边界处理的 bug 很难暴露。

## padding 后为什么还需要 attention mask？

让模型在注意力和 loss 计算时忽略补齐位置。
