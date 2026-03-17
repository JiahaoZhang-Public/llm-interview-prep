# 面试追问

## 基础理解
- 为什么要成对旋转维度？不成对行不行？
- RoPE 如何体现**相对**位置信息？证明 Q_m · K_n 只依赖 (m-n)。
- RoPE vs 绝对位置编码 vs ALiBi：各自的优缺点？

## 长度外推
- 什么是 NTK-aware scaling？如何修改 base 做长度外推？
- YaRN 在 RoPE 上做了什么改进？
- 为什么 RoPE 在超出训练长度时效果会退化？

## 工程实现
- 如果是 Grouped-Query Attention，RoPE 应该怎么适配？
- cos/sin 表可以预计算吗？如何节省计算？
- 在 half precision (fp16) 下 RoPE 有数值问题吗？
