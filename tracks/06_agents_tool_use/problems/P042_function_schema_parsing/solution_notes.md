# P042 Function Schema Parsing 题解

## 核心思路

1. 解析 OpenAI 风格的 function/tool schema JSON，提取函数名、描述、参数定义
2. Schema 结构：`{"name": str, "description": str, "parameters": {"type": "object", "properties": {...}, "required": [...]}}`
3. 遍历 properties，解析每个参数的 name、type、description、enum、default 等字段
4. 根据 required 列表标记必填参数
5. 可生成函数签名字符串或校验给定参数是否符合 schema

## 面试口述模板

> "这道题的核心是解析 JSON Schema 格式的函数定义。我先提取顶层的 name 和 description，然后进入 parameters.properties 遍历每个参数。对每个参数读取 type、description，检查是否在 required 列表中。难点在于处理嵌套类型——比如 array 类型要看 items 字段，object 类型需要递归解析。我还会处理 enum 约束和 default 值。最终可以用解析结果做参数校验：检查必填参数是否齐全、类型是否匹配。"

## 常见坑

- **嵌套 object**：properties 中的参数自身可能是 object 类型，需要递归解析
- **array 的 items**：array 类型必须检查 items 字段确定元素类型
- **可选参数默认值**：schema 中可能没有 default 字段，不要假设所有可选参数都有默认值
- **anyOf / oneOf**：OpenAI schema 支持联合类型，解析时需特殊处理

## 复杂度

- 时间：O(P)，P 为 schema 中参数总数（含嵌套），每个参数常数时间解析
- 空间：O(P) 存储解析后的参数结构
