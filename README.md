# Agent demo — document intelligence with deterministic tools

一个能跑的最小演示，用来给海外客户看「AI Agent 不是聊天机器人」。
零 API Key 也能完整跑通，跑出来的每个数字都能人工手算复核。

---

## 它演示什么

| 卖点 | 在代码里的位置 | 客户能看懂的点 |
|---|---|---|
| 混合检索（向量 + BM25，RRF 融合） | `store.py` | 精确词（发票号）和语义查询一起命中 |
| 数字类任务不用 LLM | `tools.py` | 每个金额来自正则 + 算术，不是模型猜的 |
| 权限过滤在查询层 | `store.py` 的 `PG_QUERY` | 不是把结果后过滤，不会先泄漏进上下文 |
| 写操作必须人工确认 | `graph.py` 的 `approve` 节点 | 图真的挂起，不点同意就什么都不写 |
| 数据不一致时拒绝回答 | `tools.py` 的混合币种分支 | 宁可报错也不给一个错数字 |

---

## 怎么跑

```
cd E:\海外接单\agent-demo
pip install -r requirements.txt
python demo.py
```

`demo.py` 是给客户看的脚本，会依次演示：
1. 比较两张发票，算出差额和百分比
2. 提出一个「标记发票」的写操作，图挂起等待人工
3. 人工同意，图恢复执行
4. 人工拒绝，什么都不写

单条提问用 CLI：

```
python cli.py ingest
python cli.py ask "Compare INV-2026-0413 and INV-2026-0418"
python cli.py ask "Flag INV-2026-0418 for finance review"
python cli.py ask "Flag INV-2026-0418 for finance review" --approve
```

---

## 没有 API Key 也能跑的原因

两处默认走离线实现，这是刻意的，不是偷懒：

- `EMBED_BACKEND=hashing`：用 hashing trick 做词级向量。它是词面匹配，不是语义模型。
  README 里明说这一点，别在客户面前把它吹成 embedding 模型。
  换真模型：`EMBED_BACKEND=openai`，或把 `embed.py` 里的函数换成 bge-m3。
- `LLM_BACKEND=mock`：模板生成。真实模式下 LLM 只负责写解释，数字仍然来自工具。

接真模型：

```
$env:LLM_BACKEND="openai"
$env:EMBED_BACKEND="openai"
$env:OPENAI_API_KEY="sk-..."
```

---

## 接 pgvector（生产路径）

`store.py` 里已经写好 `PgVectorStore`，同一个接口，换后端只改环境变量：

```
$env:DATABASE_URL="postgresql://user:pass@host:5432/db"
```

表结构在 `PG_DDL` 里，包含 `tenant_id` / `visibility` 两个权限列，以及 HNSW 索引。
关键在 `PG_QUERY` 的 WHERE 子句：权限过滤发生在数据库查询里，不是拿到结果之后再删。
后者的漏洞是敏感片段已经进了模型上下文。

注意：`compare_invoices` 里比较的是「金额最高的那个数」，它把 unit price 和 subtotal
排除掉了。这是个启发式规则，真实项目里应该改成先识别 `Total due` 这类字段标签再取值。
演示里没做，因为真实的发票格式千奇百怪，这一段必须在客户数据上做，不能提前假装通用。

---

## 目录

```
agent-demo/
  config.py      配置与环境变量
  embed.py       向量化（hashing 离线 / openai）
  store.py       混合检索 + SQLite / pgvector 双后端
  ingest.py      文档入库
  tools.py       确定性工具：字段抽取、发票比较、标记动作
  llm.py         LLM 只写解释，不产数字
  graph.py       LangGraph：retrieve → route → tool → approve → answer
  cli.py         命令行
  demo.py        给客户看的脚本
  data/          4 份样例文档
```

---

## 演示时说的话（英文，可直接念）

> "The model in this demo writes zero numbers. Every figure you see was extracted by a
> parser and compared with subtraction — 19,120 minus 18,450 is 670, and the model
> cannot change that. The model only writes the sentence around it."

> "Watch what happens when it wants to change something. The graph suspends. It cannot
> write until a human approves. That's the difference between an agent and a chatbot."

> "And when the documents can't be compared honestly — mixed currencies, missing totals —
> it refuses and tells you why, instead of producing a confident wrong number."

---

## 已知边界（别在客户面前夸大成通用产品）

- 字段抽取是正则，只认 `INV-2026-0413` 这种格式和标准金额写法。真实票据要换 LLM 抽取 + 规则校验双跑。
- hashing embedding 不理解同义词，「higher」「expensive」这类词靠 BM25 命中。
- `flag_for_review` 是模拟动作，没有真的写数据库，接真实系统时在 `approve` 之后接事务。
- 没有做 PDF 解析、OCR、多语言。这些是下一层的事，不是这一层的卖点。
