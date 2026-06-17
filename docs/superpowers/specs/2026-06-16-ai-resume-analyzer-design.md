# AI 简历智能分析师 & 职位匹配引擎

> 一个展示 RAG + Function Calling + 多 Agent 协作 + Eval 完整能力的 AI 工程作品

## 1. 项目概述

### 1.1 目标

构建一个纯 Python 的简历智能分析系统，展示以下 AI 工程能力：

- **RAG Pipeline 完整实现**：文档解析 → 切片 → Embedding → 向量检索 → 重排序
- **Function Calling / 工具调用**：DeepSeek 函数调用提取结构化信息
- **多 Agent 协作**：分工明确的 Agent 团队完成复杂分析任务
- **Prompt Engineering**：Few-shot、Chain-of-Thought、结构化输出
- **Eval 方法论**：定量的测试用例评估
- **MCP 扩展性**：预留 MCP Server 接口作为进阶能力

### 1.2 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| **LLM** | DeepSeek API（deepseek-chat） | Function Calling + 推理 |
| **Embedding** | Ollama + `bge-m3` 或 `nomic-embed-text` | 本地 CPU 运行，零费用 |
| **向量数据库** | ChromaDB（PersistentClient） | 纯本地持久化 |
| **Agent 框架** | LangChain（LangGraph 可选） | Agent 编排 + Tool Calling |
| **PDF 解析** | PyMuPDF / pdfplumber | 提取简历文本 |
| **UI** | Streamlit | Python 原生可视化 |
| **评估** | 自定义 Eval 脚本 | 20+ 测试用例 + 指标计算 |

### 1.3 功能概览

```
📂 简历管理模式（RAG 核心）
   上传多份简历 → 解析切片 → Embedding → 存入 ChromaDB
   → 输入 JD → 语义检索最佳匹配 → 多 Agent 联合评估

📄 单份分析模式（Agent 核心）
   上传 PDF + 粘贴 JD → 双 Agent 分别提取结构化信息
   → 匹配评估 → 输出评分和改进建议
```

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                          │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │ 单份分析页面   │  │ 简历库管理页面 │  │ JD 匹配检索页面    │   │
│  │ Upload + JD   │  │ Batch Upload  │  │ Input JD → Search │   │
│  └───────┬───────┘  └───────┬───────┘  └─────────┬─────────┘   │
└──────────┼──────────────────┼────────────────────┼──────────────┘
           │                  │                    │
┌──────────▼──────────────────▼────────────────────▼──────────────┐
│                    Application Layer                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AnalysisOrchestrator                         │   │
│  │  (流程编排：决定单份/批量模式，协调各 Agent)              │   │
│  └────┬──────────┬──────────────┬────────────────┬──────────┘   │
│       │          │              │                │              │
│  ┌────▼───┐ ┌───▼────┐  ┌──────▼──────┐  ┌─────▼──────┐       │
│  │ Resume │ │ JD     │  │  Matching   │  │ Library    │       │
│  │ Agent  │ │ Agent  │  │  Agent      │  │ Agent      │       │
│  └────┬───┘ └───┬────┘  └──────┬──────┘  └─────┬──────┘       │
│       │         │              │                │              │
└───────┼─────────┼──────────────┼────────────────┼──────────────┘
        │         │              │                │
┌───────▼─────────▼──────────────▼────────────────▼──────────────┐
│                     AI Layer                                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DeepSeek API (Function Calling)             │   │
│  │   tool_choice="auto"  →  Agent 自主决定调哪些工具       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Ollama (Local)     │  ChromaDB (Persistent)            │   │
│  │  bge-m3 / nomic-emb │  collection: resume_chunks        │   │
│  │  CPU / 0 cost       │  本地文件持久化                    │   │
│  └─────────────────────┴───────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Agent 设计

### 3.1 Resume Extractor Agent（简历提取 Agent）

**职责**：从 PDF 简历中提取结构化信息

**工具**：
- `parse_resume_pdf(file_path) → str` — 解析 PDF 提取文本

**输出（JSON Schema）**：
```json
{
  "personal_info": { "name": "..., "email": "...", "phone": "..." },
  "summary": "...
  "skills": ["Python", "PyTorch", ...],
  "experience": [
    { "company": "...", "role": "...", "duration": "...", "highlights": [...] }
  ],
  "education": [{ "school": "...", "degree": "...", "major": "..." }],
  "projects": [{ "name": "...", "description": "...", "technologies": [...] }]
}
```

**Prompt 策略**：Few-shot + 结构化约束，System Prompt 明确要求 JSON 输出

### 3.2 JD Analyzer Agent（职位描述分析 Agent）

**职责**：从 JD 文本中提取关键要求

**工具**：（无，纯 LLM 分析）

**输出（JSON Schema）**：
```json
{
  "role_name": "AI Engineer",
  "required_skills": [{ "skill": "Python", "level": "proficient", "importance": "must" }],
  "preferred_skills": [{ "skill": "Kubernetes", "level": "familiar", "importance": "plus" }],
  "experience_required": { "years_min": 3, "description": "..." },
  "education_required": { "degree": "Bachelor", "major": "CS/AI related" },
  "responsibilities": ["..."]
}
```

### 3.3 Matching & Evaluation Agent（匹配评估 Agent）

**职责**：综合简历和 JD 信息，给出匹配度评分和改进建议

**工具**：
- `search_resume_library(jd_embedding, top_k=5) → list[dict]` — 从简历库语义检索（仅库模式）
- `calculate_skill_overlap(resume_skills, jd_skills) → dict` — 技能重合度计算

**输出（JSON Schema）**：
```json
{
  "overall_score": 78,
  "dimensions": [
    { "name": "技能匹配", "score": 85, "weight": 0.4, "details": "..." },
    { "name": "经验匹配", "score": 70, "weight": 0.3, "details": "..." },
    { "name": "教育背景", "score": 90, "weight": 0.1, "details": "..." },
    { "name": "项目匹配", "score": 75, "weight": 0.2, "details": "..." }
  ],
  "summary": "候选人整体匹配度良好，建议加强...",
  "suggestions": [
    { "category": "skill_gap", "priority": "high", "content": "缺乏 Kubernetes 经验，建议..." },
    { "category": "presentation", "priority": "medium", "content": "项目描述应突出量化结果..." }
  ],
  "matched_resumes": []  // 库模式返回其他匹配简历
}
```

### 3.4 Library Agent（简历库管理 Agent）

**职责**：管理多条简历的入库和检索

**关键流程**：
1. 接收 PDF → 切片（chunk_size=500, chunk_overlap=50）
2. 调用 Ollama Embedding → 生成向量
3. 存入 ChromaDB（metadata 包含文件名、切片序号）
4. 检索：JD 文本 → Embedding → ChromaDB similarity_search → Top-K

---

## 4. RAG Pipeline 设计

```
PDF 简历上传
    │
    ▼
文本提取 (PyMuPDF)
    │
    ▼
文本切片 (RecursiveCharacterTextSplitter)
    │  chunk_size=500, chunk_overlap=50
    │  separators=["\n\n", "\n", "。", ".", " ", ""]
    │
    ▼
Embedding (Ollama 本地)
    │  bge-m3 / nomic-embed-text
    │  CPU 运行，无 GPU 要求
    │
    ▼
向量存储 (ChromaDB PersistentClient)
    │  collection: "resume_chunks"
    │  metadata: {filename, chunk_index, page_num}
    │
    ▼
语义检索 ──→ Rerank (可选) ──→ Agent 读取 → 分析
    │
    ▼
JD Embedding 作为 query
Top-K (k=5~10)
```

### 切片策略

中文简历特点：短句多、专有名词多（技术栈、公司名）

| 参数 | 值 | 理由 |
|------|-----|------|
| chunk_size | 500 | 中文约 500 字/段，匹配简历段落粒度 |
| chunk_overlap | 50 | 保持上下文连贯 |
| separators | ["\n\n", "\n", "。", ".", " ", ""] | 中文优先按段落和句号切分 |

### Embedding 选型对比

| 模型 | 大小 | 维度 | 中文效果 | 速度 |
|------|------|------|---------|------|
| `bge-m3` (BAAI) | 2.2GB | 1024 | ⭐⭐⭐⭐⭐ 最佳 | ⭐⭐⭐ 中等 |
| `nomic-embed-text` | 274MB | 768 | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 最快 |
| `bge-small-zh-v1.5` | 33MB | 512 | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 最快 |

**推荐**：先用 `nomic-embed-text`（体积小、速度快），如果需要更高精度可换 `bge-m3`。

---

## 5. 数据模型

```python
# 简历文档
class ResumeDocument:
    id: str              # UUID
    filename: str        # 原文件名
    raw_text: str        # PDF 提取的全文
    chunks: list[str]    # 切片后的文本块
    created_at: datetime

# 职位描述
class JobDescription:
    id: str
    raw_text: str
    extracted: JDRequirements | None  # Agent 提取的结构化要求

# 分析结果
class AnalysisResult:
    id: str
    resume_id: str
    jd_id: str
    overall_score: float          # 0-100
    dimensions: list[DimensionScore]
    suggestions: list[Suggestion]
    summary: str
    created_at: datetime

# 维度评分
class DimensionScore:
    name: str                     # e.g. "技能匹配"
    score: float                  # 0-100
    weight: float                 # 权重，总和=1
    details: str                  # 详细分析

# 改进建议
class Suggestion:
    category: str                 # skill_gap / presentation / format / experience
    priority: str                 # high / medium / low
    content: str                  # 建议内容

# JD 要求（Agent 输出）
class JDRequirements:
    role_name: str
    required_skills: list[Skill]  # 硬性要求
    preferred_skills: list[Skill] # 加分项
    experience_required: str
    education_required: str
    responsibilities: list[str]

class Skill:
    name: str
    level: str                    # proficient / familiar / basic
    importance: str               # must / plus
```

---

## 6. 项目结构

```
resume-analyzer/
├── README.md
├── pyproject.toml               # 依赖管理
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # Streamlit 入口
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Agent 基类
│   │   ├── resume_agent.py      # 简历提取 Agent
│   │   ├── jd_agent.py          # JD 分析 Agent
│   │   ├── matching_agent.py    # 匹配评估 Agent
│   │   └── library_agent.py     # 简历库管理 Agent
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # 流程编排
│   │   ├── models.py            # 数据模型 (Pydantic)
│   │   └── config.py            # 配置管理 (API Key 等)
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py            # PDF 加载器
│   │   ├── splitter.py          # 文本切片
│   │   ├── embeddings.py        # Ollama Embedding 封装
│   │   ├── vector_store.py      # ChromaDB 封装
│   │   └── retriever.py         # 检索 + 重排序
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py        # PDF 解析工具
│   │   ├── skill_matcher.py     # 技能匹配计算工具
│   │   └── search.py            # 简历库检索工具
│   │
│   └── ui/
│       ├── __init__.py
│       ├── app.py               # Streamlit 主应用
│       ├── pages/
│       │   ├── single_analysis.py   # 单份分析页面
│       │   ├── library_manage.py    # 简历库管理页面
│       │   └── library_match.py     # 库匹配检索页面
│       └── components/
│           ├── score_chart.py       # 评分图表（雷达图/柱状图）
│           ├── suggestion_card.py   # 建议卡片展示
│           └── resume_preview.py    # 简历内容预览
│
├── data/
│   ├── chroma_db/              # ChromaDB 持久化存储（gitignore）
│   └── uploads/                # 上传的 PDF（gitignore）
│
├── eval/
│   ├── __init__.py
│   ├── test_cases.py           # 测试用例（简历 + JD + 预期结果）
│   ├── metrics.py              # 评估指标
│   └── run_eval.py             # 运行评估
│
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-16-ai-resume-analyzer-design.md
```

---

## 7. 用户界面设计

### 7.1 页面一：单份分析

```
┌───────────────────────────────────────────────────────┐
│  📄 单份分析    📚 简历库   🔍 库匹配      ⚙️ 设置   │  ← Sidebar
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │ 上传简历 PDF         │  │ 粘贴职位描述            │ │
│  │ [Drag & Drop Area]   │  │ [TextArea]             │ │
│  │                      │  │                        │ │
│  │ file.pdf ✓           │  │ AI Engineer @ ByteDance│ │
│  └─────────────────────┘  └─────────────────────────┘ │
│                                                       │
│  [▶ 开始分析]                                           │
│                                                       │
│  ──── 分析结果 ────                                    │
│                                                       │
│  综合匹配度: 85/100                                    │
│  ┌─────────────────────────────────────────────────┐  │
│  │             雷达图 (Radar Chart)                │  │
│  │  技能匹配 ████████████ 90  经验匹配 ████████ 75 │  │
│  │  教育背景 ██████████ 85   项目匹配 ████████ 80  │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  📋 改进建议                                          │
│  ├─ 🔴 [高优] 缺乏 Kubernetes 经验                  │
│  ├─ 🟡 [中优] 项目描述应突出量化结果                  │
│  └─ 🟢 [低优] 建议补充教育经历详细描述                │
│                                                       │
│  📝 详细分析                                          │
│  [Markdown 格式的长文本分析报告]                       │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 7.2 页面二：简历库管理

```
┌───────────────────────────────────────────────────────┐
│  📄 单份分析    📚 简历库   🔍 库匹配      ⚙️ 设置   │
├───────────────────────────────────────────────────────┤
│                                                       │
│  [📤 上传简历] 支持批量上传 PDF                        │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 当前库中有 12 份简历                              │  │
│  │                                                 │  │
│  │ ┌─────┬────────────┬──────────┬──────────┐     │  │
│  │ │ #   │ 文件名      │ 状态     │ 操作     │     │  │
│  │ ├─────┼────────────┼──────────┼──────────┤     │  │
│  │ │ 1   │ 张三_AI.pdf│ ✅ 已入库 │ [删除]   │     │  │
│  │ │ 2   │ 李四_ML.pdf│ ✅ 已入库 │ [删除]   │     │  │
│  │ │ 3   │ 王五.pdf   │ ⏳ 处理中 │          │     │  │
│  │ └─────┴────────────┴──────────┴──────────┘     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 7.3 页面三：库匹配检索

```
┌───────────────────────────────────────────────────────┐
│  📄 单份分析    📚 简历库   🔍 库匹配      ⚙️ 设置   │
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 粘贴职位描述                                     │  │
│  │ [TextArea]                                      │  │
│  │                                                 │  │
│  │ 我们正在寻找一位 AI 工程师...                    │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  [🔍 从库中检索最佳匹配]                               │
│                                                       │
│  ──── Top-5 匹配结果 ────                              │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ #1 张三_AI.pdf  ████████████ 87%  ██████████   │  │
│  │   匹配分析：技能覆盖全面，3 年经验与 JD 高度吻合  │  │
│  │   [查看详细分析]                                  │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ #2 李四_ML.pdf  ████████████ 72%  ████████     │  │
│  │   匹配分析：ML 经验丰富，但缺少工程化能力         │  │
│  │   [查看详细分析]                                  │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 8. Eval 评估体系

### 8.1 目标

通过定量评估证明系统的分析质量，这是简历上的亮点。

### 8.2 测试用例设计

20 个测试用例，覆盖：

| 分类 | 数量 | 场景 |
|------|:----:|------|
| **完美匹配** | 3 | 简历与 JD 高度吻合 |
| **部分匹配** | 5 | 技能覆盖 50-70%，存在差距 |
| **不匹配** | 3 | 方向完全不同 |
| **边界情况** | 4 | 简历极短/极长、PDF 扫描件、图片式 PDF |
| **中文实体** | 3 | 含非常见技术栈、非标准学位名称 |
| **英文/混排** | 2 | 英文简历 + 中文 JD 等 |

### 8.3 评估指标

| 指标 | 计算方式 | 目标 |
|------|---------|:----:|
| **技能召回率** | Agent 识别的技能数 / 真实技能总数 | >85% |
| **匹配精度** | 高优先级建议中正确的比例 | >80% |
| **评分误差** | 系统评分 vs 人工标注评分的 RMSE | <10 |
| **维度覆盖** | 是否覆盖所有预设评分维度 | =100% |
| **响应时间** | 单次分析耗时 | <30s |

### 8.4 评估运行

```bash
python eval/run_eval.py --test-cases eval/test_cases.py --output eval/report.json
```

输出 HTML 报告，包含：
- 每个测试用例的详细结果
- 汇总指标表
- 失败案例分析

---

## 9. 部署方案

| 组件 | 方案 |
|------|------|
| **Streamlit 应用** | 本地运行 `streamlit run` |
| **DeepSeek API** | 用户自备 API Key（环境变量） |
| **Ollama** | 用户自行安装，拉取 embedding 模型 |
| **ChromaDB** | 本地文件存储，随应用启动 |
| **部署到公网** | 可选：Streamlit Cloud / Railway / 自建服务器（需暴露 Ollama 端口） |

**启动方式**：
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Ollama（若未运行）
ollama pull nomic-embed-text
ollama serve

# 3. 配置 API Key
export DEEPSEEK_API_KEY="sk-xxx"
# 或在 Streamlit 界面中输入

# 4. 运行
streamlit run src/main.py
```

---

## 10. 简历技能映射

| 项目功能 | 对应简历技能点 | 面试话题 |
|---------|---------------|---------|
| PDF 解析 + 文本切片 | 文档处理 pipeline | "如何处理非结构化数据？" |
| Ollama 本地 Embedding | 向量化 + 模型选型 | "为什么选 bge-m3 而不是 OpenAI？" |
| ChromaDB 存储 + 检索 | 向量数据库设计 | "Top-K 参数怎么调？" |
| DeepSeek Function Calling | 工具调用设计 | "Tool Schema 怎么定义？" |
| 多 Agent 协作 | Agent 编排模式 | "Agent 之间怎么通信？" |
| Eval 测试用例 | 评估方法论 | "怎么衡量 LLM 输出质量？" |
| Streamlit UI | 工程交付能力 | "怎么让非技术人员使用你的模型？" |

---

## 11. 扩展方向（MVP 之后）

- **MCP Server 接入**：将分析功能暴露为 MCP Tool，供其他 AI 系统调用
- **多种 LLM 支持**：支持 OpenAI / Claude / 本地 Ollama 模型切换
- **批量分析**：一次 JD 匹配全库简历，输出排名报告
- **定时任务**：定期爬取职位，自动匹配库中简历
- **可视化仪表盘**：简历库技能分布统计、趋势分析
- **Rerank 优化**：接入 Cross-Encoder 提升检索精度

---

## 12. 非功能性需求

- **隐私**：所有数据本地存储，不上传第三方（除 DeepSeek API 调用外）
- **可配置**：API Key 通过环境变量或 UI 输入，不硬编码
- **错误处理**：PDF 解析失败、API 超时、Ollama 未启动等场景有友好提示
- **可扩展**：Agent 和工具注册式设计，方便新增 Agent/工具

---

## 13. 开发计划

### Phase 1：基础设施（~1 天）
- [ ] 项目初始化：pyproject.toml、requirements.txt
- [ ] ChromaDB + Ollama Embedding 集成
- [ ] PDF 解析 + 文本切片
- [ ] 验证 RAG 链路端到端

### Phase 2：Agent 系统（~1 天）
- [ ] Agent 基类 + 工具注册
- [ ] Resume Extractor Agent（Function Calling）
- [ ] JD Analyzer Agent
- [ ] Matching Agent

### Phase 3：Streamlit UI（~1 天）
- [ ] 单份分析页面
- [ ] 简历库管理页面
- [ ] 库匹配检索页面
- [ ] 结果可视化（雷达图）

### Phase 4：Eval 体系（~0.5 天）
- [ ] 20 个测试用例
- [ ] 评估脚本 + 指标计算
- [ ] 评估报告生成

### Phase 5：打磨与文档（~0.5 天）
- [ ] README 编写
- [ ] 错误处理完善
- [ ] 首次分析演示流程验证
