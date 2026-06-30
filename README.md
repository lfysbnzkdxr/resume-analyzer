# AI 简历智能分析师 & 职位匹配引擎

一个展示 **RAG + Function Calling + 多 Agent 协作 + Eval 评估** 的 AI 工程化项目。

## 技术亮点

| 能力 | 具体实现 |
|------|----------|
| **RAG 检索增强生成** | ChromaDB + fastembed（bge-small-zh-v1.5）本地语义检索 |
| **Function Calling** | DeepSeek API 工具调用 + LangChain Tool Calling Agent |
| **多 Agent 协作** | 4 个专用 Agent（提取∥分析→匹配→库管理）并行编排 |
| **中文优化 RAG** | PyMuPDF 解析 + 中文断句切片（支持 。和句子边界） + **混合检索** |
| **语义+关键词混合检索** | ChromaDB 语义搜索 + BM25 关键词匹配 + RRF 融合排序 |
| **Eval 评估体系** | 多维指标（skill_recall, RMSE, pass_rate）量化分析质量 |

## 技术栈

| 层 | 技术 |
|---|------|
| **LLM** | DeepSeek API（OpenAI 兼容接口，Function Calling） |
| **Embedding** | BAAI/bge-small-zh-v1.5（512维，ONNX 纯 CPU 推理） |
| **向量数据库** | ChromaDB（PersistentClient，cosine 相似度） |
| **Agent 框架** | LangChain Tool Calling Agent |
| **前端** | Streamlit |
| **PDF 解析** | PyMuPDF |
| **数据模型** | Pydantic v2 |

## 架构

```
用户 (Browser)
   │
   ▼ Streamlit UI
   ├── 单份分析 ──┐
   ├── 简历库管理 ─┤
   └── 库匹配检索 ─┘
         │
         ▼ Orchestrator（流程编排）
   ┌──────┬──────┐           Resume ⟂ JD（并行执行）
   │      │                  ThreadPoolExecutor
   ▼      ▼
Resume  JD
Agent  Agent —— Matching Agent
   │      │      │
   ▼      ▼      ▼
   工具函数层（PDF解析 / 技能匹配 / 检索）
         │
         ▼ RAG Pipeline
   ┌──────┼──────┐
   ▼      ▼      ▼
 PyMuPDF 文本   ChromaDB
 解析器  切片器  +Embedding
```

## 功能

- **📄 单份分析** — 上传简历 PDF + 粘贴职位描述，AI 自动提取结构信息并给出多维匹配评分和改进建议
- **📚 简历库管理** — 批量上传简历，自动切片向量化入库，支持增删管理
- **🔍 库匹配检索** — 粘贴职位描述，从简历库中检索最匹配的候选人，按简历聚合去重（块级别→简历级别），展示语义相似度百分比
- **📊 多维评分** — 技能、经验、教育、项目四个维度加权评分，可视化展示
- **💡 改进建议** — 基于差距分析的高/中/低优先级改进建议
- **✅ Eval 评估** — 量化评估分析质量的测试体系（skill_recall, RMSE, pass_rate）

## 快速开始

### 前置条件

- Python 3.9+
- DeepSeek API Key（[deepseek.com](https://platform.deepseek.com) 注册获取）

### 安装

```bash
# 1. 克隆
git clone <repo-url>
cd resume-analyzer

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. 安装依赖（含 dev 可选依赖）
pip install -e ".[dev]"
# 或使用 Makefile:
# make install

# 4. 配置 API Key
# 方式 A：复制 .env 文件
cp .env.example .env
# 然后编辑 .env 填入你的 Key

# 方式 B：启动后在 Streamlit 侧边栏输入
```

### 运行

```bash
streamlit run src/main.py
# 或: make run
```

浏览器打开 `http://localhost:8501`，在侧边栏输入 DeepSeek API Key 即可使用。

### 运行测试

```bash
pytest -v
# 或: make test
```

### 运行 Eval 评估

```bash
python eval/run_eval.py
```

## 项目结构

```
resume-analyzer/
├── src/
│   ├── main.py                     # 入口（支持 --version / --healthcheck CLI）
│   ├── core/
│   │   ├── config.py               # 配置（RA_* 环境变量覆盖 / 路径 / 参数）
│   │   ├── models.py               # Pydantic 数据模型
│   │   ├── orchestrator.py         # 多 Agent 流程编排
│   │   ├── logging_config.py       # 日志配置（RotatingFileHandler）
│   │   └── healthcheck.py          # 健康检查（ChromaDB/Embedding/API）
│   ├── agents/
│   │   ├── base.py                 # LLM 初始化（DeepSeek，api_key 显式传入）
│   │   ├── resume_agent.py         # 简历信息提取 Agent
│   │   ├── jd_agent.py             # 职位描述分析 Agent
│   │   ├── matching_agent.py       # 匹配评估 Agent
│   │   └── utils.py                # JSON 提取工具 + 重试逻辑
│   ├── rag/
│   │   ├── loader.py               # PDF 解析（PyMuPDF）
│   │   ├── splitter.py             # 中文断句切片
│   │   ├── embeddings.py           # Embedding 封装（fastembed）
│   │   ├── vector_store.py         # ChromaDB 向量存储 + BM25 混合检索
│   │   ├── retriever.py            # 检索结果格式化
│   │   └── library.py              # 简历库管理（add/list/delete/stats）
│   ├── tools/
│   │   ├── pdf_parser.py           # PDF 解析工具（@tool）
│   │   └── skill_matcher.py        # 技能匹配工具（@tool）
│   └── ui/
│       ├── app.py                  # Streamlit 主应用（含 error_boundary）
│       ├── theme.py                # 主题常量（颜色/阈值）
│       ├── pages/
│       │   ├── single_analysis.py  # 单份分析页面
│       │   ├── library_manage.py   # 简历库管理页面
│       │   └── library_match.py    # 库匹配检索页面
│       └── components/
│           └── score_chart.py      # 评分图表组件
├── tests/
│   ├── conftest.py                 # pytest 配置
│   ├── test_skill_matcher.py       # 技能匹配单元测试
│   ├── test_models.py              # Pydantic 模型验证
│   ├── test_extract_json.py        # JSON 提取边界测试
│   ├── test_splitter.py            # 中文分块测试
│   └── test_rag_pipeline.py        # RAG 端到端测试
├── eval/
│   ├── test_cases.py               # 5 个评估测试用例
│   ├── metrics.py                  # 评估指标计算
│   └── run_eval.py                 # 评估运行器
├── data/
│   ├── chroma_db/                  # 向量数据库持久化（gitignored）
│   └── uploads/                    # 临时上传文件（gitignored）
├── docs/
│   ├── deployment.md               # 部署指南（Docker/Cloud/手动）
│   └── superpowers/specs/
│       └── 2026-06-16-ai-resume-analyzer-design.md  # 设计文档
├── .github/workflows/ci.yml        # GitHub Actions CI（3.9/3.11/3.12 矩阵）
├── .pre-commit-config.yaml         # pre-commit hooks（ruff + 安全检查）
├── Dockerfile                      # 多阶段 Docker 构建
├── docker-compose.yml              # Docker Compose（ChromaDB 持久化）
├── .dockerignore
├── Makefile                        # 快捷命令（install/run/test/lint/format）
├── .env.example                    # API Key 配置模板
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Eval 评估体系

量化分析质量的测试框架，覆盖 5 个场景：

| 测试用例 | 场景 | 指标 |
|----------|------|------|
| perfect_match_01 | 完美匹配（AI 算法 vs AI 算法） | skill_recall ≥ 0.8 |
| partial_match_01 | 部分匹配（Java 后端 vs AI 算法） | 评分 20-50 |
| no_match_01 | 完全不匹配（会计 vs AI） | 评分 ≤ 20 |
| boundary_empty_01 | 边界情况（空简历） | 预期报错 |
| english_mixed_01 | 中英混合 | 正常分析 |

运行：
```bash
python eval/run_eval.py
```

输出 JSON 报告，包含通过率、技能召回率均值、评分 RMSE 等指标。

## 关键技术决策

### 为什么选 bge-small-zh-v1.5？
- 33MB 模型体积，纯 CPU 毫秒级推理，77 语言覆盖（含中文）
- 512 维向量，检索精度与 768 维接近但存储减半
- ONNX 运行时（fastembed）无需 PyTorch，避免 Windows 进程冲突

### 为什么选 fastembed 而非 sentence-transformers？
- fastembed 用 ONNX 推理，无 PyTorch 依赖
- 避免 PyTorch 与 ChromaDB 的 onnxruntime 在同一进程冲突导致的段错误
- 安装体积小，首次下载模型后完全离线

### 中文分块策略
- 主分隔符优先级：段落 > 行 > 句号 > 英文句点 > 空格
- 500 字符块大小 + 50 字符重叠，平衡上下文完整性和检索精度
- rfind 反向查找分割点，避免在词中间截断

### 混合检索策略
- 语义搜索使用 bge-small-zh-v1.5 向量化余弦相似度
- 关键词搜索使用 jieba 分词 + BM25Okapi，精确匹配技能名称
- Reciprocal Rank Fusion 融合两个结果集，`rank_constant=60`（用于排序）
- MMR（λ=0.7）多样化排序，避免同一份简历的切片垄断 Top-N
- **简历级聚合**：库匹配结果按文件名聚合去重，取每份简历的最佳匹配块作为代表，显示真实余弦相似度而非 RRF 排名值
- **BM25 脏标记延迟重建**：单文件上传立即重建索引，批量上传逐文件跳过，结束后统一重建一次，避免冗余全量扫描

### 为什么 4 个 Agent 而非 1 个？
- 单一 Agent 做所有事：prompt 过长超出上下文、工具数量过多导致选择不准
- 分离 Agent：每个 prompt 控制在一页内，工具 1-2 个，Function Calling 精度更高
- 编排器只负责传参和组装结果，Agent 间完全解耦
