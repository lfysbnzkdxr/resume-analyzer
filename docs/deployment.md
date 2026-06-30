# 部署指南

## 环境变量

所有配置通过环境变量覆盖，支持 `.env` 文件（开发环境）或系统环境变量（生产环境）。

| 变量 | 默认值 | 必填 | 说明 |
|------|--------|------|------|
| `DEEPSEEK_API_KEY` | — | 是 | DeepSeek API 密钥 |
| `RA_BASE_URL` | `https://api.deepseek.com` | 否 | API 基础地址 |
| `RA_MODEL` | `deepseek-chat` | 否 | 模型名称 |
| `RA_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 否 | 嵌入模型 |
| `RA_CHUNK_SIZE` | `500` | 否 | 文本分块大小 |
| `RA_CHUNK_OVERLAP` | `50` | 否 | 分块重叠大小 |
| `RA_TOP_K` | `5` | 否 | 检索返回数量 |
| `RA_TIMEOUT` | `60` | 否 | API 超时秒数 |
| `RA_HF_ENDPOINT` | `https://hf-mirror.com` | 否 | HuggingFace 镜像端点（中国大陆用） |

## Docker 部署

### 构建镜像

```bash
docker build -t resume-analyzer .
```

中国大陆用户使用镜像加速：

```bash
docker build -t resume-analyzer --build-arg HF_ENDPOINT=https://hf-mirror.com .
```

### 运行容器

```bash
docker run -d \
  --name resume-analyzer \
  -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -e DEEPSEEK_API_KEY=your_key_here \
  resume-analyzer
```

### docker-compose

```bash
# 创建 .env 文件并填入 DEEPSEEK_API_KEY
echo "DEEPSEEK_API_KEY=your_key_here" > .env

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

浏览器打开 `http://localhost:8501`。

### 健康检查

容器内置 HEALTHCHECK，每 30 秒检查一次：

- **ChromaDB** — 验证向量数据库可读写
- **Embedding 模型** — 验证 ONNX 模型已加载（首次启动可能因模型下载超时，第二次检查应通过）
- **DeepSeek API** — 验证 API 密钥有效且服务可达（无密钥时跳过）

手动检查：

```bash
docker exec resume-analyzer python -m src.core.healthcheck
```

输出示例（正常）：

```json
{
  "status": "ok",
  "duration_ms": 423,
  "checks": {
    "chromadb": { "passed": true, "message": "ChromaDB heartbeat OK" },
    "embeddings": { "passed": true, "message": "Embedding OK (dim=512)" },
    "deepseek_api": { "passed": true, "message": "DeepSeek API OK (model=deepseek-chat)" }
  }
}
```

### 持久化数据

`data/` 目录需要挂载为卷或绑定挂载，包含：

- `data/chroma_db/` — 向量数据库（简历库持久化）
- `data/logs/` — 应用日志（自动轮转，5MB×3）
- `data/uploads/` — 临时上传文件（自动清理）

## Streamlit Community Cloud 部署

1. 将代码推送到 GitHub 仓库
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 "New app"，选择仓库、分支和入口文件 `src/main.py`
4. 在 Settings → Secrets 中设置：

```toml
DEEPSEEK_API_KEY = "your_key_here"
```

### 注意事项

- **文件系统是临时的**：Streamlit Cloud 每次部署会重置文件系统，ChromaDB 数据不会持久化
- **嵌入模型下载**：首次启动需要下载 ONNX 模型（~33MB），可能增加启动时间
- **内存限制**：免费版 1GB RAM，足以运行 fastembed + ChromaDB
- **会话超时**：一段时间无活动后应用会休眠

## 手动部署

### 前置条件

- Python 3.9+
- DeepSeek API Key

### 步骤

```bash
# 1. 克隆
git clone <repo-url>
cd resume-analyzer

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装
pip install -e ".[dev]"

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 5. 启动
streamlit run src/main.py
```

### Nginx 反向代理示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

## 运行测试

```bash
# 单元测试（无外部依赖）
pytest -v

# 包含覆盖率
pytest -v --cov=src

# Lint 检查
ruff check src/ tests/

# 类型检查
mypy src/
```

## 常见问题

### Embedding 模型下载失败

**症状**：首次启动慢，日志显示 ONNX 模型下载超时。

**解决**：设置中国大陆镜像 `HF_ENDPOINT=https://hf-mirror.com`。模型缓存后不会重复下载。

### ChromaDB 权限错误

**症状**：容器启动后 healthcheck 显示 ChromaDB 失败。

**解决**：确保 `data/` 目录已挂载，且容器用户 `appuser` 有写入权限。挂载前：`mkdir -p data && chmod 777 data`。

### DeepSeek API 连接失败

**症状**：分析页面报错 "API 调用失败"。

**解决**：检查 API Key 是否有效、账户余额是否充足、`RA_BASE_URL` 是否正确（中国大陆用户可能需要切换代理或使用 API 中转）。
