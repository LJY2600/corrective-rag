# Corrective RAG

> 本项目基于 [Shubham Saboo 的 Corrective RAG](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/corrective_rag) 改造，遵循 Apache-2.0 许可证。

基于 **LangGraph + 通义千问 + Chroma** 的纠正式检索增强生成（CRAG）应用。

## ✨ 功能

- 📄 支持 URL / 本地文件（PDF、TXT、MD）作为知识源
- 🔍 向量检索（Chroma）+ 相关性打分（LLM 逐条评分）
- ✏️ 查询重写（优化检索语义）
- 🌐 联网搜索兜底（Tavily）
- 🤖 基于通义千问的答案生成
- 📊 中间步骤可视化（Streamlit expander）

## 🛠 技术栈

| 组件 | 选型 |
|------|------|
| UI | Streamlit |
| 编排 | LangGraph |
| LLM | 阿里云通义千问（OpenAI 兼容接口）|
| Embedding | 阿里云 text-embedding-v3 |
| 向量库 | Chroma |
| 联网搜索 | Tavily（可选）|

## 🚀 快速开始

### 1. 安装依赖
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. 配置 API Key
启动后，在左侧边栏填入 **DashScope API Key**（必需，阿里云百炼）。

### 3. 启动
\`\`\`bash
streamlit run app.py
\`\`\`

## 📁 项目结构

\`\`\`
corrective-rag/
├── app.py            # Streamlit 入口
├── config.py         # 常量配置
├── prompts.py        # Prompt 模板
├── services.py       # LLM / Embedding / VectorStore 工厂
├── ingestion.py      # 文档加载 + 切分 + 入库
├── graph.py          # LangGraph 节点 + 图构建
└── requirements.txt
\`\`\`

## 🙏 Acknowledgements

本项目的核心架构基于以下开源项目改造：

- **Original Project**: [Corrective RAG (CRAG)](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/corrective_rag)
- **Author**: [Shubham Saboo](https://github.com/Shubhamsaboo)
- **Repository**: [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- **License**: Apache-2.0

### 本项目的主要改动

在原始项目基础上，本项目做了以下修改：

1. **代码结构**：从单文件拆分为 6 个模块化文件
2. **LLM**：从 Claude 替换为阿里云通义千问（`qwen-plus`）
3. **Embedding**：从 OpenAI `text-embedding-3-small` 替换为阿里云 `text-embedding-v3`
4. **向量库**：从 Qdrant 替换为 Chroma（本地持久化）
5. **兼容性修复**：解决了阿里云 Embedding 接口的兼容性问题

## 📝 License

Apache-2.0（与原项目一致）
