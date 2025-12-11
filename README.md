# DeepCode: 开放式智能体编程框架

本项目基于 DeepCode 开发，是一个集成了多智能体协同工作的自动化代码生成与编排系统。通过引入专门的研究分析、工作空间管理、文档处理及代码生成智能体，我们实现了从概念验证到生产级代码的全流程自动化。

## 🏗️ 架构 (Architecture)

我们的系统采用模块化多智能体架构，各组件协同工作以完成复杂的编程任务。

![Architecture](Architecture.png)

### 核心智能体组件

系统包含以下十大核心智能体/模块：

1.  **中心编排引擎 (Orchestrator)**
    *   **功能**: 负责统筹全局执行顺序，协调各智能体的工作流程，并进行多智能体管理，确保任务高效流转。

2.  **研究分析智能体 (ResearchAnalyzerAgent)**
    *   **功能**: 专注于解析用户输入，深入理解研究背景、目的与具体需求。

3.  **工作空间构建与资源处理智能体 (Workspace Infrastructure & Resource Processor)**
    *   **工作空间构建智能体 (Workspace Infrastructure Agent)**: 负责构建标准化的工作目录结构。
    *   **资源处理智能体 (ResourceProcessorAgent)**:
        *   将资源下载或移动到指定工作区。
        *   统一将 PPT、DOCX、TXT 等多格式输入转换为 PDF。
        *   基于 MinerU 模型提取文档中的图片、公式与表格。
        *   将处理后的 PDF 转换为结构化的 Markdown 文档。

4.  **文档分割智能体 (DocumentSegmentationAgent)**
    *   **功能**: 对 Markdown 文档进行语义分段，为后续的规划与检索阶段提供精确的文档内容接口。

5.  **代码规划智能体集 (Code Planning Agent Suite)**
    *   **组成**: 概念分析 (ConceptAnalysis) / 算法分析 (AlgorithmAnalysis) / 代码规划 (CodePlanner)。
    *   **功能**: 多模态深度解析文档，分析代码开发需求，并生成目标代码的详细实现规划。

6.  **参考情报与索引智能体集 (Reference Intelligence & Indexing Suite)**
    *   **组成**: 参考分析 (ReferenceAnalysis) / Github下载 (GithubDownload) / 代码实现参考 (CodeImplementation)。
    *   **功能**: 分析文档参考文献，检索并下载关联的 GitHub 仓库，生成已有代码库与目标代码结构的关联映射。

7.  **记忆智能体 (ConciseMemoryAgent)**
    *   **功能**: 精简代码实现与迭代过程中的上下文信息，在长对话中保持关键记忆。

8.  **代码实现智能体 (CodeImplementationAgent)**
    *   **功能**: 基于规划方案逐文件完成代码生成，并实时跟踪代码实现进度。

9.  **代码迭代智能体 (CodeIterationAgent)**
    *   **功能**: 支持多轮对话形式的代码迭代，根据用户反馈与需求定向优化代码。

10. **测试生成智能体 (TestGenerationAgent)**
    *   **功能**: 为已实现的代码生成完整测试套件，生成测试代码目录结构与执行说明，确保代码质量。

## 🚀 快速开始 (Quick Start)

### 🔽 克隆存储库
```bash
git clone https://github.com/liweiyan-li/DeepCode.git
cd DeepCode/
```

### 1. 环境配置

```bash
# 安装项目依赖
pip install -r requirements.txt
```
```bash
# 🔑 配置API密钥 (必需)
# 使用您的API密钥和base_url编辑mcp_agent.secrets.yaml
```
```bash
# 🔑 配置搜索API密钥用于Web搜索 (可选)
# 编辑mcp_agent.config.yaml设置
```



### 2. 运行

```bash
# 使用UV
uv run streamlit run ui/streamlit_app.py
# 或使用传统Python
streamlit run ui/streamlit_app.py
```




#### 🪟 **Windows用户: 额外的MCP服务器配置**

如果您使用Windows，可能需要在`mcp_agent.config.yaml`中手动配置MCP服务器:

```bash
# 1. 全局安装MCP服务器
npm i -g @modelcontextprotocol/server-brave-search
npm i -g @modelcontextprotocol/server-filesystem

# 2. 找到您的全局node_modules路径
npm -g root
```

然后更新您的`mcp_agent.config.yaml`使用绝对路径:

```yaml
mcp:
  servers:
    filesystem:
      command: "node"
      args: ["C:/Program Files/nodejs/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js", "."]
```
