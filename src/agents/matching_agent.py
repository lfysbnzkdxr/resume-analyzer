"""Matching & Evaluation Agent — compares resume against JD and generates scores."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.agents.base import get_llm
from src.agents.utils import extract_json, invoke_with_retry
from src.tools.skill_matcher import calculate_skill_overlap

SYSTEM_PROMPT = """你是一个专业的简历匹配评估专家。你的任务是比较简历和职位描述，给出匹配度评分和改进建议。

请严格按照以下 JSON 格式输出：

{{
  "overall_score": 0-100的整数,
  "dimensions": [
    {{"name": "技能匹配", "score": 0-100, "weight": 0-1间的小数, "details": "详细分析"}},
    {{"name": "经验匹配", "score": 0-100, "weight": 0-1间的小数, "details": "详细分析"}},
    {{"name": "教育背景", "score": 0-100, "weight": 0-1间的小数, "details": "详细分析"}},
    {{"name": "项目匹配", "score": 0-100, "weight": 0-1间的小数, "details": "详细分析"}}
  ],
  "suggestions": [
    {{"category": "skill_gap/presentation/format/experience", "priority": "high/medium/low", "content": "具体建议"}}
  ],
  "summary": "总体评价"
}}

评分标准：
- 90-100: 高度匹配，几乎完美
- 70-89: 良好匹配，有少数差距
- 50-69: 部分匹配，有明显差距
- 20-49: 弱匹配，有较大差距但存在可迁移技能（例如：Java后端转AI算法，JD要求Python但候选人有Java工程经验+Docker+Kafka+计算机专业，应评25-45）
- 0-19: 基本不匹配

对于跨技术栈转行场景：
- 编程语言是互通的（会Java等于有编程基础），不要因为语言不同就打0分
- 工程工具（Docker、Redis、Kafka、MySQL）在任何技术栈中都有价值
- 计算机相关专业教育背景应给予认可

dimensions 的 weight 总和必须为 1.0。

你可以使用 calculate_skill_overlap 工具计算技能重合度来辅助分析。

只返回 JSON，不要包含 markdown 代码块标记或其他说明文字。"""


def create_matching_agent(api_key: str):
    """Create and return a Matching & Evaluation agent executor."""
    llm = get_llm(api_key=api_key)
    tools = [calculate_skill_overlap]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", (
            "请评估以下简历与职位描述的匹配度。\n\n"
            "--- 简历信息 ---\n{resume_info}\n\n"
            "--- 职位描述要求 ---\n{jd_info}"
        )),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def evaluate_match(resume_info: str, jd_info: str, api_key: str) -> dict:
    """Evaluate resume-JD match. Returns scores and suggestions."""
    agent = create_matching_agent(api_key)
    return invoke_with_retry(agent, {"resume_info": resume_info, "jd_info": jd_info})
