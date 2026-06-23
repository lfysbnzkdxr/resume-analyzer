"""JD Analyzer Agent — extracts structured requirements from job descriptions."""

import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.agents.base import get_llm
from src.agents.utils import extract_json, invoke_with_retry

SYSTEM_PROMPT = """你是一个专业的职位描述（JD）分析助手。你的任务是从职位描述文本中提取关键要求。

请提取以下JSON格式的信息：

{{
  "role_name": "职位名称",
  "required_skills": [
    {{"skill": "技能名", "level": "proficient/familiar/basic", "importance": "must"}}
  ],
  "preferred_skills": [
    {{"skill": "技能名", "level": "proficient/familiar/basic", "importance": "plus"}}
  ],
  "experience_required": "经验要求描述",
  "education_required": "学历要求描述",
  "responsibilities": ["职责1", "职责2"]
}}

判断标准：
- required_skills: JD中明确要求"必须"、"需要"、"要求"的技能
- preferred_skills: JD中"优先"、"加分"、"了解"等软性要求的技能
- level: 根据描述判断 proficient(精通) / familiar(熟悉) / basic(了解)

只返回 JSON，不要包含 markdown 代码块标记或其他说明文字。
"""


def create_jd_agent():
    """Create and return a JD Analyzer agent executor."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "请分析这份职位描述：\n\n{jd_text}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False, handle_parsing_errors=True)


def analyze_jd(jd_text: str) -> dict:
    """Analyze a job description and return structured requirements."""
    agent = create_jd_agent()
    return invoke_with_retry(agent, {"jd_text": jd_text})
