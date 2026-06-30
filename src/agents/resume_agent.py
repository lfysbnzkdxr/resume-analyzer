"""Resume Extractor Agent — extracts structured info from a PDF resume."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.agents.base import get_llm
from src.agents.utils import extract_json, invoke_with_retry
from src.tools.pdf_parser import parse_resume_pdf

SYSTEM_PROMPT = """你是一个专业的简历解析助手。你的任务是从简历PDF中提取结构化信息。

请使用 parse_resume_pdf 工具读取简历内容，然后提取以下JSON格式的信息：

{{
  "personal_info": {{"name": "姓名（如能找到）", "email": "邮箱", "phone": "电话"}},
  "summary": "个人简要总结",
  "skills": ["技能1", "技能2", ...],
  "experience": [
    {{"company": "公司名", "role": "职位", "duration": "时间范围", "highlights": ["要点1", "要点2"]}}
  ],
  "education": [
    {{"school": "学校名", "degree": "学位", "major": "专业"}}
  ],
  "projects": [
    {{"name": "项目名", "description": "描述", "technologies": ["技术栈"]}}
  ]
}}

注意：
- 如果某个字段找不到对应信息，使用空字符串或空列表。
- 只返回 JSON，不要包含 markdown 代码块标记或其他说明文字。
"""


def create_resume_agent(api_key: str):
    """Create and return a Resume Extractor agent executor."""
    llm = get_llm(api_key=api_key)
    tools = [parse_resume_pdf]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "请分析这份简历 PDF：{file_path}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


TEXT_SYSTEM_PROMPT = """你是一个专业的简历解析助手。你的任务是从简历文本中提取结构化信息。

请从以下简历文本中提取JSON格式的信息：

{{
  "personal_info": {{"name": "姓名（如能找到）", "email": "邮箱", "phone": "电话"}},
  "summary": "个人简要总结",
  "skills": ["技能1", "技能2", ...],
  "experience": [
    {{"company": "公司名", "role": "职位", "duration": "时间范围", "highlights": ["要点1", "要点2"]}}
  ],
  "education": [
    {{"school": "学校名", "degree": "学位", "major": "专业"}}
  ],
  "projects": [
    {{"name": "项目名", "description": "描述", "technologies": ["技术栈"]}}
  ]
}}

注意：
- 如果某个字段找不到对应信息，使用空字符串或空列表。
- 只返回 JSON，不要包含 markdown 代码块标记或其他说明文字。
"""


def create_text_resume_agent(api_key: str):
    """Create a resume agent that extracts from raw text (no PDF tool needed)."""
    llm = get_llm(api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", TEXT_SYSTEM_PROMPT),
        ("human", "请分析这份简历文本：\n\n{resume_text}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False, handle_parsing_errors=True)


def extract_resume_from_text(resume_text: str, api_key: str) -> dict:
    """Extract structured info from resume text directly. Returns a dict."""
    agent = create_text_resume_agent(api_key)
    return invoke_with_retry(agent, {"resume_text": resume_text})


def extract_resume(file_path: str, api_key: str) -> dict:
    """Extract structured info from a resume PDF. Returns a dict."""
    agent = create_resume_agent(api_key)
    return invoke_with_retry(agent, {"file_path": file_path})
