from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

#model setup 
model_name = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
llm = ChatGroq(
    model=model_name,
    temperature=0,
    timeout=60,
    max_tokens=8192,
    **({"reasoning_effort": "low"} if model_name.startswith("openai/gpt-oss-") else {}),
)


def research_output(result):
    """Preserve retrieved evidence when the agent returns an empty summary."""
    messages = result.get("messages", [])
    if messages and isinstance(messages[-1].content, str) and messages[-1].content.strip():
        return messages[-1].content
    evidence = [
        message.content for message in messages
        if getattr(message, "type", "") == "tool"
        and getattr(message, "status", "success") != "error"
        and isinstance(message.content, str)
        and message.content.strip()
        and not message.content.startswith("Could not scrape URL:")
    ]
    if evidence:
        return "\n\n".join(evidence)
    raise ValueError("Research agent returned no usable content. Please retry.")


#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search],
        system_prompt="Search the web for the requested topic. Use one search unless it returns no relevant results. Then provide a concise factual summary including source URLs. Always return a non-empty final answer.",
    )

#2nd agent 

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        system_prompt="Read the most relevant supplied URL using scrape_url. Summarize the retrieved content and include the URL. If scraping fails, explain that limitation. Always return a non-empty final answer.",
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

