import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent
from tools import order_tool, policy_tool

load_dotenv()

# Initialize Gemini LLM
llm = GoogleGenAI(model="gemini-3.5-flash")

system_prompt = (
    "You are KartEase support, a helpful customer service assistant for an Indian e-commerce store. "
    "Use your tools (search_policies and get_order_status) to answer customer questions. "
    "Never guess policy details or order statuses—always use the tools. "
    "If a question has nothing to do with KartEase policies or orders, politely reply: "
    "'Sorry, I can only help with KartEase orders and policies.'"
)

# Create the LlamaIndex FunctionAgent workflow
agent = FunctionAgent(
    llm=llm,
    tools=[order_tool, policy_tool],
    system_prompt=system_prompt
)

# Async loop helper required for LlamaIndex agents
loop = asyncio.new_event_loop()

async def _run(question):
    return await agent.run(user_msg=question)

def ask_agent(question):
    result = loop.run_until_complete(_run(question))
    return result.response.content

if __name__ == "__main__":
    print("=== KartEase Support Agent is Online! Type 'exit' to quit. ===")
    while True:
        try:
            q = input("\nCustomer: ")
            if q.strip().lower() == "exit":
                break
            if not q.strip():
                continue
            answer = ask_agent(q)
            print(f"\nAgent:\n{answer}")
        except Exception as e:
            print(f"Error: {e}")