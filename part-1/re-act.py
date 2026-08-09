import re
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

def get_user_info(user_id: str) -> str:
  db = {"u123": "Plan: Premium, Balance: $100", "u456": "Plan: Free, Balance: $0"}
  return db.get(user_id.strip(), "User not found.")

def calculate_discount(amount: str) -> str:
  clean_amount = float(amount.replace("$", ""))
  return f"${clean_amount * 0.80}"

known_actions = {
  "get_user_info": get_user_info,
  "calculate_discount": calculate_discount
}

system_prompt = """
You are an autonomous billing assistant. You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.

Use Thought to describe your reasoning.
Use Action to run one of the available tools, then return PAUSE.
Observation will be provided to you by the system after you run an action.

Available actions:
get_user_info:
e.g. Action: get_user_info: u123
calculate_discount:
e.g. Action: calculate_discount: 100
"""

action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE)

def call_llm(messages: list) -> str:
  prompt_text = ""
  for msg in messages:
    if msg["role"] == "system":
      prompt_text += f"System Rules:\n{msg['content']}\n\n"
    elif msg["role"] == "user":
      prompt_text += f"msg:\n{msg['content']}\n\n"
    elif msg["role"] == "assistant":
      prompt_text += f"{msg['content']}\n"

  prompt_text += "\n"


  response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents=prompt_text
  )
  
  return response.text
    

def execute_query(user_query:str , max_turn: int = 5):
  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Question: {user_query}"}
  ]

  for turn in range(max_turn):
    print(f"\n>>> TURN {turn + 1}")
    try:
      response_text = call_llm(messages)
    except Exception as e:
      print(f"Error: {e}")
      break

    if "Answer:" in response_text:
      print("\n🏁 Agent finished successfully.")
      return response_text.split("Answer:")[-1].strip()
    
    actions = action_re.findall(response_text)
    if not actions:
      print("\n⚠️ Error: Agent got stuck (No Action or Answer provided).")
      break
    
    action_name, action_input = actions[0]

    if(action_name not in known_actions):
      print(f"\n⚠️ Error: Hallucinated action: {action_name}")
      break 

    print(f"\n[System] Executing Tool: {action_name}({action_input})")
    observation = known_actions[action_name](action_input)
    print(f"\n[System] Observation: {observation}")

    messages.append({"role": 'assistant' , "content": response_text})
    messages.append({ "role": "user", "content": f"Observation: {observation}"})

  return "Error: Max turns reached."

if __name__ == "__main__":
  query = "What is the discounted balance for user u123?"
  # print(f"Query: {query}")

  # final_answer = execute_query(query)
  # print(f"Final Answer: {final_answer}")

  query2 = "give me the user info for u456?"
  final_answer = execute_query(query2)
  print(f"Final Answer: {final_answer}")
  