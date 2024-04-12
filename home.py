from langchain.tools import DuckDuckGoSearchResults
import json
import streamlit as st
import openai
import time
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

st.set_page_config(
    page_title="AssistantGPT",
    page_icon="💼",
)

st.markdown(
    """
    # AssistantGPT
            
    Welcome to AssistantGPT.
            
    Write down the issue and our Assistant will do the research for you.
"""
)

with st.sidebar:
    user_api_key = st.text_input("Please enter your API key on app page")

    if user_api_key:
        st.session_state['api_key'] = user_api_key
        st.write("API Key Complete")

        api_key = st.session_state.get('api_key', None)
        client = openai.OpenAI(api_key=api_key)

    else:
        st.stop()

# Tools
def get_issue_from_ddg(inputs):
    ddg = DuckDuckGoSearchResults()
    issue = inputs["issue"]
    return ddg.run(issue)

def get_issue_from_wikipedia(inputs):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    issue = inputs["issue"]
    return wikipedia.run(issue)

functions_map = {
    "get_issue_from_ddg": get_issue_from_ddg,
    "get_issue_from_wikipedia": get_issue_from_wikipedia,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_issue_from_ddg",
            "description": "Use this tool to find the Issue using DuckDuckGoSearch",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "The name of the issue",
                    }
                },
                "required": ["issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_issue_from_wikipedia",
            "description": "Use this tool to find the Issue using Wikipedia",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "The name of the issue",
                    }
                },
                "required": ["issue"],
            },
        },
    },
]

def setup_openai_assistant():
    if 'assistant' not in st.session_state:
        st.session_state.assistant = client.beta.assistants.create(
            name="Investor Assistant For Streamlit(Assign)",
            instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
            model="gpt-4-1106-preview", # https://platform.openai.com/playground?mode=assistant : Model List
            tools=functions,
        )

    return st.session_state.assistant

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        content = message.content[0].text.value.replace("$", "\$")

        st.write(f"{message.role}: {content}")

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []

    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        
        st.write(f"Calling function: {function.name} with arg {function.arguments}")
        
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )

if api_key:
    query = st.text_input("Write the issue you are interested on.")

    if query:
        assistant = setup_openai_assistant()

        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ]
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        status = get_run(run.id, thread.id).status

        while status != "completed":
            if status == "requires_action":
                submit_tool_outputs(run.id, thread.id)

            time.sleep(2)

            status = get_run(run.id, thread.id).status

        get_messages(thread.id)
