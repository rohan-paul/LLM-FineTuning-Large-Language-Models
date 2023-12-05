# !pip install streamlit transformers
# Run the whole app with below kind of command
# `streamlit run app.py`

import re
from threading import Thread
import streamlit as st
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM

# Constants
class Config:
    BASE_MODEL = "TheBloke/Phind-CodeLlama-34B-v2-GPTQ"
    Config.MODEL_MAX_LEN = 16384
    SYSTEM_PROMPT = "You are an AI coding assistant."
    GEN_LENGTH = 2048
    DEFAULT_PROMPT_LEN = None

st.set_page_config(page_title="Code Generation conversation", page_icon="🤗")

def load_models():
    """
    Loads the language model and tokenizer.
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL,
            device_map="auto",
            trust_remote_code=False,
            revision="gptq-4bit-32g-actorder_True"
        )
        tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

    return model, tokenizer

model, tokenizer = load_models()

def get_token_length(text):
    """
    Calculates the length of a given text in tokens.
    Args:
        text (str): Text to be tokenized.
    Returns:
        int: Length of the tokenized text.
    """
    return len(tokenizer(text)[0])

Config.DEFAULT_PROMPT_LEN = get_token_length(f"""### System Prompt:
{Config.SYSTEM_PROMPT}

### User Message:

### Assistant:""")

def create_conversation_pairs():
    """
    Creates conversation pairs from session messages.
    Returns:
        list: List of conversation pairs with token count.
    """
    conversation_history = []
    temp_dict = {}
    turn_token_len = 0
    for i in st.session_state.messages[1:]:
        role = i['role']
        content = i['content']
        tokenized_content = f"""### {role.capitalize()}:{content}</s>"""
        turn_token_len += get_token_length(tokenized_content)

        if role == "assistant":
            temp_dict["token_count"] = turn_token_len
            temp_dict['content'] += tokenized_content
            conversation_history.append(temp_dict)
            temp_dict = {}
            turn_token_len = 0
        else:
            temp_dict['content'] = tokenized_content

    return conversation_history

def get_prompt_with_history(instruction, max_tokens=Config.MODEL_MAX_LEN, generation_length=Config.GEN_LENGTH):
    """
    Creates a prompt for the model.
    Args:
        instruction (str): User instruction to be included in the prompt.
        max_tokens (int): Maximum token length for the model.
        generation_length (int): Length of the generation.
    Returns:
        str: The created prompt.
    """
    current_instruction_len = get_token_length(instruction)
    max_usable_tokens = max_tokens - generation_length - Config.DEFAULT_PROMPT_LEN - current_instruction_len
    conversation_history = create_conversation_pairs()
    conversation_history.reverse()

    usable_history = []
    history_len = 0
    for pair in conversation_history:
        history_len += pair['token_count']
        if history_len > max_usable_tokens:
            break
        usable_history.append(pair['content'])

    usable_history = "".join(reversed(usable_history))
    prompt = f"""### System Prompt:
{Config.SYSTEM_PROMPT}

{usable_history}

### User Message: {instruction}

### Assistant:"""
    return prompt

def generate_response(instruction, max_new_tokens=Config.GEN_LENGTH):
    """
    Generates a response from the model.
    Args:
        instruction (str): Instruction for generating the response.
        max_new_tokens (int): Maximum new tokens for the generation.
    Returns:
        str: Generated text.
    """
    prompt = get_prompt_with_history(instruction, max_new_tokens)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(inputs=inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    with st.empty():
        for idx, new_text in enumerate(streamer):
            generated_text += new_text
            generated_text = re.sub(r"</s>", "", generated_text)
            st.write(generated_text)
    return generated_text

def main():
    """
    Main function to handle the chat interface and response generation.
    """
    # Initialization
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help?"}]

    # Chat Interface
    # Displaying each message in the chat
    for message in st.session_state.messages:
        with st.container():
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input outside any container or layout widget
    user_input = st.chat_input()
    if user_input:
        # Append user message to the chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Generate and append the assistant's response
        generate_and_append_response(user_input)

def generate_and_append_response(user_input):
    """
    Generates a response for the given user input and appends it to the chat.
    Args:
        user_input (str): User's input text.
    """
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = generate_response(user_input)
            # remove any end-of-string tokens (`</s>`).
            # These tokens are used by language models to signify the end of a text
            # sequence, but they are not needed in the final output shown to the user.
            response = re.sub("</s>", "", response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# Run the application
if __name__ == "__main__":
    main()


"""
#####################################################
## `st.session_state`
#####################################################

📌 `st.session_state` in Streamlit is a powerful feature that allows you to maintain state across user interactions. When a Streamlit app is running, each user interaction, like clicking a button or entering text, typically causes the whole script to re-run. This can lead to a loss of state - for example, all variables are reset. `st.session_state` solves this problem by providing a way to store and persist values across reruns.

📌 Each `st.session_state` is unique to a user session. It behaves like a Python dictionary and can store any kind of Python object. You can set key-value pairs in this state, read them, and update them. This enables the app to remember information like user inputs, the state of interactions, or any other data that should persist across reruns.

📌 Here in this code, `st.session_state` is used to store the conversation history in a chat application. Every time a user enters a message, it's appended to the `messages` list in `st.session_state`. This list then persists across reruns, allowing the app to maintain the context of the conversation and display the entire chat history.

--------------

##############################################
## `thread = Thread(target=model.generate, kwargs=generation_kwargs)`
##############################################

The `target` parameter specifies the callable object to be invoked by the `run()` method of the thread.

In Python, multithreading allows multiple parts of a program to run concurrently. Each thread runs independently and can execute different parts of the code simultaneously.

When you create a new thread, you need to specify the function it will execute. This is where the `target` parameter is used.

The `target` parameter in the `Thread` class specifies the function that the thread will execute. So, `target=model.generate` means that the `generate` method of the `model` object will be run in a separate thread.

When the thread is started using `thread.start()`, the `run()` method of the thread is called. This method, in turn, calls the function specified in the `target` parameter, in this case, `model.generate`.

The arguments required by `model.generate` are passed through the `kwargs` parameter, ensuring that the method has all the necessary information to execute properly.

Without multithreading, calling `model.generate` directly in the main thread would block the entire execution of the program until the text generation is complete. This would make the Streamlit app unresponsive.


"""