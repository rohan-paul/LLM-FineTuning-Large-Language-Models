#######################################################
🦙 `apply_chat_template` in HuggingFace is just great. 🔥🚀
#######################################################

📌 Why templates? ❓

An increasingly common use case for LLMs is chat. In a chat context, rather than continuing a single string of text (as is the case with a standard language model), the model instead continues a conversation that consists of one or more messages, each of which includes a role as well as message text.

Chat models have been trained with very different formats for converting conversations into a single tokenizable string. Using a format different from the format a model was trained with will usually cause severe, silent performance degradation, so matching the format used during training is extremely important!

So, `apply_chat_template` attribute can be used to save the chat format the model was trained with. This attribute contains a Jinja template that converts conversation histories into a correctly formatted string.

#######################################################
### Structure of the `history` Variable
#######################################################

📌 The history variable in your script is structured as a list of dialogue pairs, where each pair consists of a user's message and the assistant's response. It's a sequential record of the conversation. Here is an example of its structure in JSON format:

```
[
    ["User's first instruction", "Assistant's first response"],
    ["User's second instruction", "Assistant's second response"],
    ["User's third instruction", null]
]

```


### Structure within `format_chat_history` Function

In the `format_chat_history` function, the `messages` list is created using dictionaries to format each message. This is where the curly braces `{}` appear. Each message is a dictionary with two keys: `role` and `content`.

```python
messages = []
for dialog in history[:-1]:
    instruction, response = dialog[0], dialog[1]
    messages.extend([
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ])
new_instruction = history[-1][0].strip()
messages.append({"role": "user", "content": new_instruction})
```

In this snippet:

- Each `dialog` from `history` is processed.
- For each `dialog`, two dictionaries are created and added to the `messages` list.
- Each dictionary represents a message, where:
  - `{"role": "user", "content": instruction}` represents the user's message.
  - `{"role": "assistant", "content": response}` represents the assistant's response.

The curly braces `{}` are used to create these dictionaries, which are then formatted into a prompt suitable for the Mixtral model. The `history` variable itself, however, remains a list of lists and does not use curly braces.

==============

The conditional `(if dialog[i % 2])` ensures that messages that are None (like the latest assistant response in an ongoing conversation) are not included.

In a typical conversation history, the latest response from the assistant might not exist yet (i.e., it's None) because the user just sent a message and the assistant hasn't replied yet. This check ensures that such None values are not included in the formatted output, as they don't represent actual messages and would not be meaningful input for the model.

#######################################################
## format_chat_history(history)
#######################################################

**Input Structure**: The function expects `history` as an input, which is a list of dialogues. Each dialogue in the history is a tuple with two elements: the first element is the user's instruction, and the second is the assistant's response. The most recent dialogue will have `None` as the response since it's the current user query awaiting a response.

**Processing Steps**:
   - The function iterates over each dialogue in the history.
   - For each dialogue, it extracts the user instruction and the assistant's response.
   - These are then formatted into a list of dictionaries, where each dictionary represents a message with two keys: `role` and `content`. The `role` can be either `"user"` or `"assistant"`, indicating who said what, and `content` is the actual text of the message.

**Formatting the Prompt**: After processing the history, the function uses the `apply_chat_template` method of the tokenizer. This method takes the formatted messages and converts them into a prompt string. This string is a properly structured input that the Mixtral model can process.

**Tokenization and Generation Prompt**: The `apply_chat_template` method is called with `tokenize=False` and `add_generation_prompt=True`. This means that the function will format the messages into a string without tokenizing them (since tokenization is typically handled by the model pipeline later) and will add a generation prompt at the end. This generation prompt is a cue for the model to start generating a response.


#######################################################
## `thread = Thread(target=pipeline, kwargs=kwargs)`
#######################################################

The use of threading in this context is aimed at preventing the UI from freezing during the execution of computationally intensive tasks, though the specific implementation here does raise some questions about its effectiveness in achieving truly asynchronous behavior.

1. **Asynchronous Operation**:
   - The `pipeline` function, which generates responses using the Mixtral model, can be time-consuming, especially for complex text generation tasks. Running this function synchronously (i.e., in the main thread) would block the execution of other parts of the code until it completes. This can lead to a frozen or unresponsive user interface, especially in a GUI application like one using Gradio.

2. **Maintaining UI Responsiveness**:
   - By using a separate thread for the `pipeline` function, the main thread (which handles the user interface) remains free to respond to user inputs and other events. This way, the application remains responsive even while the model is processing.

### Analysis of the Thread Implementation

- `Thread(target=pipeline, kwargs=kwargs)`: This line creates a new thread object, setting the `pipeline` function as the target to be executed in this thread. The `kwargs` are the arguments passed to the `pipeline` function.

- `thread.start()`: This initiates the execution of the `pipeline` function in a new thread, separate from the main thread.