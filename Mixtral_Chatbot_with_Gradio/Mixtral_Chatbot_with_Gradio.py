from transformers import AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
import gradio as gr
import transformers
import torch

# Run the entire app with `python run_mixtral.py`

""" The messages list should be of the following format:

messages =

[
    {"role": "user", "content": "User's first message"},
    {"role": "assistant", "content": "Assistant's first response"},
    {"role": "user", "content": "User's second message"},
    {"role": "assistant", "content": "Assistant's second response"},
    {"role": "user", "content": "User's third message"}
]

"""
""" The `format_chat_history` function below is designed to format the dialogue history into a prompt that can be fed into the Mixtral model. This will help understand the context of the conversation and generate appropriate responses by the Model.
The function takes a history of dialogues as input, which is a list of lists where each sublist represents a pair of user and assistant messages.
"""

def format_chat_history(history) -> str:
    messages = [{"role": ("user" if i % 2 == 0 else "assistant"), "content": dialog[i % 2]}
                for i, dialog in enumerate(history) for _ in (0, 1) if dialog[i % 2]]
    # The conditional `(if dialog[i % 2])` ensures that messages
    # that are None (like the latest assistant response in an ongoing
    # conversation) are not included.
    return pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True)

def model_loading_pipeline():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True,
                      "quantization_config": BitsAndBytesConfig(
                                                                load_in_4bit=True,
                                                                bnb_4bit_compute_dtype=torch.float16)},
        streamer=streamer
    )
    return pipeline, streamer

def launch_gradio_app(pipeline, streamer):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            prompt = format_chat_history(history)

            history[-1][1] = ""
            kwargs = dict(text_inputs=prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            thread = Thread(target=pipeline, kwargs=kwargs)
            thread.start()

            for token in streamer:
                history[-1][1] += token
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(share=True, debug=True)

if __name__ == '__main__':
    pipeline, streamer = model_loading_pipeline()
    launch_gradio_app(pipeline, streamer)

# Run the entire app with `python run_mixtral.py`
