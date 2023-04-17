import argparse
import json
import os
import requests
import time

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")


esco_indices = ["skills", "occupations", "skillGroups"]

document_types = ["job description", "user profile"]
esco_fields = { 
    "occupations": ["rank", "preferredLabel", "conceptType", "code", "altLabels", "description", "conceptUri"],
    "skillGroups": ["rank", "preferredLabel", "conceptType", "code", "altLabels", "description", "conceptUri"],
    "skills": ["rank", "preferredLabel", "conceptType", "skillType", "altLabels", "description", "conceptUri"]
}

notice_markdown = ("""
# SkillGPT
### A RESTful API service for skill extraction and standardization from job descriptions and user profiles using large language model
Nan Li, Bo Kang, and Tijl De Bie
IDLAB - Department of Electronics and Information Systems (ELIS), Ghent University, Belgium
""")

learn_more_markdown = ("""
#### &copy; 2023 Ghent University Artificial Intelligence & Data Analytics Group
""")
                       
css = """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""

def summrize(text, document_type):    
    prompt = f"""
### Human: I want you to act as a human resource expert and summarize the top five skills from the following {document_type} using the same language:
----
{text}
----
### Assistant:          
"""
    sep = "###"
    worker_addr = "http://127.0.0.1:21002"
    headers = {"User-Agent": "SkillGPT Client"}
    pload = {
        "model": "vicuna-13b",
        "prompt": prompt,
        "max_new_tokens": 500,
        "temperature": 0.7,
        "stop": sep,
    }
    response = requests.post(worker_addr + "/generate_stream", headers=headers,
            json=pload, stream=False)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(sep)[-1]
    return output[len("Assistant: "):].strip()

def format_esco_concepts(esco_concepts, esco_index):
    df_res = pd.DataFrame.from_records([json.loads(concept_str) for concept_str in esco_concepts])
    df_res["rank"] = range(1, 1+len(df_res))
    return df_res[esco_fields[esco_index]]


def label(esco_index, text):
    prompt = f"""
{text}
"""
    sep = "###"
    worker_addr = "http://127.0.0.1:21002"
    headers = {"User-Agent": "SkillGPT Client"}
    pload = {
        "model": "vicuna-13b",
        "prompt": prompt,
        "esco_index": esco_index,
        "redis_host": REDIS_HOST,
        "num_relevant": 10,
    }
    response = requests.post(worker_addr + "/label_text", headers=headers,
            json=pload, stream=False)
    return format_esco_concepts(json.loads(response.content)["labels"], esco_index)

def add_text(state, text, document_type, request: gr.Request):
    summary = summrize(text, document_type)
    return (state, summary)

def label_text(state, text, esco_index, request: gr.Request):
    df_labels = label(esco_index, text)
    return (state, df_labels)


def load_demo(url_params, request: gr.Request):
    state = None
    return (state,
            gr.Textbox.update(visible=True),
            gr.Radio.update(visible=True),            
            gr.Button.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Radio.update(visible=True),
            gr.Button.update(visible=True),
            gr.Dataframe.update(visible=True),
           )

def build_demo():
    with gr.Blocks(title="SkillGPT", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()

        # Draw layout
        notice = gr.Markdown(notice_markdown)
        url_params = gr.JSON(visible=False)

        textbox = gr.Textbox(placeholder="Enter text and press ENTER", visible=False, label="Document") 
        
        with gr.Row():
            with gr.Column(scale=20):
                document_type_selector = gr.Radio(choices=document_types, value=document_types[0] if len(document_types) > 0 else "", interactive=True, label="Document type")
            with gr.Column(scale=2, min_width=50):
                summarize_btn = gr.Button(value="Summarize", visible=False)            
                   
        summarybox = gr.Textbox(visible=False, label="Summary")                
        with gr.Row():
            with gr.Column(scale=20):
                esco_selector = gr.Radio(choices=esco_indices, value=esco_indices[0] if len(esco_indices) > 0 else "", interactive=True, label="ESCO concept type")                
            with gr.Column(scale=2, min_width=50):
                label_btn = gr.Button(value="Extract", visible=False)
        escoframe = gr.Dataframe(visible=False, label="ESCO concepts", headers=esco_fields[esco_indices[0]])
        gr.Markdown(learn_more_markdown)
        
        # Register listeners
        textbox.submit(add_text, [state, textbox, document_type_selector], [state, summarybox])
        summarize_btn.click(add_text, [state, textbox, document_type_selector], [state, summarybox])
        
        label_btn.click(label_text, [state, summarybox, esco_selector], [state, escoframe])
        
        demo.load(load_demo, [url_params], [state, textbox, document_type_selector, summarize_btn, summarybox, esco_selector, label_btn, escoframe])
    return demo
        

gr.close_all()
demo = build_demo()
demo.launch(inline=True, share = True)