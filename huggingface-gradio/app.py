import gradio as gr
import spaces
import torch
from transformers import pipeline
import datetime
import json
import logging

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Load model for cache
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

@spaces.GPU
def classify(query):
    torch_device = 0 if torch.cuda.is_available() else -1
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=torch_device)

    request_type = type(query)
    try:
        data = json.loads(query)
        if type(data) != list:
            data = [query]
        else:
            request_type = type(data)
    except Exception as e:
        print(e)
        data = [query]
        pass

    start_time = datetime.datetime.now()
    
    result = sentiment_task(data, batch_size=128, top_k=3)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    logging.debug("elapsed predict time: %s", str(elapsed_time))
    print("elapsed predict time:", str(elapsed_time))
    
    output = {}
    output["time"] = str(elapsed_time)
    output["device"] = torch_device
    output["result"] = result

    return json.dumps(output)

demo = gr.Interface(fn=classify, inputs="text", outputs="text")
demo.launch()