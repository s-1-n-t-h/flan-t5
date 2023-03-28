from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import re
import runpod

def handler(event):
    device = 0 if torch.cuda.is_available() else -1
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained('braindao/flan-t5-cnn',from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained('braindao/flan-t5-cnn')
    model = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)

    # Parse out your arguments
    prompt = event['input'].get('content',None)

    CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\\r|\\n|\s{2,}')

    # cleaning text beforing feeding to model
    if prompt != None:
        prompt = re.sub(CLEANER, '', prompt)

    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    result = model(prompt)
    return result

runpod.serverless.start({"handler": handler})
