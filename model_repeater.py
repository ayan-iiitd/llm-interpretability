from model_runner import run_model

print('started')

model_names = ['microsoft/Phi-4-mini-instruct',
               'meta-llama/Llama-3.2-3B-Instruct',
               'google/gemma-3-4b-it']

for model_name in model_names:
    run_model(model_name)