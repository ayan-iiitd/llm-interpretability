import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from rich import print as pp
import multiprocessing


def get_token_probs_per_layer(model_name, input_text, tokens_of_interest):
    
    
    """
    Analyze a model's hidden states to track token probabilities layer by layer.

    Args:
        model: The HuggingFace transformer model.
        tokenizer: The tokenizer for the model.
        input_text (str): The prompt to analyze.
        tokens_of_interest (list[str]): A list of specific tokens whose probability we want to track.

    Returns:
        dict: A results dictionary containing:
            - A list of dictionaries, one for each layer.
            - Each list item has the layer index, probabilities for the tokens of interest and the 5 tokens with the highest probability.
    """
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "auto",
        torch_dtype = "auto",
        trust_remote_code = True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ## Tokenize input
    inputs = tokenizer(input_text, return_tensors = "pt").to(model.device)
    
    ## Get token IDs for tokens of interest
    token_ids_of_interest = []
    for token in tokens_of_interest:
        token_id = tokenizer.encode(" " + token, add_special_tokens = False)[0] ## if word is tokenized into multiple tokens, so using the first one: {token_id[0]}
        token_ids_of_interest.append(token_id)
    
    
    generated_ids = model.generate(**inputs,
                                   max_new_tokens = 50,
                                   pad_token_id = tokenizer.eos_token_id)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens = True)
    
    ## Run model with hidden states output for jus the prompt
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states = True)
    
    ## Get the layer of the model that projects hidden states to vocabulary logits
    output_projection = model.get_output_embeddings()

    results = []

    ## For each layer find top probable tokens and probabilities of tokens of interest
    for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
        
        ## Get last token position hidden state
        last_token_hidden = layer_hidden_state[0, -1, :]
        
        ## Project to vocabulary space
        logits = output_projection(last_token_hidden)
        
        ## Convert to probabilities
        probs = torch.softmax(logits, dim = -1)
        
        ## Extract probabilities for tokens of interest
        token_probs = {token: probs[token_id].item()\
                       for token, token_id in zip(tokens_of_interest, token_ids_of_interest)}
        
        top_5_probs, top_5_ids = torch.topk(probs, 5)
        top_5_tokens = [
            {
                "token": tokenizer.decode(token_id),
                "probability": prob.item()
            }
            for token_id, prob in zip(top_5_ids, top_5_probs)
        ]

        results.append({
            "layer": layer_idx,
            "tokens_of_interest": token_probs,
            "top_5_tokens": top_5_tokens
        })
        
        results.append((layer_idx, token_probs))
    
    return results


def run_inference_worker_entry(model_name, eval_prompt, tokens_to_check, idx):
    
    results = get_token_probs_per_layer(model_name, eval_prompt, tokens_to_check)
    with open(f'../results/token_prob/phi/sample_{idx + 1}.json', 'w') as file_writer:
        json.dump(results, file_writer)



if __name__ == '__main__':

    path_to_all_data = "../data/text_with_entities.json"

    with open(path_to_all_data, 'r') as json_file:
        ddi_data = json.load(json_file)

    pattern = r'[\x00-\x1F\x7F]'
    model_name = "microsoft/Phi-4-mini-instruct"

    multiprocessing.set_start_method('spawn', force = True)

    for idx, entry in enumerate(ddi_data):

        tokens_to_check = ddi_data[entry]['entities']
        eval_prompt = "The drugs, chemicals and medical entitites mentioned in the text. - \"" + ddi_data[entry]['full_text'] + "\" are the following: "
        eval_prompt = re.sub(pattern, '', eval_prompt)

        process = multiprocessing.Process(target = run_inference_worker_entry,
                                        args = (model_name, eval_prompt, tokens_to_check, idx))
        
        process.start()
        process.join() # Wait for the current inference process to complete

        if process.exitcode != 0:
            pp(f"[bold red]Error processing sample {idx}. Worker exited with code {process.exitcode}.[/bold red]")

        pp(f"[bold green]Finished processing sample {idx}. GPU memory should now be released.[/bold green]")
        print("-" * 70)
        

    print("All samples processed.")