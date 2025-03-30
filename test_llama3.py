from eagle.model.ea_model import EaModel
import torch

def reformat_llama_prompt(text):
    """
    Remove the "Cutting Knowledge Date" and "Today Date" lines from the text. \n
    Add a newline before the "<|start_header_id|>user<|end_header_id|>" marker.
    """
    marker_user = "<|start_header_id|>user<|end_header_id|>"
    marker_assistant = "<|start_header_id|>assistant<|end_header_id|>"
    
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Cutting Knowledge Date:"):
            i += 1
            continue
        elif lines[i].startswith("Today Date:"):
            i += 1
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        else:
            if marker_user in lines[i]:
                modified_line = lines[i].replace(marker_user, "\n"+marker_user)
                result.append(modified_line)
            else:
                result.append(lines[i])
            i += 1
            
    if result:
        result[-1] = result[-1] + marker_assistant
        
    return "\n".join(result)

def apply_template(input:str=None, tokenizer=None):
    if None in (input, tokenizer):
        raise ValueError("input and tokenizer cannot be None")
    if type(input) != str:
        raise TypeError("input must be a string")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible."},
        {"role": "user", "content": input}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    formatted_input = reformat_llama_prompt(formatted_input)
    
    # Debugging: Print the formatted input
    print(formatted_input)
    
    return formatted_input


# Model candidates
model_list = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'yuhuili/EAGLE-LLaMA3.1-Instruct-8B',
    'yuhuili/EAGLE-LLaMA3-Instruct-8B'
]
   
    
def main():
    base_model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    EAGLE_model_path = 'yuhuili/EAGLE-LLaMA3-Instruct-8B'

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )
    model.eval()
    tokenizer = model.tokenizer

    # Format the input text
    input_text = "if x+7=10, what is x?"
    formatted_input = apply_template(input_text, tokenizer)
    input_ids=model.tokenizer([formatted_input]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()

    output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512, is_llama3=True)
    output=model.tokenizer.decode(output_ids[0])

    # Print out the result
    print(output)
    
    
if __name__ == "__main__":
    main()

