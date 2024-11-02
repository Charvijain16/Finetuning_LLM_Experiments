llama_3_2_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>"""

llama_3_1_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>"""


qwen_2_5_prompt = """ """


mistral_prompt = """ """
ministral_prompt = """ """


hermes_2_pro_llama3_prompt = """ """

hammer_2_prompt = """ """

tool_ace_prompt= """ """

xlam_prompt = """ """


def get_prompt_template(model_name):
    if "Llama-3.2" in model_name:
        return llama_3_2_prompt
    elif "Llama-3.1" in model_name:
        return llama_3_1_prompt
    elif "Mistral" in model_name:
        return mistral_prompt
    elif "Ministral" in model_name:
        return ministral_prompt
    elif "ToolACE" in model_name:
        return tool_ace_prompt
    elif "xLAM" in model_name:
        return xlam_prompt
    elif "Qwen2.5" in model_name:
        return qwen_2_5_prompt
    elif "Hammer2.0" in model_name:
        return hammer_2_prompt
    elif "Hermes-2-Pro" in model_name:
        return hermes_2_pro_llama3_prompt
    else:
        raise Exception("the prompt template is not present for the asked model, please add prompt template!")