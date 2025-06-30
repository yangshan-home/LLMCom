from modelscope import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM as ms_amfcl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer as trans_autoTokenizer
import numpy as np
import modelscope
import transformers

device = "cuda"


def chatglm3_6b(text_list, name='temp'):
    model_dir = "/T20050027/ShanYang/llms/chatglm3-6b/"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
    model = model.eval()

    reuslt_list = []
    for num, text in enumerate(text_list):
        input_ids = tokenizer.encode(text, add_special_tokens=True)

        # tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # print(tokens)

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).cuda()

        with torch.no_grad():
            embedding_layer = model.get_input_embeddings()
            word_embeddings = embedding_layer(input_ids_tensor)
            word_embeddings = word_embeddings[:, 2:, :]
            average_embedding = word_embeddings.mean(dim=1, keepdim=True)
        embedding = average_embedding[0][0].tolist()
        reuslt_list.append(' '.join([str(x) for x in embedding]))

    with open(
            f'/T20050027/ShanYang/paper_second/paper2/dataset/real_networks_by_llms/{name}_embedding_by_chatglm3_6b.txt',
            'w', encoding='utf-8') as f1:
        for line in reuslt_list:
            f1.write(line.strip() + '\n')


def glm4_9b_chat(data_list, name='temp'):
    tokenizer = trans_autoTokenizer.from_pretrained("/T20050027/ShanYang/llms/glm-4-9b-chat/", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "/T20050027/ShanYang/llms/glm-4-9b-chat/",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

    reuslt_list = []
    for num, query in enumerate(data_list):
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_tensors="pt",
                                               return_dict=True
                                               )
        input_ids = inputs['input_ids'].to(device)
        temp_input_ids = inputs['input_ids'].tolist()[0]
        print(temp_input_ids)
        tokens = tokenizer.convert_ids_to_tokens(temp_input_ids)
        print(tokens, "::::", tokens[4].decode('utf-8'), tokens[5].decode('utf-8'), tokens[6].decode('utf-8'),
              tokens[7].decode('utf-8'))
        # inputs = inputs.to(device)

        with torch.no_grad():
            embedding_layer = model.get_input_embeddings()
            word_embeddings = embedding_layer(input_ids)
            word_embeddings = word_embeddings[:, 4:-1, :]
            average_embedding = word_embeddings.mean(dim=1, keepdim=True)
        embedding = average_embedding[0][0].tolist()
        reuslt_list.append(' '.join([str(x) for x in embedding]))

    with open(
            f'/T20050027/ShanYang/paper_second/paper2/dataset/real_networks_by_llms/{name}_embedding_by_glm4_9b_chat.txt',
            'w', encoding='utf-8') as f1:
        for line in reuslt_list:
            f1.write(line.strip() + '\n')


def llama31_70b_instruct(data_list, name='temp'):
    model_id = "/T20050027/ShanYang/llms/Meta-Llama-3.1-70B-Instruct/"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    quantized_model = modelscope.AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with open(
            f'/T20050027/ShanYang/paper_second/paper2/dataset/real_networks_by_llms/{name}_embedding_by_llama31_70b_instruct.txt',
            'w', encoding='utf-8') as f1:
        for num, input_text in enumerate(data_list):
            input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
            # temp_input_ids = input_ids['input_ids'].tolist()[0]
            # tokens = tokenizer.convert_ids_to_tokens(temp_input_ids)

            with torch.no_grad():
                outputs = quantized_model.base_model(**input_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            embedding = embeddings[0].tolist()
            f1.write(' '.join([str(x) for x in embedding]).strip() + '\n')


def llama31_8b_instruct(data_list, name='temp'):
    model_id = "/T20050027/ShanYang/llms/Meta-Llama-3.1-8B-Instruct/"
    model = modelscope.AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    reuslt_list = []
    for num, input_text in enumerate(data_list):
        print(len(input_text.split(' ')))
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        temp_input_ids = input_ids['input_ids'].tolist()[0]
        tokens = tokenizer.convert_ids_to_tokens(temp_input_ids)

        with torch.no_grad():
            outputs = model.base_model(**input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        embedding = embeddings[0].tolist()
        reuslt_list.append(' '.join([str(x) for x in embedding]))
    with open(
            f'/T20050027/ShanYang/paper_second/paper2/dataset/real_networks_by_llms/{name}_embedding_by_llama31_8b_instruct.txt',
            'w', encoding='utf-8') as f1:
        for line in reuslt_list:
            f1.write(line.strip() + '\n')


def qwen2_72b_instruct(data_list, name='temp'):
    model = ms_amfcl.from_pretrained(
        "/T20050027/ShanYang/llms/Qwen2.5-72B-Instruct/",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("/T20050027/ShanYang/llms/Qwen2.5-72B-Instruct/")

    with open(
            f'/T20050027/ShanYang/paper_second/paper2/dataset/real_networks_by_llms/{name}_embedding_by_qwen2.5_72b_instruct.txt',
            'w', encoding='utf-8') as f1:
        for num, prompt in enumerate(data_list):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.base_model(**model_inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  #  average pooling layer
            embedding = embeddings[0].tolist()
            f1.write(' '.join([str(x) for x in embedding]).strip() + '\n')



if __name__ == '__main__':
    name = 'citeseer_nlp'
    with open(f'.././dataset/real_networks_by_llms/{name}.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data_list = []
        for line in lines:
            data_list.append(line.strip())
    # chatglm3_6b(data_list, name=name)
    # glm4_9b_chat(data_list, name=name)
    # llama31_70b_instruct(data_list, name=name)
    # llama31_8b_instruct(data_list, name=name)
    qwen2_72b_instruct(data_list, name=name)
