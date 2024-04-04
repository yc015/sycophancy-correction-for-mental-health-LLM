import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import torch
from tqdm.auto import tqdm
from collections import OrderedDict


class TextDataset(Dataset):
    def __init__(self, directory, tokenizer, model, chats, labels,
                 convert_to_llama2_format=False, user_identifier="HUMAN:", ai_identifier="ASSISTANT:",
                 additional_datas=None, new_format=False,
                 include_inst=False, one_hot=False):
        self.tokenizer = tokenizer
        self.chats = chats
        self.labels = labels
        self.acts = []
        self.texts = []
        self.label_idf = label_idf
        self.label_to_id = label_to_id
        self.model = model
        self.convert_to_llama2_format = convert_to_llama2_format
        self.user_identifier = user_identifier
        self.ai_identifier = ai_identifier
        self.new_format = new_format
        self.include_inst = include_inst
        self.one_hot = one_hot
        self._load_in_data()

    def __len__(self):
        return len(self.texts)
    
    def _load_in_data(self):
        for idx in tqdm(range(len(self.chats))):
            if self.convert_to_llama2_format:
                messages_dict = self.chats[idx]
                try:
                    text = llama_v2_prompt(messages_dict)
                except Exception as e:
                    print(e)
                    continue
              
            if self.new_format and self.if_remove_last_ai_response and self.include_inst:
                text = text[text.find("<s>") + len("<s>"):]
            elif self.new_format and self.include_inst:
                text = text[text.find("<s>") + len("<s>"):]
            elif self.new_format:
                text = text[text.find("<s>") + len("<s>"): text.rfind("[/INST]") - 1]
                    
            with torch.no_grad():
                encoding = self.tokenizer(
                  text,
                  truncation=True,
                  max_length=2048,
                  return_attention_mask=True,
                  return_tensors='pt'
                )
                
                features = OrderedDict()
                for name, module in self.model.named_modules():
                    if name.endswith(".mlp") or name.endswith(".embed_tokens"):
                        features[name] = ModuleHook(module)
                        
                output = self.model(input_ids=encoding['input_ids'].to("cuda"),
                                    attention_mask=encoding['attention_mask'].to("cuda"),
                                    output_hidden_states=True,
                                    return_dict=True)
                for feature in features.values():
                    feature.close()
            
            last_acts = []
            for layer_num in range(41):
                last_acts.append(output["hidden_states"][layer_num][:, -1].detach().cpu().clone().to(torch.float))
            last_acts = torch.cat(last_acts)
            
            self.texts.append(text)
            self.acts.append(last_acts)
            
        for path in corrupted_file_paths:
            self.file_paths.remove(path)
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]

        hidden_states = self.acts[idx]
        
        return {
            'hidden_states': hidden_states,
            'label': label,
            'text': text,
        }
    