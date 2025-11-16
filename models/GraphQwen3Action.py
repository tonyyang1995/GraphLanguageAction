import torch
import torch.nn as nn 
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from models.gnn_modules import GNN_node, GNN_node_Virtualnode

from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os

HF_TOKEN=""
# define HF download location
os.environ["HF_HOME"] = "/media/volume/HuggingFace"
os.environ["HUGGINGFACE_HUB_CACHE="] = "/media/volume/TonySpaIM/huggingface_cache"


class GraphQwen3Action(nn.Module):
    def __init__(self, configs):
        super(GraphQwen3Action, self).__init__()
        self.graph_encoder = GNN_node_Virtualnode(
            num_layer=configs["model"]["graph_encoder"]["num_layers"],
            emb_dim=configs["model"]["graph_encoder"]["emb_dim"],
            JK=configs["model"]["graph_encoder"]["JK"],
            drop_ratio=configs["model"]["graph_encoder"]["drop_ratio"],
            residual=configs["model"]["graph_encoder"]["residual"],
            gnn_name=configs["model"]["graph_encoder"]["gnn_name"],
        )

        self.lm_encoder = AutoModelForCausalLM.from_pretrained(
            configs["model"]["Qwen3"]["model_path"],
            token=HF_TOKEN
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            configs["model"]["Qwen3"]["model_path"],
            token=HF_TOKEN
        )
        self.lm_dim = self.lm_encoder.get_input_embeddings().weight.shape[1]
        
        self.graph_proj = nn.Linear(self.graph_encoder.emb_dim, self.lm_dim)

        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(configs["model"]["graph_encoder"]["emb_dim"], 2 * configs["model"]["graph_encoder"]["emb_dim"], bias=True)
        )

        self.task_decoder = MLP(self.lm_dim, hidden_features=4 * configs["model"]["graph_encoder"]["emb_dim"], out_features=1)

        self.max_length = configs["model"]["Qwen3"]["max_length"]

        if self.configs[model][Qwen3][use_lora]:
            lora_config = LoraConfig(
                r=self.configs[model][Qwen3][lora_configs][rank],
                lora_alpha=self.configs[model][Qwen3][lora_configs][alpha],
                lora_dropout=self.configs[model][Qwen3][lora_configs][dropout],
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.lm_encoder = get_peft_model(self.lm_encoder, lora_config)

        else:
            # freeze the lm encoder, do not use lora
            self.freeze_lm_encoder()
        # TODO: full supervised learning for the lm encoder

    def freeze_lm_encoder(self):
        for param in self.lm_encoder.parameters():
            param.requires_grad = False

    def forward(self, batched_data):
        graph_emb = self.graph_encoder(batched_data)
        device = graph_emb.device

        # use the LM to generate the prompt from language of smiles
        smiles_explaination = batched_data.smiles
        # build prompt from languauge of smiles
        # the prompt should have been built with a larger LLM like GPT-5 or Qwen3 70B
        smile_prompt_emb, smile_prompt_mask = self.build_smile_prompt_emb(smiles_explaination)

        graph2action = self.graph_proj(graph_emb)
        graph2action = graph2action.unsqueeze(1)
        graph_mask = torch.ones_like(graph2action.shape[:-1])
        language2action = self.lm_proj(smile_prompt_emb)

        # concat
        hidden_state = torch.cat([graph2action, language2action], dim=1)
        hidden_state_mask = torch.cat([graph_mask, smile_prompt_mask], dim=1)

        # number of tasks
        B = hidden_state.shape[0]
        num_tasks = self.configs["dataset"]["num_tasks"]
        predict_states = torch.randn((B, num_tasks, self.lm_dim)).to(device)
        predict_states_mask = torch.ones_like(predict_states.shape[:-1])

        input_states = torch.cat([hidden_state, predict_states], dim=1)
        input_states_mask = torch.cat([hidden_state_mask, predict_states_mask], dim=1)
        
        output_hidden_states = self.lm_encoder(input_states, attention_mask=input_states_mask)

        lm_out = output_hidden_states.hidden_states[-1][:,-num_tasks:]

        task_out = self.task_decoder(lm_out)
        task_out = task_out.squeeze(-1)
        return task_out

    def build_smile_prompt_emb(self, smiles_explaination):
        # build prompt from languauge of smiles
        # the prompt should have been built with a larger LLM like GPT-5 or Qwen3 70B
        prompt_tokens = self.tokenizer(smiles_explaination, return_tensors="pt", padding="max_length", truncation=False, max_length=self.max_length)
        prompt_id = prompt_tokens.input_ids.to(device)
        prompt_mask = prompt_tokens.attention_mask.to(device)
        prompt_emb = self.lm_encoder.get_input_embeddings()(prompt_id)
        return prompt_emb
    

        