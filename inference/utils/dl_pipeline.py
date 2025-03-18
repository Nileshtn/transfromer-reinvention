from utils import TransformerMini
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

class TextGeneator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TransformerMini(context_l=512, n_vocal=256, embedding_dim=128, attention_heads=16).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    def generate(self, token_len=100):
        token = torch.tensor([[95, 113]]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i in range(token_len):
                out = self.model(token)
                sm_out = F.softmax(out[:,-1,:], dim=-1)
                predict = torch.multinomial(sm_out, num_samples=1)
                token = torch.cat((token, predict), dim=1)[:, -50:]
                print(self.tokenizer.decode(predict[0].cpu().detach().tolist()), end='')