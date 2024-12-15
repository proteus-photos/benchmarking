import sys
sys.path.append("/data8/john/inr_adv_defense/liif")
import models as inr_models
from utils import make_coord
from tqdm import tqdm
import json
import torch
from torchvision import transforms


class INR(object):
    def __init__(self, model, device, data_name, pretrain_inr_path, height=224, width=224):
        self.data_name = data_name 
        self.model = model
        self.device = device
        self.inr_model = inr_models.make(torch.load(pretrain_inr_path)['model'], load_sd=True).to(self.device)
        self.height = height
        self.width = width

        self.coord = make_coord((self.height, self.width)).to(self.device)
        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.height
        self.cell[:, 1] *= 2 / self.width


    def batched_predict(self, inp, coord, cell, bsize):
        with torch.no_grad():
            self.inr_model.gen_feat(inp)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.inr_model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self,x):
        lst_img = []
        for img in x:
           img_tensor = img.unsqueeze(0)
           inr_output = self.batched_predict(((img_tensor - 0.5) / 0.5), self.coord.unsqueeze(0), self.cell.unsqueeze(0), bsize=30000)[0]
           inr_output = (inr_output * 0.5 + 0.5).clamp(0, 1).view(self.height, self.width, 3).permute(2, 0, 1)
           lst_img.append(inr_output)
        
        return x.new_tensor(torch.stack(lst_img))
 
 
