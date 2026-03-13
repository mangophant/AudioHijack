import os
import joblib
import fairseq
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torchaudio.functional import resample
from fairseq.data.dictionary import Dictionary
from torch.serialization import add_safe_globals

add_safe_globals([Dictionary])


class FeatureReader(object):
    def __init__(self, ckpt_path, device, layer, max_chunk=1600000, fp16=False, sampling_rate=16000):
        (model, cfg, task,) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate
        
    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    def get_feats(self, waveform):
        x = waveform
        if self.fp16:
            x = x.half()
        else:
            x = x.float()
        if self.task.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start: start + self.max_chunk]
            feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer + self.layer_shift,
            )
    
            feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)
    

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1), dist
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    
    def __init__(self, ckpt_dir, device='cpu', layer=11, max_chunk=1600000, fp16=False, sampling_rate=16000):
        super().__init__()
        ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3.pt")
        km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")
        self.feature_reader = FeatureReader(ckpt_path, device, layer, max_chunk, fp16, sampling_rate)
        self.apply_kmeans = ApplyKmeans(km_path)
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def __call__(self, data, merged=True):        
        feat = self.feature_reader.get_feats(data)
        cluster_ids, _ = self.apply_kmeans(feat)
        cluster_ids = cluster_ids.cpu().numpy().tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)
        merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        unmerged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"
        if merged:
            return merged_units
        else:
            return unmerged_units
