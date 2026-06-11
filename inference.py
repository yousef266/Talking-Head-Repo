
AUDIO_PATH        = r"C:\Users\user\Downloads\inference\audio\ai_5.wav"
INFERENCE_DIR     = r"C:\Users\user\Downloads\inference"          
SPECTRE_DIR       = r"C:\Users\user\Downloads\inference\spectre"  
OUTPUT_DIR        = r"C:\Users\user\Downloads\inference\output"   


INFER_TIMESTEPS   = 70
CHUNK_SIZE_50HZ   = 200
CFG_SCALE         = 1.7
SEED              = 0

EXPORT_FLAME_PT        = True   
EXPORT_OBJ_SEQUENCE    = True   
EXPORT_VERTICES        = False  



import os, sys, math
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import inspect
import json
import shutil

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


sys.path.insert(0, SPECTRE_DIR)

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


np.bool    = np.bool_
np.int     = np.int64
np.float   = np.float64
np.complex = np.complex128
np.object  = np.object_
np.unicode = np.str_
np.str     = np.str_

from config import cfg as spectre_cfg
from src.models.FLAME import FLAME


from transformers import HubertModel, Wav2Vec2FeatureExtractor

print("Loading mHuBERT-147 …")
mhubertmodel_path = "utter-project/mHuBERT-147"
processor     = Wav2Vec2FeatureExtractor.from_pretrained(mhubertmodel_path)
audio_encoder = HubertModel.from_pretrained(mhubertmodel_path, attn_implementation="eager").to(device)
audio_encoder.eval()
for p in audio_encoder.parameters():
    p.requires_grad = False
print("  mHuBERT loaded.")



def format_mhubert_features(hidden_states, gt_frame_num):
    if hidden_states.shape[1] % 2 != 0:
        hidden_states = hidden_states[:, :-1, :]
    target_audio_len  = gt_frame_num * 2
    current_audio_len = hidden_states.shape[1]
    if current_audio_len < target_audio_len:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=target_audio_len, align_corners=True, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)
    elif current_audio_len > target_audio_len:
        hidden_states = hidden_states[:, :target_audio_len, :]
    return hidden_states


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def apply_rotary_emb(x, freqs):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return x * freqs.cos() + torch.cat((-x2, x1), dim=-1) * freqs.sin()


class SDPATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.qkv   = nn.Linear(d_model, 3 * d_model)
        self.proj  = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.nhead = nhead

    def forward(self, x, rope_freqs, mask=None):
        B, T, C = x.shape
        H, D    = self.nhead, C // self.nhead
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = q.view(B,T,H,D), k.view(B,T,H,D), v.view(B,T,H,D)
        q = apply_rotary_emb(q, rope_freqs.unsqueeze(0).unsqueeze(2))
        k = apply_rotary_emb(k, rope_freqs.unsqueeze(0).unsqueeze(2))
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        x = x + self.proj(out.transpose(1,2).reshape(B,T,C))
        x = x + self.ffn(self.norm2(x))
        return x


class FLAMEKLEncoder(nn.Module):
    def __init__(self, flame_dim=53, hidden_dim=256, latent_dim=32, num_layers=4, num_heads=4, dropout=0.1, compression=2):
        super().__init__()
        self.compression  = compression
        self.input_proj   = nn.Linear(flame_dim, hidden_dim)
        self.rope         = RotaryEmbedding(hidden_dim // num_heads)
        self.layers       = nn.ModuleList([SDPATransformerLayer(hidden_dim, num_heads, hidden_dim*4, dropout) for _ in range(num_layers)])
        self.temporal_pool= nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        self.to_mean      = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar    = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        freqs = self.rope(h.shape[1], h.device)
        for layer in self.layers:
            h = layer(h, freqs, mask)
        if self.compression > 1:
            h = self.temporal_pool(h.transpose(1,2)).transpose(1,2)
        return self.to_mean(h), self.to_logvar(h)


class FLAMEKLDecoder(nn.Module):
    def __init__(self, flame_dim=53, hidden_dim=256, latent_dim=32, num_layers=4, num_heads=4, dropout=0.1, compression=2):
        super().__init__()
        self.compression       = compression
        self.input_proj        = nn.Linear(latent_dim, hidden_dim)
        self.temporal_upsample = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        self.rope              = RotaryEmbedding(hidden_dim // num_heads)
        self.layers            = nn.ModuleList([SDPATransformerLayer(hidden_dim, num_heads, hidden_dim*4, dropout) for _ in range(num_layers)])
        self.out_proj          = nn.Linear(hidden_dim, flame_dim)

    def forward(self, z, mask=None):
        h = self.input_proj(z)
        if self.compression > 1:
            h = self.temporal_upsample(h.transpose(1,2)).transpose(1,2)
        freqs = self.rope(h.shape[1], h.device)
        aligned_mask = mask[:, :h.shape[1]] if mask is not None else None
        for layer in self.layers:
            h = layer(h, freqs, aligned_mask)
        return self.out_proj(h)


class FLAMEKLVAE(nn.Module):
    def __init__(self, flame_dim=53, hidden_dim=256, latent_dim=32, num_layers=4, num_heads=4, dropout=0.1, compression=2):
        super().__init__()
        self.encoder     = FLAMEKLEncoder(flame_dim, hidden_dim, latent_dim, num_layers, num_heads, dropout, compression)
        self.decoder     = FLAMEKLDecoder(flame_dim, hidden_dim, latent_dim, num_layers, num_heads, dropout, compression)
        self.latent_dim  = latent_dim
        self.compression = compression

    def reparameterise(self, mean, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mean + std * torch.randn_like(std)
        return mean

    def encode(self, x, mask=None):
        mean, _ = self.encoder(x, mask)
        return mean

    def forward(self, x, mask=None):
        mean, logvar = self.encoder(x, mask)
        logvar = logvar.clamp(-30.0, 20.0)
        z = self.reparameterise(mean, logvar)
        recon = self.decoder(z, mask)
        recon = recon[:, :x.shape[1], :]
        return recon, mean, logvar


class LightweightAudioStrider(nn.Module):
    def __init__(self, output_dim=512, num_heads=4):
        super().__init__()
        self.stride           = 4
        self.layer_weights    = nn.Parameter(torch.ones(12))
        self.temporal_stride  = nn.Conv1d(768, output_dim, kernel_size=3, stride=self.stride, padding=1)
        self.activation       = nn.GELU()
        self.self_attention   = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

    def forward(self, precomputed_audio):
        norm_weights     = F.softmax(self.layer_weights, dim=0).view(1, 12, 1, 1)
        weighted_sum     = (precomputed_audio * norm_weights).sum(dim=1).transpose(1, 2)
        strided_features = self.activation(self.temporal_stride(weighted_sum)).transpose(1, 2)
        attn_out, _      = self.self_attention(strided_features, strided_features, strided_features)
        return strided_features + attn_out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim   = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


def build_time_mlp():
    return nn.Sequential(SinusoidalPositionEmbeddings(128), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256))


class GRUUniDenoiser(nn.Module):
    def __init__(self, out_dim, hidden_dim=512):
        super().__init__()
        self.time_mlp    = build_time_mlp()
        self.xt_proj     = nn.Sequential(nn.Linear(out_dim, 256), nn.GELU())
        self.gru         = nn.GRU(input_size=1024, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.4)
        self.attn        = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm        = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, 256), nn.GELU(), nn.Linear(256, out_dim))

    def forward(self, xt, audio, t, hidden_state=None):
        t_emb    = self.time_mlp(t).unsqueeze(1).expand(-1, xt.shape[1], -1)
        xt_feat  = self.xt_proj(xt)
        fused    = torch.cat([xt_feat, audio, t_emb], dim=-1)
        gru_out, new_hidden = self.gru(fused, hidden_state)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        out      = self.norm(gru_out + attn_out)
        return self.output_proj(out), new_hidden


class FlowMatchingDiffusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, gt_x0, strided_audio):
        B, T, D = gt_x0.shape
        t       = torch.rand(B, 1, 1, device=gt_x0.device)
        noise   = torch.randn_like(gt_x0)
        x_t     = (1 - t) * noise + t * gt_x0
        pred_vt, _ = self.model(x_t, strided_audio, t.squeeze())
        return x_t + (1 - t) * pred_vt


print("Loading VAE …")
vae_model = FLAMEKLVAE().to(device)
vae_ckpt  = torch.load(os.path.join(INFERENCE_DIR, "klvae_ep100.pth"), map_location=device)
vae_model.load_state_dict(vae_ckpt['klvae_state'])
vae_model.eval()
for p in vae_model.parameters():
    p.requires_grad = False

print("Loading FLAME …")
flame_model_path = os.path.join(INFERENCE_DIR, "generic_model.pkl")
spectre_cfg.defrost()
spectre_cfg.model.flame_model_path = flame_model_path
spectre_cfg.model.use_tex = False
spectre_cfg.freeze()

flame_model = FLAME(spectre_cfg.model).to(device)

def fix_module_tensors(module):
    for attr_name, attr_value in list(vars(module).items()):
        if isinstance(attr_value, torch.Tensor) and not isinstance(attr_value, nn.Parameter):
            delattr(module, attr_name)
            module.register_buffer(attr_name, attr_value)
    for child in module.children():
        fix_module_tensors(child)

fix_module_tensors(flame_model)
flame_model.eval()

print("Loading diffusion checkpoint …")
target_dim    = 32
audio_strider = LightweightAudioStrider(output_dim=512, num_heads=4).to(device)
denoiser_core = GRUUniDenoiser(out_dim=target_dim).to(device)
diffusion_model = FlowMatchingDiffusion(denoiser_core).to(device)

ckpt_path = os.path.join(INFERENCE_DIR, "checkpoint_ep75.pth")
checkpoint = torch.load(ckpt_path, map_location=device)
audio_strider.load_state_dict(checkpoint['audio_strider_state'])
denoiser_core.load_state_dict(checkpoint['denoiser_state'])
print("  All checkpoints loaded.")



print(f"\nLoading audio: {AUDIO_PATH}")
speech_array, _ = librosa.load(AUDIO_PATH, sr=16000)
print(f"  Duration: {len(speech_array)/16000:.2f}s")


import noisereduce as nr

NOISE_SAMPLE_DURATION = 0.5   
noise_sample_end = int(NOISE_SAMPLE_DURATION * 16000)
noise_sample = speech_array[:noise_sample_end]

speech_array = nr.reduce_noise(
    y=speech_array,
    sr=16000,
    y_noise=noise_sample,   
    stationary=False,      
    prop_decrease=0.9,      
    n_fft=512,
    n_std_thresh_stationary=1.5,
)
print(f"  Denoised: {len(speech_array)/16000:.2f}s  (noisereduce applied)")


inputs        = processor([speech_array], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
input_values  = inputs.input_values.to(device)
attention_mask= inputs.attention_mask.to(device)

print("Running mHuBERT forward pass …")
with torch.no_grad():
    outputs = audio_encoder(input_values, attention_mask=attention_mask, output_hidden_states=True)

stacked_hidden = torch.stack(outputs.hidden_states[1:], dim=1)   

base_model = audio_encoder.module if hasattr(audio_encoder, 'module') else audio_encoder
valid_len  = base_model._get_feat_extract_output_lengths(attention_mask.sum(-1))[0].item()
features   = stacked_hidden[0, :, :valid_len, :]                  
print(f"  Feature shape: {tuple(features.shape)}")

long_audio_tensor = features.unsqueeze(0).to(device) 



@torch.no_grad()
def streaming_inference(diffusion_model, audio_strider, long_audio_tensor,
                        timesteps=50, chunk_size_50hz=150, cfg_scale=1.7):
    diffusion_model.eval()
    audio_strider.eval()

    total_time  = long_audio_tensor.shape[2]
    hidden_state= None
    all_outputs = []
    t_steps     = torch.linspace(0.0, 1.0, timesteps + 1, device=device)

    for start_idx in range(0, total_time, chunk_size_50hz):
        end_idx     = min(start_idx + chunk_size_50hz, total_time)
        audio_chunk = long_audio_tensor[:, :, start_idx:end_idx]

        strided_audio  = audio_strider(audio_chunk)
        uncond_audio   = torch.zeros_like(strided_audio)

        xt = torch.randn(1, strided_audio.shape[1], target_dim, device=device)

        for i in range(timesteps):
            t      = t_steps[i].unsqueeze(0).expand(xt.shape[0])
            t_next = t_steps[i+1].unsqueeze(0).expand(xt.shape[0])

            v_t_cond,   current_hidden = diffusion_model.model(xt, strided_audio, t, hidden_state)
            v_t_uncond, _              = diffusion_model.model(xt, uncond_audio,  t, hidden_state)

            v_t = v_t_uncond + cfg_scale * (v_t_cond - v_t_uncond)
            dt  = t_next - t
            xt  = xt + v_t * dt.view(-1, 1, 1)

        hidden_state = current_hidden.detach() if current_hidden is not None else None
        all_outputs.append(xt)
        print(f"  Chunk {start_idx}–{end_idx} done.")

    full_output = torch.cat(all_outputs, dim=1)
    full_output = vae_model.decoder(full_output)
    return full_output


print("\nStarting streaming inference …")
predicted_sequence = streaming_inference(
    diffusion_model=diffusion_model,
    audio_strider=audio_strider,
    long_audio_tensor=long_audio_tensor,
    timesteps=INFER_TIMESTEPS,
    chunk_size_50hz=CHUNK_SIZE_50HZ,
    cfg_scale=CFG_SCALE,
)
print(f"  Predicted shape: {tuple(predicted_sequence.shape)}")

predicted_sequence[:, :, 51] *= 0.1
predicted_sequence[:, :, 52] *= 0.1

if EXPORT_FLAME_PT:
    flame_pt_path = os.path.join(OUTPUT_DIR, "predicted_flame.pt")
    torch.save(predicted_sequence.cpu(), flame_pt_path)
    print(f"\n✓ FLAME params saved → {flame_pt_path}")



if EXPORT_VERTICES:
    pred    = predicted_sequence.squeeze(0) if predicted_sequence.dim() == 3 else predicted_sequence
    T       = pred.shape[0]
    base_shape = torch.zeros(1, 100).to(device)
    all_verts  = []

    print("\nExporting vertices …")
    with torch.no_grad():
        for i in range(0, T, 500):
            chunk       = pred[i:i+500].to(device)
            fixed_shape = base_shape.expand(chunk.shape[0], -1)
            pose        = torch.zeros((chunk.shape[0], 6), device=device, dtype=chunk.dtype)
            pose[:, 3:6]= chunk[:, 50:53]
            flame_out   = flame_model(shape_params=fixed_shape, expression_params=chunk[:, :50], pose_params=pose)
            all_verts.append(flame_out[0].cpu())

    vertices = torch.cat(all_verts, dim=0)
    verts_path = os.path.join(OUTPUT_DIR, "predicted_vertices.pt")
    torch.save(vertices, verts_path)
    print(f"✓ Vertices saved → {verts_path}")



if EXPORT_OBJ_SEQUENCE:
    flame_tensor = predicted_sequence
    if flame_tensor.dim() == 3:
        flame_tensor = flame_tensor.squeeze(0)

    obj_dir    = os.path.join(OUTPUT_DIR, "flame_objs")
    if os.path.exists(obj_dir):
        shutil.rmtree(obj_dir)
    os.makedirs(obj_dir, exist_ok=True)
    
    faces      = flame_model.faces_tensor.cpu().numpy()
    base_shape = torch.zeros(1, 100).to(device)
    T_total    = flame_tensor.shape[0]

    print(f"\nExporting {T_total} OBJ frames …")
    with torch.no_grad():
        for frame_idx in range(T_total):
            frame = flame_tensor[frame_idx:frame_idx+1].to(device)
            pose  = torch.zeros((1, 6), device=device, dtype=frame.dtype)
            pose[:, 3:6] = frame[:, 50:53]

            verts, _, _ = flame_model(
                shape_params=base_shape,
                expression_params=frame[:, :50],
                pose_params=pose
            )
            verts = verts[0].cpu().numpy()

            obj_path = os.path.join(obj_dir, f"frame_{frame_idx:05d}.obj")
            with open(obj_path, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            if (frame_idx + 1) % 100 == 0 or frame_idx == T_total - 1:
                print(f"  {frame_idx+1}/{T_total} frames written")


flame_tensor_json = predicted_sequence.squeeze(0) if predicted_sequence.dim() == 3 else predicted_sequence
exp_data = flame_tensor_json[:, :50].detach().cpu().numpy().astype(float).tolist()
jaw_data = flame_tensor_json[:, 50:53].detach().cpu().numpy().astype(float).tolist()
faces_data = flame_model.faces_tensor.cpu().numpy().astype(int).tolist()

payload = {
    "expressions": exp_data,
    "jaw": jaw_data,
    "faces": faces_data
}

json_path = os.path.join(OUTPUT_DIR, "test_sequence.json")
with open(json_path, "w") as f:
    json.dump(payload, f)
print(f"\n✓ JSON saved → {json_path}")

print("\nAll done!")