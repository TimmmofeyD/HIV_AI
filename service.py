import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import random
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from collections import UserList, defaultdict
from tqdm import tqdm


def seed_all():
    np.random.seed(42)
    random.seed(42)

seed_all()


df_train = pd.read_csv("./cleared.csv")
train_data = list(df_train.Smiles)

chars = set()
for string in train_data:
    chars.update(string)
all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
vocab = all_sys
c2i = {c: i for i, c in enumerate(all_sys)}
i2c = {i: c for i, c in enumerate(all_sys)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vector = torch.eye(len(c2i))


def char2id(char):
    if char not in c2i:
        return c2i['<unk>']
    else:
        return c2i[char]


def id2char(id):
    if id not in i2c:
        return i2c[32]
    else:
        return i2c[id]


def string2ids(string, add_bos=False, add_eos=False):
    ids = [char2id(c) for c in string]
    if add_bos:
        ids = [c2i['<bos>']] + ids
    if add_eos:
        ids = ids + [c2i['<eos>']]
    return ids


def ids2string(ids, rem_bos=True, rem_eos=True):
    if len(ids) == 0:
        return ''
    if rem_bos and ids[0] == c2i['<bos>']:
        ids = ids[1:]
    if rem_eos and ids[-1] == c2i['<eos>']:
        ids = ids[:-1]
    string = ''.join([id2char(id) for id in ids])
    return string


def string2tensor(string, device='model'):
    ids = string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(ids, dtype=torch.long, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tensor


tensor = [string2tensor(string, device=device) for string in train_data]
vector = torch.eye(len(c2i))

q_bidir = True
q_d_h = 256
q_n_layers = 1
q_dropout = 0.5
d_n_layers = 3
d_dropout = 0
d_z = 128
d_d_h = 512


class VAE(nn.Module):
    def __init__(self, vocab, vector):
        super().__init__()
        self.vocabulary = vocab
        self.vector = vector

        n_vocab, d_emb = len(vocab), vector.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, c2i['<pad>'])
        self.x_emb.weight.data.copy_(vector)

        # Encoder
        self.encoder_rnn = nn.GRU(d_emb, q_d_h, num_layers=q_n_layers, batch_first=True,
                                  dropout=q_dropout if q_n_layers > 1 else 0, bidirectional=q_bidir)
        q_d_last = q_d_h * (2 if q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, d_z)
        self.q_logvar = nn.Linear(q_d_last, d_z)

        # Decoder
        self.decoder_rnn = nn.GRU(d_emb + d_z, d_d_h, num_layers=d_n_layers, batch_first=True,
                                  dropout=d_dropout if d_n_layers > 1 else 0)
        self.decoder_latent = nn.Linear(d_z, d_d_h)
        self.decoder_fullyc = nn.Linear(d_d_h, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([self.encoder_rnn, self.q_mu, self.q_logvar])
        self.decoder = nn.ModuleList([self.decoder_rnn, self.decoder_latent, self.decoder_fullyc])
        self.vae = nn.ModuleList([self.x_emb, self.encoder, self.decoder])

    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device if device == 'model' else device)
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def forward(self, x):
        z, kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x, z)
        return kl_loss, recon_loss

    def forward_encoder(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def forward_decoder(self, x, z):
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=c2i['<pad>'])
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_latent(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, _ = self.decoder_rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fullyc(output)

        recon_loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)), x[:, 1:].contiguous().view(-1),
                                     ignore_index=c2i['<pad>'])
        return recon_loss

    def sample_z_prior(self, n_batch):
        return torch.randn(n_batch, self.q_mu.out_features, device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
                z = z.to(device)
                z_0 = z.unsqueeze(1)
                h = self.decoder_latent(z)
                h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
                w = torch.tensor(c2i['<bos>'], device=device).repeat(n_batch)
                x = torch.tensor([c2i['<pad>']], device=device).repeat(n_batch, max_len)
                x[:, 0] = c2i['<bos>']
                end_pads = torch.tensor([max_len], device=device).repeat(n_batch)
                eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=device)

                for i in range(1, max_len):
                    x_emb = self.x_emb(w).unsqueeze(1)
                    x_input = torch.cat([x_emb, z_0], dim=-1)

                    o, h = self.decoder_rnn(x_input, h)
                    y = self.decoder_fullyc(o.squeeze(1))

                    y = F.softmax(y / temp, dim=-1)
                    y = torch.clamp(y, 1e-8, 1.0)

                    w = torch.multinomial(y, 1)[:, 0]
                    x[~eos_mask, i] = w[~eos_mask]
                    i_eos_mask = ~eos_mask & (w == c2i['<eos>'])
                    end_pads[i_eos_mask] = i + 1
                    eos_mask = eos_mask | i_eos_mask

                new_x = []
                for i in range(x.size(0)):
                    new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]


@st.cache_resource
def load_model_gnn(model_path, _vector, vocab):
    model = VAE(vocab, _vector)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def generate_smiles(model, num_samples=5):
    with torch.no_grad():
        smiles_list = model.sample(num_samples)
    return smiles_list


def display_molecules(smiles_list):
    images = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(200, 200))
            images.append(img)
    return images


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Извлекаем дескрипторы
    features = {}
    for i, j in Descriptors.descList:
        features[i] = j(mol)

    return features


THRESHOLD = 0.28
COLUMNS = ['PEOE_VSA10',
           'NumHDonors',
           'BCUT2D_MRHI',
           'SMR_VSA3',
           'BCUT2D_MWHI',
           'MaxPartialCharge',
           'SMR_VSA5',
           'MaxEStateIndex',
           'MinAbsEStateIndex',
           'PEOE_VSA3',
           'BCUT2D_MRLOW',
           'BalabanJ',
           'FpDensityMorgan3',
           'BCUT2D_CHGLO',
           'BCUT2D_LOGPLOW',
           'SMR_VSA4',
           'fr_aryl_methyl',
           'SlogP_VSA4',
           'SlogP_VSA8',
           'PEOE_VSA4',
           'SMR_VSA9',
           'BCUT2D_MWLOW',
           'SlogP_VSA12',
           'VSA_EState6',
           'VSA_EState10',
           'fr_azide',
           'fr_SH'
           ]


@st.cache_resource
def load_model_cls(path):
    with open(path, 'rb') as f:
        model_cls = pickle.load(f)
        return model_cls


model_path_cls = "D:\\pythonProject\\model_rfc.pkl"
model_cls = load_model_cls(model_path_cls)

st.title("HIV GenAI")

st.subheader("Choose your mode")
mode = st.radio("", ["Classification", "Generation"])

if mode == "Classification":
    smiles_input = st.text_input("Enter your SMILES string:")

    if st.button("Classify"):
        if smiles_input:
            try:
                features = calculate_descriptors(smiles_input)

                if features is None:
                    st.error("Invalid SMILES string!")
                else:
                    feature_vector = [features.get(col, 0) for col in COLUMNS]
                    features_array = np.array(feature_vector).reshape(1, -1)

                    probs = model_cls.predict_proba(features_array)[:, 1]
                    prediction = "Active" if probs[0] >= THRESHOLD else "Inactive"

                    st.success(f"Result: {prediction} (Probability: {probs[0]:.2f})")
            except Exception as e:
                st.error(f"Error processing input: {e}")
        else:
            st.warning("Please enter a SMILES string.")

elif mode == "Generation":
    st.write("Generation")

    model_path_gnn = "D:\\pythonProject\\vae_model_epoch100.pt"
    model_gnn = load_model_gnn(model_path_gnn, vector, vocab)
    st.success("Model loaded!!")

    smiles = generate_smiles(model_gnn, 5)
    st.write("Generated SMILES:", smiles)
    images = display_molecules(smiles)
    for img in images:
        st.image(img)

# успех
