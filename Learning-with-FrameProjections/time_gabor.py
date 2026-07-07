import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/home/gella.saikrishna/code/Learning-with-FrameProjections"
LOCAL_DATA_DIR = "/home/gella.saikrishna/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette"
SAVE_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(SAVE_DIR, "sleep_combined_optimized.pt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCH_SEC = 30
TARGET_SFREQ = 100
TIME_STEPS = 3000
LATENT_DIM = 128
NUM_CLASSES = 5
PRETRAIN_EPOCHS = 40
EVAL_EPOCHS = 30
BATCH_SIZE = 64

STAGE_MAPPING = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4
}

def prepare_data():
    if not os.path.exists(OUTPUT_FILE):
        psg_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*PSG.edf")))
        hypno_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*Hypnogram.edf")))
        raw_data_files = list(zip(psg_files, hypno_files))

        all_epochs_list = []
        all_labels_list = []

        for psg_file, hypno_file in raw_data_files:
            raw = mne.io.read_raw_edf(psg_file, preload=True)
            eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
            if len(eeg_channels) > 0:
                raw.pick_channels([eeg_channels[0]])
            else:
                continue
                
            raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin')
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, npad="auto")
                
            annotations = mne.read_annotations(hypno_file)
            raw.set_annotations(annotations, emit_warning=False)
            
            events, event_id = mne.events_from_annotations(raw, event_id=STAGE_MAPPING, chunk_duration=float(EPOCH_SEC))
            tmax = 30.0 - 1.0 / TARGET_SFREQ 
            epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None, preload=True, reject=None)
            
            data = epochs.get_data() 
            labels = epochs.events[:, -1]
            data = np.squeeze(data, axis=1)
            
            all_epochs_list.append(data)
            all_labels_list.append(labels)

        X = np.concatenate(all_epochs_list, axis=0)
        y = np.concatenate(all_labels_list, axis=0)

        X = (X - np.mean(X)) / (np.std(X) + 1e-6)

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)

        def get_gabor(data_tensor):
            window = torch.hann_window(128)
            stft = torch.stft(data_tensor, n_fft=128, hop_length=64, window=window, return_complex=True)
            wavelet = torch.abs(stft)[:, :64, :]
            return F.pad(wavelet, (0, 1))

        processed_dataset_object = {
            "train": {"samples": torch.from_numpy(X_train).float(), "labels": torch.from_numpy(y_train).long(), "gabor": get_gabor(torch.from_numpy(X_train).float())},
            "val": {"samples": torch.from_numpy(X_val).float(), "labels": torch.from_numpy(y_val).long(), "gabor": get_gabor(torch.from_numpy(X_val).float())},
            "test": {"samples": torch.from_numpy(X_test).float(), "labels": torch.from_numpy(y_test).long(), "gabor": get_gabor(torch.from_numpy(X_test).float())}
        }
        torch.save(processed_dataset_object, OUTPUT_FILE)

class SleepEDF_Pretrain_Dataset(Dataset):
    def __init__(self, pt_file_path=OUTPUT_FILE, split="train"):
        data_obj = torch.load(pt_file_path, map_location="cpu")[split]
        self.data = data_obj["samples"]
        self.gabor = data_obj["gabor"]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.gabor[idx].unsqueeze(0)

class SleepEDF_Evaluation_Dataset(Dataset):
    def __init__(self, pt_file_path=OUTPUT_FILE, split="train"):
        data_obj = torch.load(pt_file_path, map_location="cpu")[split]
        self.samples = data_obj["samples"].unsqueeze(1)
        self.labels = data_obj["labels"].long()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=kernel_size, stride=stride, groups=groups)
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.conv(x)

class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.max_pool = torch.nn.MaxPool1d(kernel_size=kernel_size)
        self.kernel_size = kernel_size
    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + 1 - 1) // 1
        p = max(0, (out_dim - 1) * 1 + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.max_pool(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.downsample = downsample
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, out_channels, kernel_size, stride, groups)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size, 1, groups)
        self.max_pool = MyMaxPool1dPadSame(stride)
    def forward(self, x):
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn: out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do: out = self.do1(out)
        out = self.conv1(out)
        if self.use_bn: out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do: out = self.do2(out)
        out = self.conv2(out)
        if self.downsample: identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        out += identity
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, backbone=False, output_dim=200):
        super(ResNet1D, self).__init__()
        self.n_block = n_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.backbone = backbone
        self.first_block_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size, 1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.basicblock_list = nn.ModuleList()
        for i_block in range(n_block):
            is_first = (i_block == 0)
            downsample = (i_block % downsample_gap == 1)
            in_ch = base_filters if is_first else int(base_filters*2**((i_block-1)//increasefilter_gap))
            out_ch = in_ch if (i_block % increasefilter_gap != 0 or i_block == 0) else in_ch * 2
            self.basicblock_list.append(BasicBlock(in_ch, out_ch, kernel_size, stride, groups, downsample, use_bn, use_do, is_first))
        self.final_bn = nn.BatchNorm1d(out_ch)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_ch, n_classes)
        self.dense2 = nn.Linear(out_ch, output_dim)
    def forward(self, x):
        out = self.first_block_relu(self.first_block_bn(self.first_block_conv(x)))
        for block in self.basicblock_list: out = block(out)
        out = self.final_relu(self.final_bn(out)).mean(-1)
        if self.backbone: return None, self.dense2(out)
        return self.dense(out), out

class UNET_2D_simp(nn.Module):
    def __init__(self, input_channels, output_channels, layer_n, spect_freq, spect_time, kernel_size):
        super(UNET_2D_simp, self).__init__()
        self.AvgPool2D1 = nn.AvgPool2d((2, 2))
        self.AvgPool2D2 = nn.AvgPool2d((4, 4))
        self.layer1 = nn.Sequential(nn.Conv2d(input_channels, layer_n, kernel_size, padding=kernel_size//2), nn.BatchNorm2d(layer_n), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(layer_n, layer_n*2, kernel_size, stride=2, padding=kernel_size//2), nn.BatchNorm2d(layer_n*2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(layer_n*2+input_channels, layer_n*3, kernel_size, stride=2, padding=kernel_size//2), nn.BatchNorm2d(layer_n*3), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(layer_n*3+input_channels, layer_n*4, kernel_size, stride=2, padding=kernel_size//2), nn.BatchNorm2d(layer_n*4), nn.ReLU())
        self.fc = nn.Linear(spect_time // 8, 1)
        self.fc2 = nn.Linear(spect_freq // 8, 1)
    def forward(self, x):
        p1, p2 = self.AvgPool2D1(x), self.AvgPool2D2(x)
        o1 = self.layer1(x)
        o2 = self.layer2(o1)
        o3 = self.layer3(torch.cat([o2, p1], dim=1))
        o4 = self.layer4(torch.cat([o3, p2], dim=1))
        return None, self.fc2(self.fc(o4).squeeze(-1)).squeeze(-1)

class IsoAlign(nn.Module):
    def __init__(self, backbone, spect_encoder, DEVICE):
        super(IsoAlign, self).__init__()
        self.encoder, self.spect_encoder, self.DEVICE = backbone, spect_encoder, DEVICE
    def nt_xent_loss(self, z1, z2):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / 0.2
        labels = torch.arange(z1.shape[0]).to(self.DEVICE)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
    def forward(self, batch_t, batch_w):
        _, feat_t = self.encoder(batch_t)
        _, feat_w = self.spect_encoder(batch_w)
        return self.nt_xent_loss(feat_t, feat_w)

if __name__ == "__main__":
    prepare_data()
    pretrain_loader = DataLoader(SleepEDF_Pretrain_Dataset(split="train"), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    time_encoder = ResNet1D(1, 32, 5, 1, 1, 3, LATENT_DIM, backbone=True, output_dim=LATENT_DIM).to(DEVICE)
    spect_encoder = UNET_2D_simp(1, LATENT_DIM, 32, 64, 48, 3).to(DEVICE)
    model = IsoAlign(time_encoder, spect_encoder, DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        total_loss = 0
        for batch_t, batch_w in pretrain_loader:
            optimizer.zero_grad()
            loss = model(batch_t.to(DEVICE), batch_w.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(pretrain_loader)
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Loss: {avg_loss:.4f}")
        torch.save(model.encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "best_time_encoder.pth"))

    eval_encoder = ResNet1D(1, 32, 5, 1, 1, 3, LATENT_DIM, backbone=True, output_dim=LATENT_DIM).to(DEVICE)
    eval_encoder.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_time_encoder.pth"), map_location=DEVICE))
    classifier = nn.Linear(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-2)
    val_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="val"), batch_size=128, shuffle=False, num_workers=2)

    for epoch in range(EVAL_EPOCHS):
        classifier.train()
        for signals, labels in DataLoader(SleepEDF_Evaluation_Dataset(split="train"), batch_size=128, shuffle=True, num_workers=2):
            optimizer.zero_grad()
            with torch.no_grad(): _, feat = eval_encoder(signals.to(DEVICE))
            loss = nn.CrossEntropyLoss()(classifier(feat), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f"Eval Epoch {epoch+1}/{EVAL_EPOCHS}")