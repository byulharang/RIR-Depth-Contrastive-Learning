import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision import models
import torchaudio

from tqdm import tqdm
import numpy as np
from PIL import Image

from HorizonNet.model import HorizonNet
from HorizonNet.misc import utils
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


os.environ['CUDA_VISIBLE_DEVICES'] = '3'   


class PairedDataset(Dataset):
    def __init__(self, dataset, transform=None, n_fft=320, hop_length=160):
        self.file_list = []
        self.root_dir = dataset   
        self.transform = transform if transform else transforms.ToTensor()  
        self.Amplitude_to_dB = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)
        self.n_fft = n_fft
        self.hop_length = hop_length

            # extract room directories
        self.room_dirs = os.listdir(self.root_dir)
        for room_name in self.room_dirs:
            image_files = []
            audio_files = []
            folder_path = os.path.join(self.root_dir, room_name)
            
            image_file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
            image_files.extend(image_file_paths)
 
            audio_file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')])
            audio_files.extend(audio_file_paths)                                 
                # check RIR & Img have same # of files
            if len(image_files) == len(audio_files):
                for img_file, audio_file in zip(image_files, audio_files):
                    self.file_list.append((img_file, audio_file))   # 1 Idx = 1 Pair
            else:
                raise ValueError(f"Number of image and audio files do not match in {room_name}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, audio_path = self.file_list[idx]

            # Image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # Tensor
        image = image.unsqueeze(0)  # [1 x 3 x 512 x 1024]
        
            # Audio 1D
        B_format, sr = torchaudio.load(audio_path)
        W = B_format[0, :].unsqueeze(0)
        Y = B_format[1, :].unsqueeze(0)
        Z = B_format[2, :].unsqueeze(0)
        X = B_format[3, :].unsqueeze(0)
        
        RIR = [W,Y,Z,X]    
        STFTs = []
        
            # STFT 
        window = torch.hann_window(self.n_fft)
        for rir in RIR:
    
            stft_result = torch.stft(rir, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                window=window,
                win_length=self.n_fft, 
                return_complex=True
            )
            STFTs.append(stft_result)   # [4, :, :]
        
        #   #  raw data
        raw_real = [stft.real for stft in STFTs]  
        raw_imag = [stft.imag for stft in STFTs]
        
        raw_mag = [torch.sqrt(real**2 + imag**2) for real, imag in zip(raw_real, raw_imag)]
        raw_mag_dB = [self.Amplitude_to_dB(mag) for mag in raw_mag]
        raw_mag_dB = [mag - mag.min() for mag in raw_mag_dB]
        raw_phase = [torch.atan2(imag, real) for imag, real in zip(raw_imag, raw_real)]     

            # Get when Record start
        stft_indices = []
        for real, imag in zip(raw_real, raw_imag):  
            non_zero_mask = (torch.abs(real) + torch.abs(imag)) != 0
            non_zero_cols = torch.any(non_zero_mask, dim=1)
            non_zero_cols = non_zero_cols [0]
            true_indices = torch.nonzero(non_zero_cols) 

            if true_indices.numel() > 0:
                stft_idx = true_indices[0].item()  
                stft_indices.append(stft_idx)
            else:
                stft_indices.append(None)

            # padding 100 (max RT60 of Dataset)
        sample_size = raw_real[0].shape[-1]
        padding_size = max(0, stft_idx + 100 - sample_size)
        
        real_padded = [F.pad(real, (0, padding_size)) for real in raw_real]
        imag_padded = [F.pad(imag, (0, padding_size)) for imag in raw_imag]
        mag_padded = [F.pad(mag, (0, padding_size)) for mag in raw_mag_dB]
        phase_padded = [F.pad(phase, (0, padding_size)) for phase in raw_phase]
        
            # We use this Slicing   # [4 x 161 x 100]
        real = [r[:, :, stft_idx:stft_idx+100].squeeze(0) for r in real_padded]
        imag = [i[:, :, stft_idx:stft_idx+100].squeeze(0) for i in imag_padded]
        mag = [m[:, :, stft_idx:stft_idx+100].squeeze(0) for m in mag_padded]
        phase = [p[:, :, stft_idx:stft_idx+100].squeeze(0) for p in phase_padded]

        real = torch.stack(real, dim=0)
        imag = torch.stack(imag, dim=0)
        mag = torch.stack(mag, dim=0)
        phase = torch.stack(phase, dim=0)
        # np.savetxt('/home/byulharang/test_spec1.csv', phase[0], delimiter=',')
        
        return image, mag, phase
        # return image, real, imag


'''         Model, Loss Define            '''


class ImageEmbeddingNet(nn.Module):
    def __init__(self):
        super(ImageEmbeddingNet, self).__init__()
        self.model = HorizonNet('resnet50', use_rnn=True)
        self.model.load_state_dict(torch.load('/home/byulharang/CRIP/resnet50_rnn__mp3d.pth'), strict=False)

    def forward(self, batch_image):
        bon, cor = self.model(batch_image)   # BS x [2, 1] x 1024
            # fusion
        fusion_bon = bon.view(bon.size(0), 1, -1) # [BS x 1 x 2048]
        
        fusion_bon = F.avg_pool1d(fusion_bon, kernel_size=2).squeeze(1)    # [BS x 1024]
        image_embed = F.avg_pool1d(fusion_bon, kernel_size=2).squeeze(1)    # [BS x 512]
        image_embed = F.normalize(image_embed, p=2, dim=1, eps=1e-6)
        return image_embed


class AudioEmbeddingNet(nn.Module):
    def __init__(self):
        super(AudioEmbeddingNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False) # 8 channels
        self.model.fc = nn.Sequential(
            # nn.Dropout(p=0.9),
            nn.Linear(self.model.fc.in_features, 512)
        )   
    
    def forward(self, batch_audio):
                
        audio_embed = self.model(batch_audio)
        audio_embed = F.normalize(audio_embed, p=2, dim=1, eps=1e-6)    # [BS x 512]
        return audio_embed


class IntegModel(nn.Module):
    
    def __init__(self):
        super(IntegModel, self).__init__()
        self.model_image = ImageEmbeddingNet()
        self.model_audio = AudioEmbeddingNet()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.img_linear = nn.Linear(512, 512)
        self.rir_linear = nn.Linear(512, 512)
    
    def loss_criterion(self, audio_embeds, image_embeds, logit_scale):
        
            # logit
        audio_logit = audio_embeds @ image_embeds.T
        image_logit = image_embeds @ audio_embeds.T
    
        audio_sim = audio_embeds @ audio_embeds.T
        image_sim = image_embeds @ image_embeds.T
        
            # Weight
        W_pos = 1
        W_neg = 1
        
        total_loss = 0
        labels = torch.arange(audio_embeds.size(0))  

        print('audio sim: ', audio_sim[0,:],'\n\n',
              'image sim: ', image_sim[0,:], '\n\n',
            #   'audio mat: ', F.softmax(audio_logit, dim=1), '\n\n'
              )
        
            # Audio logit
        for i in range(len(labels)):
            row = F.softmax(audio_logit[i, :], dim=0)
            pos = row[i]  
            # print(pos)
            loss = ( -torch.log(pos)) 
            total_loss += loss

            # Image logit
        for i in range(len(labels)):
            row = F.softmax(image_logit[i, :], dim=0)
            pos = row[i]  

            loss = ( -torch.log(pos)) 
            total_loss += loss
            
            # return
        return total_loss / (2*len(labels))

    
    def validation(self, Target_list, pool_list):
        
        logit = F.softmax(Target_list @ pool_list.T, dim=1)
        correct = 0
        
        for i in range(len(logit)):
            eval_mat = logit[i]
            eval_Idx = torch.argmax(eval_mat)
            if eval_Idx == i:
                correct += 1
            
            # print(len(eval_mat), eval_mat)
            # print(f'eval: {eval_Idx}, Target: {i}')            
                
        return torch.tensor(correct/len(logit), device=logit.device)    # correct

    def forward(self, batch_image, batch_1, batch_2):  # 1: real 2: imaginary

            # batch preproc
        batch_image = batch_image.squeeze(1)
        # batch_1 = batch_1.squeeze(1)  
        # batch_2 = batch_2.squeeze(1)
        batch_audio = torch.cat((batch_1, batch_2), dim=1)

            # Embedding
        image_embed = self.model_image(batch_image)     # input: [BS x 3 x 512 x 1024]   output: [BS x 512]
        audio_embed = self.model_audio(batch_audio)     # input: [BS x 8 x 161 x 100]    output: [BS x 512]
            # Projection
        image_embed = self.img_linear(image_embed)
        audio_embed = self.rir_linear(audio_embed)
            # For cosine sim
        image_embed = F.normalize(image_embed, p=2, dim=1, eps=1e-8)
        audio_embed = F.normalize(audio_embed, p=2, dim=1, eps=1e-8)
    
            # logit
        logit_scale = self.logit_scale.exp()    
        
            # Debugging
        
        # print(f'embeds: {audio_embeds.shape}, | {image_embeds.shape}')
        # print(f"audio_embeds shape: {audio_embeds.shape}, first few values: {audio_embeds[:, :100:20]}")
        # print(f"image_embeds shape: {image_embeds.shape}, first few values: {image_embeds[:, :100:20]}")
        # print(f'logit: {audio_embeds @ image_embeds.T}')
        
        loss = self.loss_criterion(audio_embed, image_embed, logit_scale)
        Audio_Accuracy = self.validation(audio_embed, image_embed)
        Image_Accuracy = self.validation(image_embed, audio_embed)

        batch_img_acc = torch.tensor(Image_Accuracy)
        batch_rir_acc = torch.tensor(Audio_Accuracy)
        
        return loss, batch_img_acc, batch_rir_acc





'''         Training, Validation           '''


#   # Train per epoch
def train_one_epoch(epoch, model, train_loader, optimizer):
    
    model.train()
    
    max_norm=1.0
    running_loss = 0.0
    total_img_acc = 0.0
    total_rir_acc = 0.0
    num_batch = 0

    for batch_idx, (batch_Img, batch_1, batch_2) in enumerate(tqdm(train_loader, desc=f"{(epoch+1)}/{epochs}")):
        optimizer.zero_grad()  # Init
        batch_Img = batch_Img.to(device)
        batch_1 = batch_1.to(device)
        batch_2 = batch_2.to(device)
        
        loss, batch_img_acc, batch_rir_acc = model(batch_Img, batch_1, batch_2) # 1 batch -> 1 GPU
        loss = loss.mean()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        running_loss += loss.item()

        batch_rir_acc = batch_rir_acc.mean() 
        batch_img_acc = batch_img_acc.mean() 
                    
        total_rir_acc += batch_rir_acc  # RIR에 맞는 Image 맞추기
        total_img_acc += batch_img_acc  # Image에 맞는 RIR 맞추기
        num_batch = batch_idx + 1
        
    ave_img_acc = total_img_acc / num_batch
    ave_rir_acc = total_rir_acc / num_batch
    Single_epoch_loss = running_loss / num_batch
    return Single_epoch_loss, ave_img_acc, ave_rir_acc

#   # Validate per epoch
def validate_one_epoch(epoch, model, val_loader):
    
    model.eval()
    
    running_loss = 0.0
    total_img_acc = 0.0
    total_rir_acc = 0.0
    num_batch = 0
    
    with torch.no_grad():
        for batch_idx, (batch_Img, batch_1, batch_2) in enumerate(tqdm(val_loader, desc=f"{(epoch+1)}/{epochs}")):

            batch_Img = batch_Img.to(device)
            batch_1 = batch_1.to(device)
            batch_2 = batch_2.to(device)
                            
            loss, batch_img_acc, batch_rir_acc = model(batch_Img, batch_1, batch_2) # 1 batch -> 1 GPU
            
            loss = loss.mean() # Average GPU's loss 
            if loss is None:
                raise ValueError("Loss returned None.")
            running_loss += loss.item()
            
            batch_rir_acc = batch_rir_acc.mean() 
            batch_img_acc = batch_img_acc.mean() 
                        
            total_rir_acc += batch_rir_acc  # RIR에 맞는 Image 맞추기
            total_img_acc += batch_img_acc  # Image에 맞는 RIR 맞추기
            num_batch = batch_idx + 1
            
    ave_img_acc = total_img_acc / num_batch
    ave_rir_acc = total_rir_acc / num_batch
    Single_epoch_loss = running_loss / num_batch

    return Single_epoch_loss, ave_img_acc, ave_rir_acc


### MAIN ###
'''         Hyper Parameter setting       '''


batch_size = 4
start_epoch = 0
epochs = 200
learning_rate = 0.00003


'''         Dataset Setting         '''


#   # Data Load
Train_DT = '/home/byulharang/dataset/mp3d'
Val_DT = '/home/byulharang/dataset/Val'

#   # Log                                                Change here
log_path = '/home/byulharang/Log_N_Loss_history/Fall/october/log_241022/test_5'
ckpt_path = os.path.join(log_path, 'Checkpoint/')
loss_path = os.path.join(log_path, 'LOSS/')
eval_path = os.path.join(log_path, 'Eval/')
os.makedirs(log_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)
os.makedirs(loss_path, exist_ok=True)
os.makedirs(eval_path, exist_ok=True)

    # Log file path
train_loss_path = os.path.join(loss_path, 'train_loss.csv')
val_loss_path = os.path.join(loss_path, 'val_loss.csv')

train_eval_path = os.path.join(eval_path, 'train_eval.csv')
val_eval_path = os.path.join(eval_path, 'val_eval.csv')



'''     Setting (CUDA, Optimizer, Data Loader)        '''
        
    
    # Parallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Count of using GPUs:', torch.cuda.device_count())

model = IntegModel()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

    # Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data Loader
gpus =  torch.cuda.device_count()

    # Train DT는 매 에폭마다 Shuffle, Val DT는 최초 셔플 후 고정 
train_dataloader = DataLoader(PairedDataset(Train_DT), batch_size=batch_size * gpus, 
                                shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(PairedDataset(Val_DT), batch_size=batch_size * gpus, 
                            shuffle=False,drop_last= True, num_workers=8, pin_memory=True)

'''         Training Process            '''
#   # loss_log list
train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []
best_loss = float('inf')

    # checkpoint load   Change here                 
ckpt_load = f'{ckpt_path}best.tar'
ckpt_save = f'{ckpt_path}best.tar'

#   # checkpoint load
try:
    checkpoint = torch.load(ckpt_load)
    
    if isinstance(model, torch.nn.DataParallel):
        original_state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in original_state_dict.items():
            # 키 이름을 교체
            new_key = k.replace('reduce_height_module.ghc_lst', 'reduce_height_ghc_lst')
            new_state_dict[new_key] = v
        model.module.load_state_dict(new_state_dict)    
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}.")

except (FileNotFoundError, KeyError, Exception) as e:
    print(f"Checkpoint not found Starting from 0")
    epoch = 0  

#   # Freeze model
for param_name, param in model.named_parameters():
    if 'model_image.model' in param_name: 
        param.requires_grad = False


#   # Training
for epoch in range(epochs):
    train_loss, train_img_acc, train_rir_acc = train_one_epoch(epoch, model, train_dataloader, optimizer)
    val_loss, val_img_acc, val_rir_acc = validate_one_epoch(epoch, model, val_dataloader)
    
        # Average accuracy
    train_ave_acc = (train_img_acc + train_rir_acc) / 2
    val_ave_acc = (val_img_acc + val_rir_acc) / 2

        # Save Loss & Checkpoint
    train_loss_log.append(train_loss)
    val_loss_log.append(val_loss)

    train_acc_log.append([train_img_acc.cpu().numpy(), 
                        train_rir_acc.cpu().numpy(),
                        train_ave_acc.cpu().numpy()])
    val_acc_log.append([val_img_acc.cpu().numpy(),
                        val_rir_acc.cpu().numpy(),
                        val_ave_acc.cpu().numpy()])

    if val_loss < best_loss:
        best_loss = val_loss
        
        if isinstance(model, torch.nn.DataParallel):
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, ckpt_save)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, ckpt_save)
            
        print(f"Saving checkpoint for epoch {epoch+1}")
        
    print(f"Saving logs to epoch {epoch+1}")

    np.savetxt(train_loss_path, np.column_stack((np.arange(1, len(train_loss_log) + 1), train_loss_log)),
            delimiter=',', header='Epoch,Train_Loss', comments='', fmt='%d,%.5f')
    np.savetxt(val_loss_path, np.column_stack((np.arange(1, len(val_loss_log) + 1), val_loss_log)),
            delimiter=',', header='Epoch,Val_Loss', comments='', fmt='%d,%.5f')
    np.savetxt(train_eval_path, np.column_stack((np.arange(1, len(train_acc_log) + 1), train_acc_log)),
            delimiter=',', header='Epoch,Img_acc,RIR_acc,Ave_acc', comments='', fmt='%d,%.2f,%.2f,%.2f')
    np.savetxt(val_eval_path, np.column_stack((np.arange(1, len(val_acc_log) + 1), val_acc_log)),
        delimiter=',', header='Epoch,Img_acc,RIR_acc,Ave_acc', comments='', fmt='%d,%.2f,%.2f,%.2f')
