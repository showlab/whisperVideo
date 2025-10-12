import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

def _cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for TalkNet but is not available')
    return torch.device('cuda')

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()
        device = _cuda_device()
        self.model = talkNetModel().to(device)
        self.lossAV = lossAV().to(device)
        self.lossA = lossA().to(device)
        self.lossV = lossV().to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        device = _cuda_device()
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(device))
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(device))
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).to(device)
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        device = _cuda_device()
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].to(device))
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(device))
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels = labels[0].reshape((-1)).to(device)
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=_cuda_device())
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

    def forward(self, inputA: torch.Tensor, inputV: torch.Tensor) -> torch.Tensor:
        """Inference forward used for DataParallel.

        Expects:
        - inputA: (B, T_a, 13) float32 tensor
        - inputV: (B, T_v, 112, 112) float32 tensor
        Returns:
        - out: (B*T_step, 256) flattened representation passed to lossAV for scoring
        Notes:
        - Do NOT move tensors to device here; DataParallel scatters inputs to replicas.
        - Matches the exact sequence used in evaluation code paths to preserve outputs.
        """
        audioEmbed = self.model.forward_audio_frontend(inputA)
        visualEmbed = self.model.forward_visual_frontend(inputV)
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        out = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        return out

    def forward(self, inputA: torch.Tensor, inputV: torch.Tensor):
        """End-to-end forward used for multi-GPU inference (DataParallel/DDP).

        This preserves the exact evaluation path used elsewhere:
          audio -> audio_frontend ->
          visual -> visual_frontend ->
          cross attention -> audio_visual backend -> out (B*T, 256)

        The caller is responsible for computing scores via self.lossAV.forward(out, labels=None).
        """
        # Expect inputA: (B, T_a, 13), inputV: (B, T_v, 112, 112) on the correct device
        audioEmbed = self.model.forward_audio_frontend(inputA)
        visualEmbed = self.model.forward_visual_frontend(inputV)
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        out = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        return out
