import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg
from lpips import LPIPS
class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.lpips=LPIPS(net='vgg')
        self.lambda_lpips=cfg.train['lambda_lpips']
        self.lambda_mask=cfg.train['lambda_mask']
    def forward(self, batch):
        output = self.net(batch)

        scalar_stats = {}
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])
        mask_loss=self.color_crit(output['weight'].squeeze(-1),batch['mask'])
        lpip_loss=self.lpips(output['rgb'].permute(0,3,1,2).float(), batch['rgb'].permute(0,3,1,2).float(),normalize=True).mean()
        scalar_stats.update({'lpips':lpip_loss})
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss
        loss+=self.lambda_lpips*lpip_loss
        loss+=self.lambda_mask*mask_loss

        psnr = -10. * torch.log(color_loss.detach()) / \
                torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
