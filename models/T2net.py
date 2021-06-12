
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import common



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, V, K, Q):

        ### search
        Q_unfold=F.unfold(Q, kernel_size=(3, 3), padding=1)
        K_unfold=F.unfold(K,kernel_size=(3,3),padding=1)
        K_unfold = K_unfold.permute(0, 2, 1)

        K_unfold = F.normalize(K_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        Q_unfold= F.normalize(Q_unfold, dim=1)  # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(K_unfold , Q_unfold)  # [N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

        ### transfer
        V_unfold = F.unfold(V, kernel_size=(3, 3), padding=1)

        T_lv3_unfold = self.bis(V_unfold, 2, R_lv3_star_arg)


        T_lv3 = F.fold(T_lv3_unfold, output_size=Q.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, Q.size(2), Q.size(3))

        return S,T_lv3


class T2Net(nn.Module):
    def __init__(self, upscale_factor, input_channels, target_channels, n_resblocks, n_feats, res_scale, bn=False, act=nn.ReLU(True), conv=common.default_conv, head_patch_extraction_size=5, kernel_size=3, early_upsampling=False):

        super(T2Net,self).__init__()

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = res_scale
        self.act = act
        self.bn = bn
        self.input_channels = input_channels
        self.target_channels = target_channels

        m_head1 = [conv(input_channels, n_feats, head_patch_extraction_size)]
        m_head2 = [conv(input_channels, n_feats, head_patch_extraction_size)]

        m_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale, bn=self.bn
            ) for _ in range(n_resblocks)
        ]
        m_body1.append(conv(n_feats, n_feats, kernel_size))

        m_body2 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale, bn=self.bn
            ) for _ in range(n_resblocks)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        m_conv1=[nn.Conv2d(n_feats*2,n_feats,kernel_size=1) for _ in range(n_resblocks)]

        #head
        self.head1 = nn.Sequential(*m_head1)
        self.head2 = nn.Sequential(*m_head2)

        #body
        self.body1=nn.Sequential(*m_body1)
        self.body2=nn.Sequential(*m_body2)

        #kersize=1 conv
        self.conv1=nn.Sequential(*m_conv1)

        #tail
        m_tail_late_upsampling = [
            common.Upsampler(conv, upscale_factor, n_feats, act=False),
            conv(n_feats, target_channels, kernel_size)
        ]
        m_tail_early_upsampling = [
            conv(n_feats, target_channels, kernel_size)
        ]
        if early_upsampling:
            self.tail = nn.Sequential(*m_tail_early_upsampling)
        else:
            self.tail = nn.Sequential(*m_tail_late_upsampling)#走这个

        self.b_tail=nn.Conv2d(n_feats,target_channels,kernel_size=1)

        #transformer modules
        m_transformers=[Transformer() for _ in range(n_resblocks)]

        self.transformers=nn.Sequential(*m_transformers)

    def forward(self, input):

        x1=self.head1(input)
        x2=self.head2(input)

        res1=x1
        res2=x2

        for i in range(self.n_resblocks):
            x1=self.body1[i](x1)
            x2=self.body2[i](x2)
            S,T=self.transformers[i](x2,x2,x1)
            T=torch.cat([x1,T],1)
            T=self.conv1[i](T)
            x1=x1+T*S

        y1=self.tail(x1+res1)
        y2=self.b_tail(x2+res2)

        return y1,y2



