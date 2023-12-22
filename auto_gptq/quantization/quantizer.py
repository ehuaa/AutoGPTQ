from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)


def quantize(x, scale, zero, maxq):     # 在第一维unsqueeze插入了一维，[768]->[768, 1]
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)           # 这里的quantize就是量化到最近的一个4bit/3bit的量化网格，再反量化回来，因为gptq其实就是最小化原权重和反量化回来之后的权重 让其之间的误差最小


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):                                      # gptq里面默认perchannel是True，sym是True，sym表示对称量化

        self.maxq = torch.tensor(2 ** bits - 1) # 量化的范围即为[0, 2^bits - 1]
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape         # [768, 768] maxq 15
        if self.perchannel:
            if weight:          # 如果是weight，直接维度1之后平铺
                x = x.flatten(1)
            else:               # 否则如果不是权重
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]) # 从bchw 把c提前
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:                                       
            x = x.flatten().unsqueeze(0)    # flatten成一维，在unsqueeze第0维插入1维

        tmp = torch.zeros(x.shape[0], device=dev)   # tmp 变为和weight 行数一样的一维数组 [768] float32
        xmin = torch.minimum(x.min(1)[0], tmp)      # 先找到每一行的最小值，然后和tmp去比较 取最小，即最小值不大于0 [768]
        xmax = torch.maximum(x.max(1)[0], tmp)      # 先找到每一行的最大值，然后和tmp去比较 取最大，即最大值不小于0 [768]

        if self.sym:        # 如果是对称量化，即零点就是0，xmax取绝对值的最大值
            xmax = torch.maximum(torch.abs(xmin), xmax)  # [768]
            tmp = xmin < 0                               # 找到xmin小于0的数
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]                   # 用bool tensor作为mask，将xmin小于0的全部变为xmax的相反数
        tmp = (xmin == 0) & (xmax == 0)                  # 对于xmin xmax都是0的，强行设置其最大最小值为+-1
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq                              # 计算scale 即映射关系 这里是4bit 用maxmin方法 [768]
            if self.sym:                                                        
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)    # 初始化为self.scale形状，填充值为(self.maxq + 1) / 2 即正好把零点设置在maxq的中间位置
            else:
                self.zero = torch.round(-xmin / self.scale)                     # 如果是非对称量化，将xmin映射到0，向上找零点即所有的负数全部映射到负值区域

        if self.mse:            # mse 确定scale zero 不用minmax
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)     # 如果不是perchannel的话，scale和zero就是对于每个元素的 peritem？
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)      # scale 和 zero都reshape成[768,1] 即一个个一维 tensor
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer"]
