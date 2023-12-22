import math
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer


logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)        # 如果是二维卷积函数，将W从第一维之后开始展平成二维矩阵
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        # 待量化二维权重矩阵，如果是Conv1D(类似线性层)，转置回来，如果是Conv2d，平铺
        # Hessian矩阵只与X 本层input 有关和其他weight没关系，所以量化二维权重矩阵所有行对应的Hessian一样，这样分割是因为类比perchannel量化，每个outputchannel有独立的量化参数
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # H=2XX^T，Hessian只与X(本层input)有关和其它weight没关系，即变换后的input的size和X一样
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0       # sample的数量
        self.quantizer = Quantizer()        # 每个layer一个quantizer

    #以ResNet第一层卷积为例，根据输入构建weight矩阵的Hessian矩阵
    #什么是权重的二维矩阵呢？这里举例resnet第一层卷积权重(weight)是四维的(64,3,7,7),
    #因为output channel独立量化,把4维变2维时候Oc不变，其它维度融合，于是变成了(64, 3*7*7)
    #根据input要计算这个权重矩阵(64, 147)的Hession阵(64, 147, 147)， 不过每一行独立量化，64行每一行的Hessian矩阵都相同，其实都是相同的(147, 147)
    #下面主要逻辑就是要把inp变成(147, X)的形状，然后用inp*inp.t()求出Hessian矩阵
    def add_batch(self, inp, out):
        #假设是resnet第一层卷积:inp.shape(128,3,224,224) out.shape(128,64,112,112)
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)  # 如果是二维，在第0维插入一个维度，其实shape[0]就是1，是不是batch_size?
        tmp = inp.shape[0] #input数据的batch=128
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))  # 将inp的shape除了最后一维全部展平（128*3*224，224）
            inp = inp.t()   # 转置inp（224， 128*3*224）
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size, #(7,7)
                dilation=self.layer.dilation,   #(1,1)
                padding=self.layer.padding,     #(3,3)
                stride=self.layer.stride        #(2,2)
            )
            # outputs (B, N, L) N:表示生成后每个局部块的大小 input_channel * kernel_size * kernel_size。L：表示有多少个局部块。
            # 即从一个批次的输入样本中，提取出滑动的局部区域块。
            inp = unfold(inp)                   # 把layer的输入变形到权重二维矩阵的输入
            #inp.shape=(128, 147, 12544)=(128, 3*7*7, 112*112)=(batch, 权重二维阵的列, 下一步的输入)
            inp = inp.permute([1, 0, 2])        
            inp = inp.flatten(1)                # (147, 1605632)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp        # nsamples的数量更新
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t()) # shape是(147, 147)
        # 这样的结果是可以保证每个batch的XX^T的结果都是根据前面所有累积平均过的，即2/(n+an) * (X1X1^T+X2X2^T+...)

    def fasterquant(
        self, blocksize=128, percdamp=.01, group_size=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()      # 首先拿到权重的clone
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)                    # 和上面初始化一样，取出W然后变成二维矩阵
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()                      # 计时

        if not self.quantizer.ready():          # 即scale如果非零，即已经被量化过，目前一个quantizer里面只有一个tensor shape为1 的scale
            self.quantizer.find_params(W, weight=True)  # 初始化scale以及zero

        H = self.H                              # 一个个batch累积得到的Hessian矩阵
        del self.H                              # 强制释放资源，手动del
        dead = torch.diag(H) == 0               # 获取Hessian阵对角线上元素等于0的下标
        H[dead, dead] = 1                       # 将这些Hessian矩阵的对角线元素改成1
        W[:, dead] = 0                          # 将dead这一列的W全部置为0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, group_size):                            # 每个group find一组param
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + group_size)], weight=True)        # 先找到scale和zero
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)        # 对Hessian矩阵对角线上的元素做降序排列
            W = W[:, perm]                                              # 对W进行重排，即元素大的排在前面
            H = H[perm][:, perm]                                        # 对H的每行每列的元素也按照perm下标进行排列
            invperm = torch.argsort(perm)                               # 将索引重新按照下标顺序排回来 同理Q的顺序和g_idx的顺序也排回来

        Losses = torch.zeros_like(W)            # Losses和Q 两个形如W的矩阵
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))             # 保证Hessian矩阵逆的稳定性
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp                                   # 即在每个对角线元素上加上damp系数
        H = torch.linalg.cholesky(H)                            # H逆的cholesky表达形式可以确保H^-1是对称正定的
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):            # 128一组
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1                                     # 一个block的大小

            W1 = W[:, i1:i2].clone()                            # 这个block要更新的权重
            Q1 = torch.zeros_like(W1)                           # 待量化存储的Qweight
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]                          # 对应的Hinv1.shape是(128，128)

            for i in range(count):
                w = W1[:, i]                                    # size [768], 并行处理所有行
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:                          # 每个group find一组param
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)], weight=True)
                            
                        if ((i1 + i) // group_size) - now_idx == -1:            # 说明已经遍历完一组group，将scale和zero append进来
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]
                        
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q        # 将quantize之后的值放到Q1中存储
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d                              # 每一步迭代的步长 [768]
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))    # 更新Block里面没有量化的权重(如论文中的图Hinv1[i, i:]多次迭代后呈上三角形)
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1                                    # 将量化之后的weight值放到Q中存储
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])          # 更新非Block里面的权重(论文中所谓的lazily batch)
            # 只有在一个块被完全处理后，才对整个H^-1和W进行全局更新， Hinv每次迭代的时候不重新计算，只是重新取一块
            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

        torch.cuda.synchronize()        # torch.cuda.synchronize()是一个同步点,保证了在它之后CPU和GPU的同步,避免了处于异步状态时可能出现的问题 即所有的matmul都跑完了
        logger.info(f'duration: {(time.time() - tick)}')
        logger.info(f'avg loss: {torch.sum(Losses).item() / self.nsamples}')

        group_size = group_size if group_size != -1 else self.columns       # 128
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]    # 生成一个列表，每个column对应的group的g_idx
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        
        # 组合两者的问题在于,组是按照整体量化过程的行顺序顺序生成的。所以如果行是无序量化的(即启用act-order),你最终会得到一个矩阵,其中任何行都可以属于任何组,由一个单独的组索引确定。
        # 现在,由于行在推理时是顺序处理的,你必须不断重新加载量化参数,这会非常慢。
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)     # 将Q量化之后的weight存到layer的权重中
        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)     # [768, 6]
        zero = torch.cat(zero, dim=1)       # [768, 6]
        return scale, zero, g_idx

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
