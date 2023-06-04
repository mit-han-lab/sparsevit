import torch
import mmcv
from collections import OrderedDict


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt

path = '../Swin-Transformer-Object-Detection/work_dirs/672_nas6_drop10_36e/epoch_36.pth'

ori_ckpt = torch.load(path)
ori_dict = ori_ckpt['state_dict']
new_dict = OrderedDict()
head_dict = OrderedDict()
for name in ori_dict:
    if name.startswith('backbone'):
        new_dict[name[9:]] = ori_dict[name]
    else:
        head_dict[name] = ori_dict[name]

new_dict = swin_converter(new_dict)
for k, v in head_dict.items():
    new_dict[k] = v

new_ckpt = {}
new_ckpt['state_dict'] = new_dict
torch.save(new_ckpt, 'checkpoint/converted_sparsevit_nas6.pth')