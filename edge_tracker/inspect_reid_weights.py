import importlib.util
import os
import torch

print('torch', torch.__version__)
print('torchreid', importlib.util.find_spec('torchreid'))
print('exists', os.path.exists('transreid_msmt17.pth'))
p = 'transreid_msmt17.pth'
if os.path.exists(p):
    obj = torch.load(p, map_location='cpu')
    print('type', type(obj))
    if isinstance(obj, dict):
        print('keys', list(obj.keys())[:20])
        if 'state_dict' in obj:
            state = obj['state_dict']
            print('state_dict_type', type(state))
            print('state_dict_len', len(state))
            print('state_dict_keys', list(state.keys())[:20])
        else:
            print('first_item', next(iter(obj.items())) if obj else None)
