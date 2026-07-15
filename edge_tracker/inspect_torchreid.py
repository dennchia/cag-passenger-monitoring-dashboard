import os
import importlib.util
import torch

import torchreid
from torchreid import models

print('torch', torch.__version__)
print('torchreid spec', importlib.util.find_spec('torchreid'))
print('exists', os.path.exists('transreid_msmt17.pth'))
print('models attrs', [a for a in dir(models) if not a.startswith('_')][:80])
print('has build_model', hasattr(models, 'build_model'))
if hasattr(models, 'build_model'):
    import inspect
    print('build_model sig', inspect.signature(models.build_model))
    if models.build_model.__doc__:
        print('build_model doc first line:', models.build_model.__doc__.splitlines()[0])

p = 'transreid_msmt17.pth'
if os.path.exists(p):
    obj = torch.load(p, map_location='cpu')
    print('checkpoint type', type(obj))
    if isinstance(obj, dict):
        print('checkpoint keys', list(obj.keys())[:20])
        if 'state_dict' in obj:
            print('state_dict length', len(obj['state_dict']))
            print('state_dict sample keys', list(obj['state_dict'].keys())[:20])
