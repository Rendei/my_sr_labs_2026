# The script downloads model's weights


# Import of modules
import os
from collections import OrderedDict

import torch

from common.dataprep import download_file


def load_model(model, lines, save_path, reload=False):
    # Load model's weights
    
    if not os.path.exists(save_path):
        os.mkdir(save_path, mode=0o777)

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]

        # Download files (pure Python replacement of wget)
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            
            try:
                download_file(url, os.path.join(save_path, outfile))
            
            except Exception as e:
                raise ValueError('Download failed %s (%s). If download fails repeatedly, use alternate URL on the VoxCeleb website.'%(url, e))

        print('File %s is downloaded.'%outfile)
        
    checkpoint = torch.load(os.path.join(save_path, 'baseline_v2_ap.model'), map_location='cpu', weights_only=True)
    
    model_weight = OrderedDict()

    for key in checkpoint.keys():
        
        if '__S__' in key:
            model_weight[key[6:]] = checkpoint[key]
            
    model.load_state_dict(model_weight)
    
    return model