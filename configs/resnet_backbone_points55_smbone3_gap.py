# nce_select has 561704 params.

import torchvision.transforms as transforms

config = {
    'params': {
        "backbone": {
            "kernel_size": 3,
            "output_dim": 64,
            "input_dim": 3,
            "stride": 2,
            "padding": 1,
            "out_img_size": 1024, # This should govern it, not be receptive.
            "inp_img_size": 2048
        },
        "primary_capsules": {
            "kernel_size": 3,
            "stride": 2,
            "input_dim": 64,
            "caps_dim": 36,
            "num_caps": 32,
            "padding": 1,
            "out_img_size": 512
        },
        "capsules": [{
            "type": "CONV",
            "num_caps": 32,
            "caps_dim": 36,
            "kernel_size": 3,
            "stride": 2,
            "out_img_size": 127
        }, {
            "type": "CONV",
            "num_caps": 32,
            "caps_dim": 36,
            "kernel_size": 3,
            "stride": 2,
            "out_img_size": 127
        }, {
            "type": "FC", 
            "num_caps": 32,
            "caps_dim": 36,
            "gap": True
        }],
        "class_capsules": {
            "num_caps": 80, 
            "caps_dim": 36,
        },
    },
}
