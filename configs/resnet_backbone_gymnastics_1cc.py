config = {
    'params': {
        "backbone": {
            "kernel_size": 3,
            "output_dim": 128,
            "input_dim": 3, # for mnist it's 1. for gymnastics, 3.
            "stride": 2,
            "padding": 1,
            "out_img_size": 32
        },
        "primary_capsules": {
            "kernel_size": 3,
            "stride": 2,
            "input_dim": 128,
            "caps_dim": 36,  # output dim is 1024 = num_caps * caps_dim
            "num_caps": 16,
            "padding": 0,
            "out_img_size": 31
        },
        "capsules": [{
            "type": "CONV",
            "num_caps": 16,
            "caps_dim": 36,
            "kernel_size": 3,
            "stride": 1,
            "matrix_pose": True,
            "out_img_size": 29,
        }, {
            "type": "FC",
            "num_caps": 10,
            "caps_dim": 36,
            "matrix_pose": True,
        }],
        "class_capsules": {
            "num_caps": 1, # was 10
            "caps_dim": 36, 
            "matrix_pose": True,
            "object_dim": 36,
        },
    },
}
