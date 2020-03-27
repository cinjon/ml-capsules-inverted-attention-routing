config = {
    'params': {
        "backbone": {
            "kernel_size": 3,
            "output_dim": 1024,
            "input_dim": 1,
            "stride": 2,
            "padding": 1,
            "out_img_size": 20 # 32
        },
        "primary_capsules": {
            "kernel_size": 3,
            "stride": 2,
            "input_dim": 1024,
            "caps_dim": 64,  # output dim is 1024 = num_caps * caps_dim
            "num_caps": 16,
            "padding": 0,
            "out_img_size": 9 # 15
        },
        "capsules": [{
            "type": "CONV",
            "num_caps": 16,
            "caps_dim": 64,
            "kernel_size": 3,
            "stride": 1,
            "matrix_pose": True,
            "out_img_size": 7 # 13,
        }, {
            "type": "FC",
            "num_caps": 10,
            "caps_dim": 64,
            "matrix_pose": True,
        }],
        "class_capsules": {
            "num_caps": 10,
            "caps_dim": 64,
            "matrix_pose": True
        },
    },
}
