import os
import random
import argparse
import numpy as np
import open3d as o3d

bad_list = [
    'de45798ef57fe2d131b4f9e586a6d334', # 02691156
    '52e27aecdd55c1bf5b03388497f76a9e', # 02691156
    'a5d68126acbd43395e9e2656aff7dd5b', # 02691156
    '31b201b7346e6cd15e9e2656aff7dd5b', # 02691156
    'b976a48c015d6ced5e9e2656aff7dd5b', # 02691156
    'd837b5228d9c010bbe584d85bf07b4ac', # 02691156
    'd9f0f7cff584b60d826faebfb5cddf3c', # 02801938
    '3d51b8ad6b622c75dd5c7f6e6acea5c1', # 02818832
    '2e8f1b6cb9b4f568316a315354726289', # 02828884
    '595e48c492a59d3029404a50338e24e7', # 02828884
    'b2a585ba5f0b4a25e76bc197b3a3ffc0', # 02828884
    '1d94afb9894bf975e76bc197b3a3ffc0', # 02828884
    '2af98dcf936576b155f28299c0ff52b7', # 02828884
    'd7f1a22419268800e76bc197b3a3ffc0', # 02871439
    '48d58a4f43125e7f34282af0231ccf9c', # 02871439
    'bbddf00c2dda43a2f21cf17406f1f25', # 02871439
    '2f30f402c00a166368068eb0ef40fbb1', # 02871439
    'abe557fa1b9d59489c81f0389df0e98a', # 02876657
    'c099c763ee6e485052470de2774d6099', # 02924116
    '809d5384f84d55273a11565e5be9cf53', # 02933112
    '9a688545112c2650ca703e831bf56f93', # 02933112
    'cdc3762d2846133adc26ec30fe28341a', # 02933112
    '2c1af98d2058a8056588620c25b809f9', # 02933112
    '3e52f25b8f66d9a8adf3df9d9e46d0', # 02933112
    'd4f4b5bf712a96b13679ccb6aaef8b00', # 02933112
    'd707228baece2270c473585373fc1fd0', # 02933112
    '6ee01e861f10b1f044175b4dddf5be08', # 02933112
    '3ffeec4abd78c945c7c79bdb1d5fe365', # 02958343
    '7aa9619e89baaec6d9b8dfa78596b717', # 02958343
    '806d740ca8fad5c1473f10e6caaeca56', # 02958343
    'ea3f2971f1c125076c4384c3b17a86ea', # 02958343
    '207e69af994efa9330714334794526d4', # 02958343
    '2307b51ca7e4a03d30714334794526d4', # 02958343
    '5bf2d7c2167755a72a9eb0f146e94477', # 02958343
    '9fb1d03b22ecac5835da01f298003d56', # 02958343
    '3c33f9f8edc558ce77aa0b62eed1492', # 02958343
    '5973afc979049405f63ee8a34069b7c5', # 02958343
    '302612708e86efea62d2c237bfbc22ca', # 02958343
    'f5bac2b133a979c573397a92d1662ba5', # 02958343
    '986ed07c18a2e5592a9eb0f146e94477', # 02958343
    'e6c22be1a39c9b62fb403c87929e1167', # 02958343
    'd6ee8e0a0b392f98eb96598da750ef34', # 02958343
    '407f2811c0fe4e9361c6c61410fc904b', # 02958343
    '4ddef66f32e1902d3448fdcb67fe08ff', # 02958343
    '8070747805908ae62a9eb0f146e94477', # 02958343
    '93ce8e230939dfc230714334794526d4', # 02958343
    'fcdbba7127ad58a84155fcb773414092', # 02992529
    '922380f231a342cf388f6c7a9d3e1552', # 02992529
    'a4910da0271b6f213a7e932df8806f9e', # 02992529
    '61e83782d92bf80b7548281e465c9303', # 02992529
    '2dd729a07206d1f5746cec00e236149d', # 03001627
    'bafb9c9602d00b3e50b42dfb503f5a87', # 03001627
    '5af850643d64c2621b17743c18fb63dc', # 03001627
    '62127325480bec8d2c6c98851414a9d8', # 03001627
    '2ae70fbab330779e3bff5a09107428a5', # 03001627
    'bee929929700e99fad8a0ee9b106700e', # 03001627
    '64871dc28a21843ad504e40666187f4e', # 03001627
    'fe8b246b47321320c3bd24f986301745', # 03001627
    '5b51df75df88c639f51f77a6d7299806', # 03001627
    'ebf8166bacd6759399513f98ce033426', # 03001627
    '42db4f765f4e6be414038d588fd1342f', # 03001627
    'a5abf524f9b08432f51f77a6d7299806', # 03001627
    'ff3581996365bdddc3bd24f986301745', # 03001627
    '4abbf49c3177b32c9f613b70ec8c2f82', # 03001627
    'c487441af8ac37a733718c332b0d7bfd', # 03001627
    '77dcd07d59503f1014038d588fd1342f', # 03001627
    '5db74dcfc73a3ea2f2ca754af3aaf35', # 03001627
    '790d554d7f9b040299513f98ce033426', # 03001627
    'adcc6534d6db1ff9dff990aa66c50fe0', # 03001627
    '3f4f6f64f5ae57a14038d588fd1342f', # 03001627
    '5875ca8510373873f51f77a6d7299806', # 03001627
    '1196ffab55e431e11b17743c18fb63dc', # 03001627
    '6498a5053d12addb91a2a5174703986b', # 03001627
    '78261b526d28a436cc786970133d7717', # 03001627
    '64ef0e07129b6bc4c3bd24f986301745', # 03001627
    'c833ef6f882a4b2a14038d588fd1342f', # 03001627
    '2d08a64e4a257e007135fc51795b4038', # 03001627
    '6f194ba6ba254aacf51f77a6d7299806', # 03001627
    'c06b5a7aa4557182f51f77a6d7299806', # 03001627
    '3c4ed9c8f76c7a5ef51f77a6d7299806', # 03001627
    'c70c1a6a0e795669f51f77a6d7299806', # 03001627
    'c535629f9661293dc16ef5c633c71b56', # 03001627
    '453be11e44a230a0f51f77a6d7299806', # 03001627
    'b13143d5f71e38d24738aee9841818fe', # 03046257
    'b0952767eeb21b88e2b075a80e28c81b', # 03211117
    'f59e8dac9451d165a68447d7f5a7ee42', # 03513137
    'dfc0f60a1b11ab7c388f6c7a9d3e1552', # 03513137
    'e4bd6dda8d29106ca16af3198c99de08', # 03593526
    '5d636123af31f735e76bc197b3a3ffc0', # 03593526
    'eaaa14d9fab71afa5b903ba10d2ec446', # 03593526
    '998c2b635922ace93c8d0fdfb1cc2535', # 03593526
    '94235090b6d5dbfae76bc197b3a3ffc0', # 03593526
    '12ec19e85b31e274725f67267e31c89', # 03593526
    '7de119e0eb48a11e3c8d0fdfb1cc2535', # 03593526
    '79ee45aa6b0c86064725f67267e31c89', # 03593526
    '5e515b18ed17a418b056c98b2e5e5e4e', # 03624134
    '8facbe9d9f4da233d15a5887ec2183c9', # 03624134
    '94eae2316754482627d265f13671170a', # 03636649
    '8464de18cd5d14e138435fc2a8dffe1b', # 03636649
    'abf04f17d2c84a160e37b3f76995f8b', # 03636649
    '1a44dd6ee873d443da13974b3533fb59', # 03636649
    'f97011a0bae2b4062d1c72b9dec4baa1', # 03636649
    '5926d3296767ab28543df75232f6ff2b', # 03636649
    '5aefcf6b38e180f06df11193c72a632e', # 03636649
    '59e852f315216f95ba9df3ea0397b1a6', # 03636649
    'a4d779cf204ff052894c6619653a3264', # 03642806
    '466ea85bb4653ba3a715ae636b111d77', # 03642806
    '1f2a8562a2de13a2c29fde65e51f52cb', # 03691459
    '539d5ef3031544fb63c6a0e477a59b9f', # 03691459
    '1788d15f49a57570a0402637f097180', # 03691459
    'a9957cf39fdd61fc612f7163ca95602', # 03691459
    'ba0f7f4b5756e8795ae200efe59d16d0', # 03790512
    '2d97e5970822beddde03ab2a27ba7531', # 03991062
    '69240d39dfbc47a0d15a5887ec2183c9', # 04090263
    '249e0936ae06383ab056c98b2e5e5e4e', # 04090263
    'c0d736964ddc268c9fb3e3631a88cdab', # 04099429
    '4c29dcad235ff80df51f77a6d7299806', # 04256520
    'dcba7668017df61ef51f77a6d7299806', # 04256520
    '621dab02dc0ac842e7891ff53b0e70d', # 04256520
    '1545a13dc5b12f51f77a6d7299806', # 04256520
    '7b1d07d932ca5890f51f77a6d7299806', # 04256520
    '22b11483d6a2461814038d588fd1342f', # 04256520
    '8159bdc913cd8a23debd258f4352e626', # 04256520
    '3a69f7f6729d8d48f51f77a6d7299806', # 04256520
    'd6f81af7b34e8da814038d588fd1342f', # 04256520
    '34d7a91d639613f6f51f77a6d7299806', # 04256520
    '221e8ea6bdcc1db614038d588fd1342f', # 04256520
    '32859e7af79f494a14038d588fd1342f', # 04256520
    '64ee5d22281ef431de03ab2a27ba7531', # 04256520
    '6e213a2ecc95c30544175b4dddf5be08', # 04256520
    '4385e447533cac72d1c72b9dec4baa1', # 04256520
    '7e832bc481a3335614038d588fd1342f', # 04256520
    '3eb9e07793635b30f51f77a6d7299806', # 04256520
    'bda8c00b62528346ad8a0ee9b106700e', # 04330267
    '581d698b6364116e83e95e8523a2fbf3', # 04379243
    'c715a29db7d888dd23f9e4320fcb3664', # 04379243
    '15bcc664de962b04e76bc197b3a3ffc0', # 04379243
    '40ee6a47e485cb4d41873672d11706f4', # 04379243
    'a13c36acbc45184de76bc197b3a3ffc0', # 04379243
    'f835366205ba8afd9b678eaf6920cd86', # 04379243
    '639d99161524c7dd54e6ead821103617', # 04379243
    '97a137cc6688a07c90a9ce3e4b15521e', # 04379243
    'c0ac5dea15f961c9e76bc197b3a3ffc0', # 04379243
    '376c99ec94d34cf4e76bc197b3a3ffc0', # 04379243
    '21b8b1e51237f4aee76bc197b3a3ffc0', # 04379243
    'a088285efee5f0dbbc6a6acad56465f2', # 04379243
    'b7eecafd15147c01fabd49ee8315e8b9', # 04379243
    'dc9a7d116351f2cca16af3198c99de08', # 04379243
    '15a95cddbc40994beefc4457af135dc1', # 04379243
    'c5a02d586ea431a1e76bc197b3a3ffc0', # 04379243
    'b8efc08bc8eab52a330a170e9ceed373', # 04379243
    '904ad336345205dce76bc197b3a3ffc0', # 04379243
    '6e77d23b324ddbd65661fcc99c72bf48', # 04379243
    'ac35b0d3d4b33477e76bc197b3a3ffc0', # 04379243
    '81d84727a6da7ea7bb8dc0cd2a40a9a4', # 04379243
    '9377b1b5c83bb05ce76bc197b3a3ffc0', # 04379243
    'e7580c72525b4bb1cc786970133d7717', # 04379243
    'f078a5940280c0a22c6c98851414a9d8', # 04379243
    'e2efc1a73017a3d2e76bc197b3a3ffc0', # 04379243
    '31f09d77217fc9e0e76bc197b3a3ffc0', # 04379243
    '5026668bb2bcedebccfcde790fc2f661', # 04379243
    'de96be0a27fe1610d40c07d3c15cc681', # 04379243
    'ce82dbe1906e605d9b678eaf6920cd86', # 04379243
    '41283ae87a29fbbe76bc197b3a3ffc0', # 04379243
    '187804891c09be04f1077877e3a9d430', # 04379243
    'aaaba1bbe037d3b1e406974af41e8842', # 04379243
    '52eaeaf85846d638e76bc197b3a3ffc0', # 04379243
    'fe4c20766801dc98bc2e5d5fd57660fe', # 04379243
    'a333abca08fceb434eec4d2d414b38e0', # 04379243
    '7f39803c32028449e76bc197b3a3ffc0', # 04379243
    '15bc36a3ce59163bce8584f8b28da0ba', # 04379243
    '703c1f85dc01baad9fb3e3631a88cdab', # 04379243
    'e5ede813e9f07ee4f3e39f2e17005efc', # 04379243
    'b0e8c331eacdc9bef3e39f2e17005efc', # 04379243
    'e36cda06eed31d11d816402a0e81d922', # 04379243
    'b9bf493040c8b434f3e39f2e17005efc', # 04379243
]

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='ShapeNetCore.v2')
parser.add_argument('--store_dir', type=str, default='./data')
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.store_dir, exist_ok=True)
    folders = sorted([p for p in os.listdir(
        args.data_root) if p != 'taxonomy.json'])

    for folder in folders:
        os.makedirs(os.path.join(args.store_dir, folder), exist_ok=True)
        obj_names = os.listdir(os.path.join(args.data_root, folder))
        # pc = []
        for i, obj_name in enumerate(obj_names):
            npy_path = os.path.join(args.store_dir, folder, '{}.npy'.format(obj_name))
            if obj_name in bad_list or os.path.exists(npy_path):
                continue
            print('{}/{} {} {}'.format(i+1, len(obj_names), folder, obj_name))
            path = os.path.join(
                args.data_root, folder, obj_name,
                'models', 'model_normalized.obj')

            mesh = o3d.io.read_triangle_mesh(path)
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
            np.save(npy_path, np.asarray(pcd.points))
            del mesh, pcd