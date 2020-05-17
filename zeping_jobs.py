"""Run the jobs in this file.

Example running jobs:

python zeping_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from zeping_run_on_cluster import do_jobarray

email = 'zz2332@nyu.edu'
code_directory = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/ml-capsules-inverted-attention-routing'

def run(find_counter=None):
    job = {
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/results',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/MNIST',
        'affnist_data_root': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/capsules/data/affNIST'
    }

    # Padded MNIST (40x40) NCE
    if find_counter == 0:
        num_gpus = 4
        time = 24
        job.update({
            'name': '2020.05.15',
            # 'optimizer': 'adam',
            'config': 'resnet_backbone_movingmnist2_20ccgray_img40',
            'criterion': 'nceprobs_selective',
            'triangle_lambda': 1,
            # 'num_routing': 1,
            'dataset': 'MovingMNist2',
            'batch_size': 32,
            'num_gpus': num_gpus,
            'time': time,
            # 'num_workers': 2,
            'epoch': 500,
            'schedule_milestones': '10,30',
            'use_scheduler': True,
            'weight_decay': 0.0005,
            'lr': 0.0003,
            'schedule_gamma': 0.3,
            'no_colored': True,
            'num_mnist_digits': 1,
            'fix_moving_mnist_center': True,
            # 'do_mnist_test_every': 1,
            # 'do_mnist_test_after': -1,
            'do_tsne_test_every': 1,
            'do_tsne_test_after': -1,
            'use_presence_probs': True,
            'lambda_sparse_presence': 0.3,
            'use_cuda_tsne': False,
            'presence_loss_type': 'l2norm',
            'fix_moving_mnist_angle': False,
            'triangle_margin_lambda': 1.0,
            'margin_gamma2': 0.1,
            'triangle_cos_lambda': 1.0,
            'no_use_angle_loss': True,
            'no_use_hinge_loss': True,
            'num_output_classes': 10,
            'lambda_between_entropy': 3.0,
            'lambda_within_entropy': 1.0,
            'presence_diffcos_lambda': 10.0,
            'use_class_sampler': False,
            'no_use_nce_loss': True,
            'use_nce_probs': True,
            'use_simclr_xforms': False,
            'use_simclr_nce': True,
            'step_length': 0.,
            'use_rand_presence_noise': True,
            'nce_presence_temperature': 0.1,
            'use_diff_class_digit': True,
            'nceprobs_selection': 'ncelinear_none',
            'image_size': 40, # 28
            'no_hit_side': True,
            # 'center_discrete': False,
            'discrete_angle': True,
            'nceprobs_selection_temperature': 1.0,
            'simclr_selection_strategy': 'default',
            'ncelinear_anchorselect': '21',
            # 'center_discrete_count': 3,
            'counter': find_counter,
            'seed': 0,
            'mnist_padding': True,
        })

        # do_jobarray(email, num_gpus, counter, job, time)

        return find_counter, job
    # Selective Reorder
    elif find_counter == 1:
        num_gpus = 1
        time = 24
        job.update({
            'name': '2020.05.15',
            'config': 'resnet_backbone_movingmnist2_20ccgray',
            'criterion': 'selective_reorder',
            'num_gpus': num_gpus,
            'time': time,
            'do_tsne_test_every': 1,
            'do_tsne_test_after': -1,
            'nceprobs_selection': 'ncelinear_maxfirst',
            'presence_loss_type': 'sigmoid_only',
            'batch_size': 8,
            'use_diff_class_digit': True,
            'lr': 1e-4,
            'weight_decay': 5e-4,
            'use_presence_probs': True,
            'nceprobs_selection': 'ncelinear_maxfirst',
            'presence_loss_type': 'sigmoid_only',
            'use_nce_probs': True,
            'use_simclr_nce': True,
            'no_colored': True,
            'fix_moving_mnist_center': True,
            'num_mnist_digits': 1,
            'counter': find_counter,
            'use_cuda_tsne': True,
            'tsne_batch_size': 32,
        })

        return find_counter, job
    # Fixed Center MNIST NCE
    elif find_counter == 2:
        num_gpus = 4
        time = 48
        job.update({
            'name': '2020.05.16',
            # 'optimizer': 'adam',
            'config': 'resnet_backbone_movingmnist2_20ccgray_img40',
            'criterion': 'nceprobs_selective',
            'triangle_lambda': 1,
            # 'num_routing': 1,
            'dataset': 'MovingMNist2',
            'batch_size': 32,
            'num_gpus': num_gpus,
            'time': time,
            # 'num_workers': 2,
            'epoch': 500,
            'schedule_milestones': '10,30',
            'use_scheduler': True,
            'weight_decay': 0.0005,
            'lr': 0.0003,
            'schedule_gamma': 0.3,
            'no_colored': True,
            'num_mnist_digits': 1,
            'fix_moving_mnist_center': True,
            # 'do_mnist_test_every': 1,
            # 'do_mnist_test_after': -1,
            'do_tsne_test_every': 1,
            'do_tsne_test_after': -1,
            'use_presence_probs': True,
            'lambda_sparse_presence': 0.3,
            'use_cuda_tsne': False,
            'presence_loss_type': 'l2norm',
            'fix_moving_mnist_angle': False,
            'triangle_margin_lambda': 1.0,
            'margin_gamma2': 0.1,
            'triangle_cos_lambda': 1.0,
            'no_use_angle_loss': True,
            'no_use_hinge_floss': True,
            'num_output_classes': 10,
            'lambda_between_entropy': 3.0,
            'lambda_within_entropy': 1.0,
            'presence_diffcos_lambda': 10.0,
            'use_class_sampler': False,
            'no_use_nce_loss': True,
            'use_nce_probs': True,
            'use_simclr_xforms': False,
            'use_simclr_nce': True,
            'step_length': 0.,
            'use_rand_presence_noise': True,
            'nce_presence_temperature': 0.1,
            'use_diff_class_digit': True,
            'nceprobs_selection': 'ncelinear_none',
            'image_size': 40, # 28
            'no_hit_side': True,
            # 'center_discrete': False,
            'discrete_angle': True,
            'nceprobs_selection_temperature': 1.0,
            'simclr_selection_strategy': 'default',
            'ncelinear_anchorselect': '21',
            # 'center_discrete_count': 3,
            'counter': find_counter,
            'seed': 0,
            'mnist_padding': False,
        })

        return find_counter, job
    # Padded MNIST (34x34) NCE with 0.1 presence_temperature
    elif find_counter == 3:
        num_gpus = 4
        time = 24
        job.update({
            'name': '2020.05.16',
            # 'optimizer': 'adam',
            'config': 'resnet_backbone_movingmnist2_20ccgray_img34',
            'criterion': 'nceprobs_selective',
            'triangle_lambda': 1,
            # 'num_routing': 1,
            'dataset': 'MovingMNist2',
            'batch_size': 32,
            'num_gpus': num_gpus,
            'time': time,
            # 'num_workers': 2,
            'epoch': 500,
            'schedule_milestones': '10,30',
            'use_scheduler': True,
            'weight_decay': 0.0005,
            'lr': 0.0003,
            'schedule_gamma': 0.3,
            'no_colored': True,
            'num_mnist_digits': 1,
            'fix_moving_mnist_center': True,
            # 'do_mnist_test_every': 1,
            # 'do_mnist_test_after': -1,
            'do_tsne_test_every': 1,
            'do_tsne_test_after': -1,
            'use_presence_probs': True,
            'lambda_sparse_presence': 0.3,
            'use_cuda_tsne': False,
            'presence_loss_type': 'l2norm',
            'fix_moving_mnist_angle': False,
            'triangle_margin_lambda': 1.0,
            'margin_gamma2': 0.1,
            'triangle_cos_lambda': 1.0,
            'no_use_angle_loss': True,
            'no_use_hinge_loss': True,
            'num_output_classes': 10,
            'lambda_between_entropy': 3.0,
            'lambda_within_entropy': 1.0,
            'presence_diffcos_lambda': 10.0,
            'use_class_sampler': False,
            'no_use_nce_loss': True,
            'use_nce_probs': True,
            'use_simclr_xforms': False,
            'use_simclr_nce': True,
            'step_length': 0.,
            'use_rand_presence_noise': True,
            'nce_presence_temperature': 0.1,
            'use_diff_class_digit': True,
            'nceprobs_selection': 'ncelinear_none',
            'image_size': 34, # 40,
            'no_hit_side': True,
            # 'center_discrete': False,
            'discrete_angle': True,
            'nceprobs_selection_temperature': 1.0,
            'simclr_selection_strategy': 'default',
            'ncelinear_anchorselect': '21',
            # 'center_discrete_count': 3,
            'counter': find_counter,
            'seed': 0,
            'mnist_padding': True,
        })

        return find_counter, job
    # Padded MNIST (34x34) NCE with 0.01 presence_temperature
    elif find_counter == 4:
        num_gpus = 4
        time = 24
        job.update({
            'name': '2020.05.16',
            # 'optimizer': 'adam',
            'config': 'resnet_backbone_movingmnist2_20ccgray_img34',
            'criterion': 'nceprobs_selective',
            'triangle_lambda': 1,
            # 'num_routing': 1,
            'dataset': 'MovingMNist2',
            'batch_size': 32,
            'num_gpus': num_gpus,
            'time': time,
            # 'num_workers': 2,
            'epoch': 500,
            'schedule_milestones': '10,30',
            'use_scheduler': True,
            'weight_decay': 0.0005,
            'lr': 0.0003,
            'schedule_gamma': 0.3,
            'no_colored': True,
            'num_mnist_digits': 1,
            'fix_moving_mnist_center': True,
            # 'do_mnist_test_every': 1,
            # 'do_mnist_test_after': -1,
            'do_tsne_test_every': 1,
            'do_tsne_test_after': -1,
            'use_presence_probs': True,
            'lambda_sparse_presence': 0.3,
            'use_cuda_tsne': False,
            'presence_loss_type': 'l2norm',
            'fix_moving_mnist_angle': False,
            'triangle_margin_lambda': 1.0,
            'margin_gamma2': 0.1,
            'triangle_cos_lambda': 1.0,
            'no_use_angle_loss': True,
            'no_use_hinge_loss': True,
            'num_output_classes': 10,
            'lambda_between_entropy': 3.0,
            'lambda_within_entropy': 1.0,
            'presence_diffcos_lambda': 10.0,
            'use_class_sampler': False,
            'no_use_nce_loss': True,
            'use_nce_probs': True,
            'use_simclr_xforms': False,
            'use_simclr_nce': True,
            'step_length': 0.,
            'use_rand_presence_noise': True,
            'nce_presence_temperature': 0.01,
            'use_diff_class_digit': True,
            'nceprobs_selection': 'ncelinear_none',
            'image_size': 34, # 40,
            'no_hit_side': True,
            # 'center_discrete': False,
            'discrete_angle': True,
            'nceprobs_selection_temperature': 1.0,
            'simclr_selection_strategy': 'default',
            'ncelinear_anchorselect': '21',
            # 'center_discrete_count': 3,
            'counter': find_counter,
            'seed': 0,
            'mnist_padding': True,
        })

        return find_counter, job
    # Padded MNIST (40x40) NCE with 0.01 nce_presence_temperature
    elif find_counter == 5:
        num_gpus = 4
        time = 24
        job.update({
            'name': '2020.05.15',
            # 'optimizer': 'adam',
            'config': 'resnet_backbone_movingmnist2_20ccgray_img40',
            'criterion': 'nceprobs_selective',
            'triangle_lambda': 1,
            # 'num_routing': 1,
            'dataset': 'MovingMNist2',
            'batch_size': 32,
            'num_gpus': num_gpus,
            'time': time,
            # 'num_workers': 2,
            'epoch': 500,
            'schedule_milestones': '10,30',
            'use_scheduler': True,
            'weight_decay': 0.0005,
            'lr': 0.0003,
            'schedule_gamma': 0.3,
            'no_colored': True,
            'num_mnist_digits': 1,
            'fix_moving_mnist_center': True,
            # 'do_mnist_test_every': 1,
            # 'do_mnist_test_after': -1,
            'do_tsne_test_every': 1,
            'do_tsne_test_after': -1,
            'use_presence_probs': True,
            'lambda_sparse_presence': 0.3,
            'use_cuda_tsne': False,
            'presence_loss_type': 'l2norm',
            'fix_moving_mnist_angle': False,
            'triangle_margin_lambda': 1.0,
            'margin_gamma2': 0.1,
            'triangle_cos_lambda': 1.0,
            'no_use_angle_loss': True,
            'no_use_hinge_loss': True,
            'num_output_classes': 10,
            'lambda_between_entropy': 3.0,
            'lambda_within_entropy': 1.0,
            'presence_diffcos_lambda': 10.0,
            'use_class_sampler': False,
            'no_use_nce_loss': True,
            'use_nce_probs': True,
            'use_simclr_xforms': False,
            'use_simclr_nce': True,
            'step_length': 0.,
            'use_rand_presence_noise': True,
            'nce_presence_temperature': 0.01,
            'use_diff_class_digit': True,
            'nceprobs_selection': 'ncelinear_none',
            'image_size': 40, # 28
            'no_hit_side': True,
            # 'center_discrete': False,
            'discrete_angle': True,
            'nceprobs_selection_temperature': 1.0,
            'simclr_selection_strategy': 'default',
            'ncelinear_anchorselect': '21',
            # 'center_discrete_count': 3,
            'counter': find_counter,
            'seed': 0,
            'mnist_padding': True,
        })

        return find_counter, job
    else:
        print('Counter not found')
        return None, None


if __name__ == '__main__':
    run()
