"""Run the jobs in this file.

Example running jobs:

python cinjon_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from run_on_cluster import do_jobarray
import os
import socket
hostname = socket.gethostname()
is_cims = hostname.startswith('cassio') or hostname.startswith('lion')
is_dgx = hostname.startswith('dgx-1')
is_prince = hostname.startswith('log-') or hostname.startswith('gpu-')

email = 'cinjon@nyu.edu'
if is_cims or is_dgx:
    code_directory = '/home/resnick/Code/ml-capsules-inverted-attention-routing'
elif is_prince:
    code_directory = '/home/cr2668/Code/ml-capsules-inverted-attention-routing'
else:
    raise


def run(find_counter=None):
    counter = 1

    # NOTE: These jobs are running the triangle loss with MovingMnist2.
    # They are using two capsules, so we may run into issues with the
    # capsules not being ordered correctly.
    job = {
        'name': '2020.04.18',
        'optimizer': 'adam',
        'config': 'resnet_backbone_movingmnist2_2cc',
        'criterion': 'triangle',
        'triangle_lambda': 1,
        'num_routing': 1,
        'dataset': 'MovingMNist2',
        'batch_size': 12,
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results',
        'data_root': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/mnist'
    }

    num_gpus = 1
    time = 8
    var_arrays = {
        'lr': [.0003],
        'triangle_lambda': [1., 3.],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # NOTE: These jobs are running the triangle loss in order to get a sense of
    # if that trains. It's working w Gymnastics but only using one capsule at
    # the end.
    job = {
        'name': '2020.04.18',
        'optimizer': 'adam',
        'config': 'resnet_backbone_gymnastics_1cc',
        'criterion': 'triangle',
        'triangle_lambda': 1,
        'num_routing': 1,
        'dataset': 'gymnasticsRgbFrame',
        'skip_videoframes': 6,
        'dist_videoframes': -19,
        'num_videoframes': 5,
        'batch_size': 4,
        'video_names': '2628',
        'gymnastics_video_directory': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/frames.feb102020',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results',
    }
    num_gpus = 1
    time = 8
    var_arrays = {
        'lr': [.0003],
        'triangle_lambda': [1., 3.],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Here, we are doing it with 6 videonames and a smaller triangle_lambda range.
    job = {
        'name': '2020.04.18',
        'optimizer': 'adam',
        'config': 'resnet_backbone_gymnastics_1cc',
        'criterion': 'triangle',
        'triangle_lambda': 1,
        'num_routing': 1,
        'dataset': 'gymnasticsRgbFrame',
        'skip_videoframes': 6,
        'dist_videoframes': -19,
        'num_videoframes': 5,
        'batch_size': 4,
        'video_names': '2628,2348,2654,2846,2660,2843',
        'gymnastics_video_directory': '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/frames.feb102020',
        'results_dir': '/misc/kcgscratch1/ChoGroup/resnick/vidcaps/results',
    }
    num_gpus = 1
    time = 8
    var_arrays = {
        'lr': [.0003],
        'triangle_lambda': [1.5, 2, 3],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    if is_prince:
        print("SETTING FOR PRINCE")
        job['data_root'] = os.path.join('/beegfs/cr2668/vidcaps', 'MovingMNist')
        job['affnist_root'] = os.path.join('/beegfs/cr2668/vidcaps', 'affnist')
        job['results_dir'] = '/beegfs/cr2668/vidcaps/results'

    # Here, we are shrinking the triangle lambda again and trying a bigger LR,
    # along with a milestone schedule.
    job.update({
        'name': '2020.04.19',
        'epoch': 500,
        'schedule_milestones': '350',
        'use_scheduler': True,
    })
    var_arrays = {
        'lr': [.0003, .001],
        'triangle_lambda': [1.1, 1.2, 1.3, 1.4],
    }
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Here, we are using MovingMNist. Let's see where these end up and then
    # can hone in on which one to use for the affnist test.
    # Counter : 16
    job.update({
        'name': '2020.04.20',
        'epoch': 500,
        'use_scheduler': True,
        'config': 'resnet_backbone_movingmnist2_2cc',
        'dataset': 'MovingMNist2',
        'batch_size': 12
    })
    var_arrays = {
        'lr': [.0003, .001],
        'triangle_lambda': [1, 1.05, 1.1, 1.15, 1.5],
        'schedule_milestones': ['50'],
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Counter: 26.
    # These are with cosine similarity as well.
    job.update({
        'triangle_lambda': 1,
        'criterion': 'triangle_cos',
    })
    var_arrays = {
        'lr': [.0003, .001],
        'schedule_milestones': ['30'],
        'triangle_cos_lambda': [10., 3.]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # These are greater loss lambdas and also yeidling the angle.
    # Counter: 30
    var_arrays = {
        'lr': [.0003, .001],
        'schedule_milestones': ['30', '10,30'],
        'triangle_cos_lambda': [20., 30., 40., 50.]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # These are triangle with margin instead of cos.
    # Counter: 46
    job.update({
        'criterion': 'triangle_margin'
    })
    var_arrays = {
        'lr': [.0003, .001],
        'schedule_milestones': ['30', '10,30'],
        'triangle_margin_lambda': [1., 3.]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Here, we do triangle margin, but we reduce the emphasis on the hinge loss
    # in order to see if that promotes teh angle more.
    # Counter: 54
    var_arrays = {
        'lr': [.0003, .001],
        'schedule_milestones': ['30', '10,30'],
        'triangle_margin_lambda': [.1, .3]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we do triangle margin2, which is putting the margin on ab and bc
    # rather than ac. It keeps the optimization loss on the triangle expr.
    # Counter: 62
    job.update({
        'criterion': 'triangle_margin2'
    })
    var_arrays = {
        'lr': [.0003, .001],
        'schedule_milestones': ['10,30', '30'],
        'triangle_margin_lambda': [.1, .3, 1., 3]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we do hinge loss on ab and bc, but then optimize the angle directly
    # instead of doing the triangle loss. We do that by optimizing -cosine_sim(ab, ac).
    # This forces cosine_sim to become 1, which aligns the two segments.
    # Counter: 78
    job.update({
        'criterion': 'triangle_margin2_angle'
    })
    var_arrays = {
        'lr': [.0003, .001],
        'schedule_milestones': ['10,30', '30'],
        'triangle_margin_lambda': [.1, .3, 1., 3]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # The hinge loss on ab and bc and then optimizing the angle directly didnt
    # work very well. The model ended up focusing on the hinges a lot more and
    # didn't really push the angle enough. This could be becuase the gamma on
    # the hinge was too high. We see that the loss on the hinge loss got really
    # low, so the model certainly emphasize that even when it had a 0.1 lambda
    # on that loss.
    # Here, we try this with smaller margin_gamma2 (0.05, 0.1), plus a range of
    # learning rate gammas of [0.3, 0.5] (instead of just 0.1).
    # Counter: 94
    job.update({
        'criterion': 'triangle_margin2_angle'
    })
    var_arrays = {
        'lr': [.001],
        'schedule_milestones': ['10,30', '30'],
        'triangle_margin_lambda': [.1, .3, 1.],
        'margin_gamma2': [0.05, 0.1],
        'schedule_gamma': [0.3, 0.5]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # triangle margin2, which is putting the margin on ab and bc rather than ac.
    # We try the 0.1, 0.3 triangle_margin_lambdas with schedules of 10,30 and
    # gammas of 0.3, 0.5.
    # Counter: 118
    job.update({
        'criterion': 'triangle_margin2'
    })
    var_arrays = {
        'lr': [.001],
        'schedule_milestones': ['10,30'],
        'triangle_margin_lambda': [.1, .3],
        'schedule_gamma': [0.3, 0.5],
        'margin_gamma2': [0.5, 0.1]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we do triangle margin, but with reduced hinge … more like
    # 0.05, 0.1, and 0.3 rather than 1. The problem is that shit is overfitting.
    # Counter: 126
    # NOTE: The 0.3 ones didn't get off.
    job.update({
        'criterion': 'triangle_margin'
    })
    var_arrays = {
        'margin_gamma': [0.05, 0.1, 0.3],
        'lr': [.001],
        'schedule_milestones': ['10,30'],
        'triangle_margin_lambda': [.1, .3]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # hinge_loss on ac and optimize the angle, with weight decay.
    # Counter: 132
    job.update({
        'name': '2020.04.23',
        'criterion': 'triangle_margin2_angle',
        'weight_decay': 5e-4,
        'schedule_milestones': '10,30',
        'lr': 1e-3
    })
    var_arrays = {
        'triangle_margin_lambda': [.1, .3],
        'margin_gamma2': [0.1, 0.3],
        'schedule_gamma': [0.3]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Counter: 136
    # triangle margin2: hinge loss on ab and bc rather than ac, with optimizing
    # the expression. ... And weight decay.
    job.update({
        'criterion': 'triangle_margin2'
    })
    var_arrays = {
        'triangle_margin_lambda': [.1, .3],
        'margin_gamma2': [0.1, 0.3],
        'schedule_gamma': [0.3],
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # triangle margin with weight decay
    # NOTE: The 0.3 ones didn't get off before and the .05 were similar to 0.1,
    # so we redo the 0.3 ones here.
    # Counter: 140
    job.update({
        'criterion': 'triangle_margin'
    })
    var_arrays = {
        'margin_gamma': [0.1, 0.3],
        'triangle_margin_lambda': [.1, .3],
        'schedule_gamma': [0.3],
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # NOTE: We fucked up before and weren't doing the right thing for these jobs.
    # So now we are fixing that and redoing some of them again to see if they
    # were at all on the right track.
    # NOTE: The problem was that they were using is_discriminating_model's
    # output, so they were going through a final_fc pose to the number of classes.
    # This might actually be a good idea, but it sucks for the exact thing we
    # were trying to accomplish.
    # Counter: 144
    job.update({
        'name': '2020.04.25',
        'config': 'resnet_backbone_movingmnist2_2cc',
        'schedule_milestones': '10,30',
        'lr': 1e-3,
        'schedule_gamma': 0.3,
    })
    var_arrays = {
        'criterion': ['triangle_margin2', 'triangle_margin2_angle', 'triangle_margin'],
        'triangle_margin_lambda': [.1, .3],
        'margin_gamma2': [0.1, 0.3],
        'weight_decay': [0, 5e-4]
    }
    time = 8
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # NOTE: Same as above jobs, except now we are doing it on single digit
    # MovingMNist2, along with grayscale (so one channel), and using the 1cc
    # config, which has only one capsule at the end. This might not work.
    # We are also putting all the criterions into one:
    # triangle_margin2_angle: hinge_loss on ac and optimize the angle.
    # triangle_margin2: hinge loss on ab and bc, plus optimize the triagnle expr.
    # triangle_margin: hinge loss on ac, plus optimize the triagnle expr.
    # Counter: 144
    job.update({
        'name': '2020.04.25',
        'config': 'resnet_backbone_movingmnist2_2ccgray',
        'schedule_milestones': '10,30',
        'lr': 1e-3,
        'schedule_gamma': 0.3,
        'no_colored': True,
        'num_mnist_digits': 1,
    })
    var_arrays = {
        'criterion': ['triangle_margin2', 'triangle_margin2_angle', 'triangle_margin'],
        'triangle_margin_lambda': [.1, .3],
        'margin_gamma2': [0.1, 0.3],
        'weight_decay': [0, 5e-4]
    }
    time = 8
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Same as above, but using mroe capsules in output.
    # Counter: 192
    job.update({
        'name': '2020.04.26',
        'config': 'resnet_backbone_movingmnist2_10ccgray',
    })
    var_arrays = {
        'criterion': ['triangle_margin2', 'triangle_margin2_angle', 'triangle_margin'],
        'triangle_margin_lambda': [.1, .3],
        'margin_gamma2': [0.1, 0.3],
        'weight_decay': [0]
    }
    time = 6
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Adding in more capsules worked well. Let's try the same, but push down
    # the margin_gamm2 even more and add in weight decay. We can prolly try
    # these models after doing this.
    job.update({
        'name': '2020.04.27',
    })
    var_arrays = {
        'criterion': ['triangle_margin2', 'triangle_margin2_angle'],
        'triangle_margin_lambda': [.1, .3, 1.],
        'margin_gamma2': [0.01, 0.03, .1],
        'weight_decay': [0]
    }
    time = 6
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Same thing as above but with triangle_margin
    job.update({
        'name': '2020.04.27',
    })
    var_arrays = {
        'criterion': ['triangle_margin'],
        'triangle_margin_lambda': [.1, .3, 1.],
        'margin_gamma': [0.01, 0.03, .1],
        'weight_decay': [0]
    }
    time = 6
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Annnnd nope. We gotta change the dataset to do fixed_moving_mnist.
    # Otherwise, it's a total shit show with how it's linearized to begin with.
    # It's just so much harder that way.
    # We are no longer doing `triangle_margin` because it's not as sound as
    # triangle_margin2.
    # Counter: 231
    job.update({
        'name': '2020.04.29',
        'fix_moving_mnist_center': True,
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
    })
    var_arrays = {
        'criterion': ['triangle_margin2', 'triangle_margin2_angle'],
        'triangle_margin_lambda': [.3],
        'margin_gamma2': [0.3, .1],
        'weight_decay': [0, 5e-4]
    }
    time = 6
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here we are doing discriminative MovingMNist1img.
    # Counter: 239
    job.update({
        'name': '2020.04.29',
        'criterion': 'xent',
        'dataset': 'MovingMNist2.img1',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
    })
    var_arrays = {
        'fix_moving_mnist_center': [False, True],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here we are doing MovingMNist with optimizing angle, margin, and nce.
    # Counter: 243
    job.update({
        'name': '2020.04.29',
        'criterion': 'triangle_margin2_angle_nce',
        'dataset': 'MovingMNist2',
        'do_tsne_test_every': 5,
        'do_tsne_test_after': -1,
    })
    var_arrays = {
        'fix_moving_mnist_center': [False, True],
        'triangle_margin_lambda': [.3, 1.],
        'margin_gamma2': [.1, .3],
        'triangle_cos_lambda': [1., 3.]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Seeing what's up re 1 capsule.
    # Counter: 257
    job.update({
        'name': '2020.04.30',
        'criterion': 'triangle_margin2_angle_nce',
        'dataset': 'MovingMNist2',
        'do_tsne_test_every': 4,
        'config': 'resnet_backbone_movingmnist2_gray1cc',
    })
    var_arrays = {
        'fix_moving_mnist_center': [False, True],
        'triangle_margin_lambda': [.3, 1.],
        'margin_gamma2': [.1, .3],
        'triangle_cos_lambda': [1., 3.]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Try to presence probs with a large margin_lambda and a small sparse_lambda
    # Counter: 273
    job.update({
        'name': '2020.04.30',
        'criterion': 'triangle_margin2_angle_nce',
        'dataset': 'MovingMNist2',
        'config': 'resnet_backbone_movingmnist2_10ccgray',
        'use_presence_probs': True,
        'lambda_sparse_presence': 0.3,
        'use_cuda_tsne': True,
        'do_tsne_test_every': 2,
        'presence_loss_type': 'sigmoid_l1'
    })
    var_arrays = {
        'fix_moving_mnist_center': [True, False],
        'triangle_margin_lambda': [1., 10.],
        'margin_gamma2': [.1, .3],
        'triangle_cos_lambda': [1., 3.],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Try to presence probs with a large margin_lambda and a small sparse_lambda
    # Counter:  289
    job.update({
        'name': '2020.05.01',
        'criterion': 'triangle_margin2_angle_nce',
        'dataset': 'MovingMNist2',
        'config': 'resnet_backbone_movingmnist2_10ccgray',
        'use_presence_probs': True,
        'lambda_sparse_presence': 0.3, # 0.3 doesn't learn
        'use_cuda_tsne': True,
        'do_tsne_test_every': 2,
        'presence_loss_type': 'sigmoid_prior_sparsity'
    })
    var_arrays = {
        'fix_moving_mnist_center': [True, False],
        'triangle_margin_lambda': [1., 3.],
        'margin_gamma2': [.1, .3],
        'triangle_cos_lambda': [1., 3.],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Try to presence probs with a large margin_lambda and a small sparse_lambda
    # Counter: 305
    job.update({
        'name': '2020.05.03',
        'use_cuda_tsne': True,
        'do_tsne_test_every': 1,
        'fix_moving_mnist_center': True,
        'fix_moving_mnist_angle': True,
        'triangle_margin_lambda': 10.,
        'margin_gamma2': 0.3,
        'triangle_cos_lambda': 1.,
        # 'no_use_hinge_loss': True,
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_prior_sparsity_example', 'sigmoid_prior_sparsity',
            'sigmoid_within_entropy', 'sigmoid_within_between_entropy',
            'softmax', 'softmax_nonoise'
        ]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Counter: 311
    job.update({
        'name': '2020.05.03',
        'triangle_margin_lambda': 1.,
        'margin_gamma2': 0.3,
        'triangle_cos_lambda': 1.,
        'lr': 1e-4
    })
    var_arrays = {        
        'presence_loss_type': [
            'softmax_within_between_entropy',
            'sigmoid_prior_sparsity_example'
        ]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we try it with just nce and presence. With that, we can increase
    # the batch size. ... 
    # Counter: 313
    job.update({
        'name': '2020.05.04',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'batch_size': 24,
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_prior_sparsity',
            'sigmoid_prior_sparsity_example_between_entropy'
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # The above failed hard because it just made everything 0. We bring back
    # the hinge loss.
    # Counter: 315
    job.update({
        'name': '2020.05.04',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'no_use_angle_loss': True,
        'no_use_hinge_loss': False,
        'num_output_classes': 10,
        'batch_size': 24, # -_- ... this whole time I could have been using bigger BS
    })
    # When it collapses below, the result is that everything goes to zero vector
    # and the gradient becomes 0.0 as well. Seems like the common factor in
    # that happening is the capsule_presence_loss, which is what pushes the
    # model to spread weight amongst the capsules. The between entropy is also
    # supposed to do that though.
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_prior_sparsity', # <-- Collapsed with 0 within and between entropy. It's really hard.
            'sigmoid_prior_sparsity_example_between_entropy',
            'sigmoid_prior_sparsity_between_entropy', # <-- Also collapsed.
            'sigmoid_within_between_entropy'
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # On the dgx ... try upping the lambda between entry and the lr.
    # When we upped the LR to 3e-4, it collapsed, err, but we weren't actually
    # using num_gpus=4.
    # NOTE: Wth the lambda_between_entropy at 3.0, the model gets stuck trying
    # to reduce nce all the way to 0 but with every capsule prob stuck at 1.0
    # Counter: 319
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': False,
        'num_output_classes': 10,
        'lambda_between_entropy': 3.0,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': 3,
        'batch_size': 30 # On the dgx.
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_within_between_entropy',
        ],
    }
    num_gpus = 4
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we are trying the squashed approach to getting the logits.
    # We turn off everything but the nce.
    # Counter: 320
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lambda_between_entropy': 3.0,
        'lambda_within_entropy': 100.0, 
        'lr': 3e-4,
        'num_workers': 4,
        'num_gpus': 1,
        'batch_size': 24, # on the regular.
    })
    var_arrays = {        
        'presence_loss_type': [
            'squash_prior_sparsity', 'squash_within_between_entropy'
        ],
    }
    num_gpus = 1
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Same re using squash to get at the logits. Using dgx with higher LR.
    # We turn off everything but the nce.
    # Counter: 322
    num_gpus = 3
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lr': 3e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 24,
    })
    var_arrays = {        
        'presence_loss_type': [
            'squash_example_between',
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # What if we didn't multiply by the presence? We can do that in the squash
    # version because the probabilities are a function of the pose as well.
    # We are using the pose to inform the probabilities, so that's getting
    # backpropped there w/o actually doing this separate multiplication.
    # We turn off everything but the nce.
    # Counter: 323
    num_gpus = 1
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lambda_between_entropy': 3.0,
        'lr': 3e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 24, # on the regular.
    })
    var_arrays = {        
        'presence_loss_type': [
            'squash_prior_sparsity_nomul', # <-- went to .05 for everything.
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Same re using squash to get at the logits. Using dgx with higher LR.
    # Counter: 324
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lr': 3e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 24,
        'lambda_within_entropy': 100.0,
    })
    var_arrays = {        
        'presence_loss_type': [
            'squash_within_between_entropy_nomul',
            'squash_example_between_nomul' # <-- .05 for everything. we know this by the wihthin entropy.
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we use the cossim thing.
    # Counter: 326
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 24,
        'lambda_within_entropy': 1.0,
        'presence_diffcos_lambda': 10.0,
    })
    var_arrays = {        
        'presence_loss_type': [
            'squash_cossim',
            'sigmoid_cossim',
            'squash_cossim_nomul',
            'sigmoid_cossim_within_entropy',
            'sigmoid_within_entropy',
            'sigmoid_l1',
            'sigmoid_l1_between',
            'sigmoid_only',
            'sigmoid_prior_sparsity_fix',
            'sigmoid_prior_sparsity_between_entropy_fix',
            'squash_prior_sparsity_fix',
            'squash_prior_sparsity_nomul_fix',
            'sigmoid_prior_sparsity_within_between_entropy_fix', # <- 338
            # 'sigmoid_hinge_presence',
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we use the cass sampler.
    # Counter: 339
    num_gpus = 1
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 24,
        'use_class_sampler': True
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_prior_sparsity_fix',
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we try between again because that was fucked up before.
    # Note though that it's the within that needs help it appears :/.
    # Counter: 340
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_l1_between',
            'sigmoid_prior_sparsity_between_entropy_fix', 
            'sigmoid_prior_sparsity_within_between_entropy_fix',
            # dafuk is this below: https://www.comet.ml/cinjon/capsules/f3b85483692d48299040f6618c1093b0.
            # ok, it just has more that are on.
            'sigmoid_within_between_entropy', 
            'squash_prior_sparsity_within_entropy_fix',
            'squash_prior_sparsity_within_entropy_nomul_fix',
            'sigmoid_prior_sparsity_fix_nospike'
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Use nce_probs and nothing else. Lol, nothing to teach the rest of it ^_^
    # Counter: 347
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': True,
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.05',
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_only'
        ],
        'config': [
            'resnet_backbone_movingmnist2_20ccgray',
            'resnet_backbone_movingmnist2_30ccgray',
        ]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Use nce_probs and nothing else. Lol, nothing to teach the rest of it ^_^
    # Counter: 349
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': False,
        'criterion': 'discriminative_probs',
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.06',
        'presence_loss_type': 'sigmoid_only'
    })
    var_arrays = {        
        'config': [
            'resnet_backbone_movingmnist2_10ccgray',
            'resnet_backbone_movingmnist2_20ccgray'
        ]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Use nce_probs and nothing else. Lol, nothing to teach the rest of it ^_^
    # Counter: 351
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': False,
        'use_nce_probs': False,
        'criterion': 'triangle_margin2_angle_nce',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.06',
        'presence_loss_type': 'sigmoid_only',
        'use_simclr_xforms': True
    })
    var_arrays = {        
        'config': [
            'resnet_backbone_movingmnist2_20ccgray',
            'resnet_backbone_movingmnist2_30ccgray'
        ],
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Use nce_probs and nothing else. Lol, nothing to teach the rest of it ^_^
    # Counter: 353
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': False,
        'use_nce_probs': False,
        'criterion': 'triangle_margin2_angle_nce',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.06',
        'use_simclr_xforms': True
    })
    var_arrays = {        
        'presence_loss_type': [
            'sigmoid_prior_sparsity_fix',
            'sigmoid_prior_sparsity_within_between_entropy_fix',
            'sigmoid_within_between_entropy',
            'sigmoid_within_entropy',
        ]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Do the probs_test with the updated nce silly billy.
    # Counter:  357
    num_gpus = 2
    job.update({
        'use_simclr_xforms': False
    })
    var_arrays = {
        'step_length': [0.035, 0.1]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Do the probs_test with the simclr repo approach.
    # Counter:  359
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': False,
        'criterion': 'probs_test',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-3,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.07',
        'use_simclr_xforms': False,
        'use_simclr_nce': True
    })
    var_arrays = {
        'step_length': [0.035, 0.1],
        'nce_presence_temperature': [1., 0.5]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Do the use_nce_probs with the simclr approach.
    # Counter:  363
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': True,
        'criterion': 'triangle_margin2_angle_nce',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.07',
        'use_simclr_xforms': False,
        'use_simclr_nce': True,
        'presence_loss_type': 'sigmoid_only',
    })
    var_arrays = {
        'step_length': [0.035, 0.1],
        'use_diff_class_digit': [False, True],
        'nce_presence_temperature': [1., 0.5]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Do the probs_test with the simclr repo approach but even lower temp
    # Counter:  371
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': False,
        'criterion': 'probs_test',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-3,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.08',
        'use_simclr_xforms': False,
        'use_simclr_nce': True,
        'step_length': 0.035,
        'use_rand_presence_noise': True
    })
    var_arrays = {
        'nce_presence_temperature': [0.4, 0.3, 0.2, 0.1],
    }
    time = 3
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we keep the nce_presence_temperature at 1.0 but change hte probs
    # test temperature to be higher. We do this w/ and w/o the rand noise
    # Counter:  375
    num_gpus = 2
    job.update({
        'nce_presence_temperature': 1.0
    })
    var_arrays = {
        'presence_temperature': [3.0, 10.0],
        'use_rand_presence_noise': [True, False]
    }
    time = 3
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here, we keep the nce_presence_temperature at 1.0 but change hte probs
    # test temperature to be higher. We do this w/ and w/o the rand noise
    # Counter:  379
    num_gpus = 2
    job.update({
        'nce_presence_temperature': 1.0
    })
    var_arrays = {
        'presence_temperature': [3.0, 10.0],
        'use_rand_presence_noise': [True, False]
    }
    time = 3
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Do the probs NCE with the temp changes plus also add in angle loss. Note
    # that we have to lower the batch size to accommodate because there are 3
    # images now.
    # NOTE: FUck I'm dumb. This is NOT doing use_diff_class_digits.
    # It might actually work if we did the right thing ....
    # May 10th --> Changing it to be the right thing. (also cominbing it with below)
    # Counter:  383
    print(counter)
    num_gpus = 2
    job.update({
        'no_use_angle_loss': False,
        'use_nce_probs': True,
        'use_presence_probs': True,
        'criterion': 'triangle_margin2_angle_nce',
        'name': '2020.05.09',
        'use_rand_presence_noise': True,
        'use_diff_class_digit': True,
        'presence_loss_type': 'sigmoid_only',
        'batch_size': 20,
    })
    var_arrays = {
        'nce_presence_temperature': [0.1],
        'presence_temperature': [1.0, 10.0],
        'no_use_hinge_loss': [True, False]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Do the probs NCE with the temp changes plus also add in angle loss AND
    # hinge loss. Note that we hd to lower the batsh size again.
    # Counter:  387
    print(counter)
    num_gpus = 2
    job.update({
        'no_use_angle_loss': False,
        'no_use_hinge_loss': False,
        'use_nce_probs': True,
        'use_presence_probs': True,
        'criterion': 'triangle_margin2_angle_nce',
        'name': '2020.05.09',
        'use_rand_presence_noise': True,
        'use_diff_class_digit': True,
        'presence_loss_type': 'sigmoid_only',
        'batch_size': 32, # 32 on dgx... TODO: we haven't done 387!
        'margin_gamma2': 0.1,
    })
    var_arrays = {
        'nce_presence_temperature': [0.5, 0.1],
        'presence_temperature': [1.0, 10.0]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Ugh, I never actually lowered the temp below 0.5 to try that out and get the
    # value more spread. 
    # Counter:  391
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': True,
        'criterion': 'triangle_margin2_angle_nce',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.10',
        'use_simclr_xforms': False,
        'use_simclr_nce': True,
        'presence_loss_type': 'sigmoid_only',
        'use_diff_class_digit': True
    })
    var_arrays = {
        'step_length': [0.035, 0.1],
        'nce_presence_temperature': [0.3, 0.1]
    }
    time = 5
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Temp = 0.1 and step_length0.035 worked really well. Can we relax the
    # fixed properties now?
    # Counter:  395
    num_gpus = 2
    job.update({
        'no_use_angle_loss': True,
        'no_use_hinge_loss': True,
        'no_use_nce_loss': True,
        'use_nce_probs': True,
        'criterion': 'triangle_margin2_angle_nce',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_output_classes': 10,
        'lr': 1e-4,
        'num_workers': 4,
        'num_gpus': num_gpus,
        'batch_size': 32,
        'lambda_within_entropy': 1.0,
        'use_class_sampler': False,
        'name': '2020.05.10',
        'use_simclr_xforms': False,
        'use_simclr_nce': True,
        'presence_loss_type': 'sigmoid_only',
        'use_diff_class_digit': True,
        'step_length': 0.035,
        'nce_presence_temperature': 0.1,
    })
    # NOTE When both of these are false, it works. When angle is false and
    # center is true, it works. But when angle is true and center is false,
    # it ... does not? Ok, but that's what I care about anyways, so fuck it lulz.
    var_arrays = {
        'fix_moving_mnist_angle': [False, True],
        'fix_moving_mnist_center': [False, True]
    }
    time = 5
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    # Doing ncelinear_maxfirst.
    # Counter:  399
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'lr': 1e-4,
        'num_gpus': num_gpus,
        'batch_size': 32 if is_dgx else 20,
        'name': '2020.05.10',
        'nceprobs_selection': 'ncelinear_maxfirst'
    })
    var_arrays = {
        'fix_moving_mnist_angle': [False, True],
        'nceprobs_selection_temperature': [1., 0.5]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Doing ncelinear_maxfirst ... with 40x40 and removing the fixed_center
    # Counter:  403
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'lr': 1e-4,
        'num_gpus': num_gpus,
        'batch_size': 80 if is_dgx else 32,
        'name': '2020.05.11',
        'config': 'resnet_backbone_movingmnist2_20ccgray_img40',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'step_length': 0.035,
        'image_size': 40,
        'no_hit_side': True,
    })
    var_arrays = {
        'nceprobs_selection_temperature': [0.8, 1., 1.2],
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # We want to relax the fix_moving_mnist_center so that this can work with
    # AffNist. It is having trouble w that getting over 45% because it's shifted.
    # Counter: 406
    num_gpus = 2
    job.update({
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'batch_size': 32 if is_dgx else 20,
        'nceprobs_selection': 'ncelinear_maxfirst',
        'image_size': 64,
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True
    })
    var_arrays = {
        'nceprobs_selection_temperature': [0.8, 1.],
        'step_length': [0.035, 0.05]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # The issue with the aboe was that:
    # A) there were too mahy directions and
    # B) the tsne was over random image positions, which is much harder to
    #    assess. While yes, it should be something that is distinguishable by
    #    class, that it wasn't isn't helpful for us.
    # Otherwise ... Doing ncelinear_maxfirst ... with 40x40 and removing the fixed_center
    # Counter:  410
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'lr': 1e-4,
        'num_gpus': num_gpus,
        'batch_size': 80 if is_dgx else 32,
        'name': '2020.05.11',
        'config': 'resnet_backbone_movingmnist2_20ccgray_img40',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'center_discrete': True,
        'discrete_angle': True,
        'step_length': 0.035,
        'image_size': 40,
        'no_hit_side': True,
    })
    var_arrays = {
        'nceprobs_selection_temperature': [0.8, 1., 1.2],
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Same as above re fixing the center_discrete and discrete_angle.
    # Counter: 413
    num_gpus = 2
    job.update({
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'batch_size': 32 if is_dgx else 20,
        'nceprobs_selection': 'ncelinear_maxfirst',
        'image_size': 64,
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': True,
        'discrete_angle': True,
    })
    var_arrays = {
        'nceprobs_selection_temperature': [0.8, 1.],
        'step_length': [0.035, 0.05]
    }
    time = 10
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Going back to the ones that worked (i.e. 399) and trying to see
    # if we can push up the discrete number of positions we use from the center.
    # Counter:  417
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'lr': 1e-4,
        'num_gpus': num_gpus,
        'batch_size': 32 if is_dgx else 20,
        'name': '2020.05.12',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': True,
        'discrete_angle': True,
        'nceprobs_selection_temperature': 1.,
        'step_length': 0.035
    })
    var_arrays = {
        'center_discrete_count': [1, 2, 3, 4]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Same as the above but using bigger step_length.
    # Counter:  421
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'lr': 1e-4,
        'num_gpus': num_gpus,
        'batch_size': 32 if is_dgx else 20,
        'name': '2020.05.12',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': True,
        'discrete_angle': True,
        'nceprobs_selection_temperature': 1.,
        'step_length': 0.07
    })
    var_arrays = {
        'center_discrete_count': [1, 2, 3, 4]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Fix the center; Float the angle; Have the NCE probs go over anchor=0, other={1,2} evenly;
    # NOTE: ALSO! We are changing the anchor in selection to be over [2, 1] rather than [2, 0]. I think should fix that it's building weird size diffs into the frames.
    # Use a bigger step_size to do this.
    # Counter:  425
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'num_gpus': num_gpus,
        'batch_size': 32 if is_dgx else 20,
        'name': '2020.05.12',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': True,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': False,
        'discrete_angle': False,
        'nceprobs_selection_temperature': 1.,
        'simclr_selection_strategy': 'anchor0_other12',
        'ncelinear_anchorselect': '21'
    })
    var_arrays = {
        'lr': [1e-4, 3e-4],
        'step_length': [.035, .07, .09]
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Doing discrete_count=3 with step_length=0.9 and lr=3e-4.
    # Also trying both simclr selection strategies.
    # Counter:  431
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'lr': 3e-4,
        'num_gpus': num_gpus,
        'batch_size': 32 if is_dgx else 20,
        'name': '2020.05.13',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': True,
        'discrete_angle': True,
        'nceprobs_selection_temperature': 1.,
        'step_length': 0.09,
        'ncelinear_anchorselect': '21',
        'center_discrete_count': 3,
    })
    var_arrays = {
        'simclr_selection_strategy': ['default', 'anchor0_other12']
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Here seeing what happens if we use the l2norm approach.
    # Also trying both simclr selection strategies.
    # Counter:  433
    # NOTE / TODO: We didn't do 433 yet because DGX was taken.
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'lr': 3e-4,
        'num_gpus': num_gpus,
        'batch_size': 32 if is_dgx else 22,
        'name': '2020.05.13',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': True,
        'discrete_angle': True,
        'nceprobs_selection_temperature': 1.,
        'step_length': 0.09,
        'ncelinear_anchorselect': '21',
        'center_discrete_count': 3,
        'presence_loss_type': 'l2norm',
    })
    var_arrays = {
        'simclr_selection_strategy': ['default', 'anchor0_other12']
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job


    # Try doing the .07 with center_discrete_count=4 and with bigger batch size
    # in the V100. Note that we also have lr=3e-4.
    # We are using .07 here because that worked for center_discrete_count=3, i.e
    # this is pretty much the same as the conditions in 423 except for the cdc=4
    # and the batchsize higher (and the lr higher fine...)
    # Counter:  435
    print(counter)
    num_gpus = 2
    job.update({
        'criterion': 'nceprobs_selective',
        'config': 'resnet_backbone_movingmnist2_20ccgray',
        'lr': 3e-4,
        'num_gpus': num_gpus,
        'batch_size': 40,
        'name': '2020.05.13',
        'nceprobs_selection': 'ncelinear_maxfirst',
        'fix_moving_mnist_angle': False,
        'fix_moving_mnist_center': False,
        'nce_presence_temperature': 0.1,
        'no_hit_side': True,
        'center_discrete': True,
        'discrete_angle': True,
        'nceprobs_selection_temperature': 1.,
        'step_length': 0.07,
        'ncelinear_anchorselect': '21',
        'center_discrete_count': 4,
        'presence_loss_type': 'sigmoid_only',
    })
    var_arrays = {
        'simclr_selection_strategy': ['default']
    }
    time = 12
    counter, _job = do_jobarray(
        email, code_directory, num_gpus, counter, job, var_arrays, time,
        find_counter=find_counter, do_job=False)
    if find_counter and _job:
        return counter, _job

    return None, None

            
                                
if __name__ == '__main__':
    run()
