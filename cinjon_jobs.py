"""Run the jobs in this file.

Example running jobs:

python cinjon_jobs.py

When you want to add more jobs just put them below and MAKE SURE that all of the
do_jobs for the ones above are False.
"""
from run_on_cluster import do_jobarray

email = 'cinjon@nyu.edu'
code_directory = '/home/resnick/Code/ml-capsules-inverted-attention-routing'


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
    print(counter)
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
        find_counter=find_counter, do_job=True)
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
        find_counter=find_counter, do_job=True)
    if find_counter and _job:
        return counter, _job


    return None, None
            
                                
if __name__ == '__main__':
    run()
