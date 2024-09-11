"""Training utils."""
import datetime
import os
import numpy as np
# LOG_DIR = './logs_no_error'
# LOG_DIR = './logs_no_euluc'
LOG_DIR = './logs_test'
# LOG_DIR = './logs_test2'
# LOG_DIR = './logs_sem_sp'
# LOG_DIR = './logs_normal'
# LOG_DIR = './logs_no_error'
def get_savedir(model, dataset, experiment, make_new=True):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        LOG_DIR, model, dataset,
        'experiment_{}'.format(experiment)
    )
    if make_new:
        os.makedirs(save_dir)
    return save_dir

def avg_both(mrs, mrrs, hits):
    """Aggregate metrics for missing lhs and rhs.

    Args:
        mrs: Dict[str, float]
        mrrs: Dict[str, float]
        hits: Dict[str, torch.FloatTensor]

    Returns:
        Dict[str, torch.FloatTensor] mapping metric name to averaged score
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h}

def avg_metrics(metrics_list):
    mrr = np.zeros(len(metrics_list))
    hit1 = np.zeros(len(metrics_list))
    hit3 = np.zeros(len(metrics_list))
    hit10 = np.zeros(len(metrics_list))
    for i in range(len(metrics_list)):
        mrr[i] = metrics_list[i]['MRR']
        hit1[i] = metrics_list[i]['hits@[1,3,10]'][0]
        hit3[i] = metrics_list[i]['hits@[1,3,10]'][1]
        hit10[i] = metrics_list[i]['hits@[1,3,10]'][2]
        
    mrr_mean = np.mean(mrr)
    mrr_std = np.std(mrr)
    hit1_mean = np.mean(hit1)
    hit1_std = np.std(hit1)
    hit3_mean = np.mean(hit3)
    hit3_std = np.std(hit3)
    hit10_mean = np.mean(hit10)
    hit10_std = np.std(hit10)
    
    return {'MRR_mean': mrr_mean, 'MRR_std': mrr_std, 'H1_mean': hit1_mean, 'H1_std': hit1_std, 'H3_mean': hit3_mean, 'H3_std': hit3_std, 'H10_mean': hit10_mean, 'H10_std': hit10_std}


def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = "\t {} MR: {:.2f} | ".format(split, metrics['MR'])
    result += "MRR: {:.3f} | ".format(metrics['MRR'])
    result += "H@1: {:.3f} | ".format(metrics['hits@[1,3,10]'][0])
    result += "H@3: {:.3f} | ".format(metrics['hits@[1,3,10]'][1])
    result += "H@10: {:.3f}".format(metrics['hits@[1,3,10]'][2])
    return result


def write_metrics(writer, step, metrics, split):
    """Write metrics to tensorboard logs."""
    writer.add_scalar('{}_MR'.format(split), metrics['MR'], global_step=step)
    writer.add_scalar('{}_MRR'.format(split), metrics['MRR'], global_step=step)
    writer.add_scalar('{}_H1'.format(split), metrics['hits@[1,3,10]'][0], global_step=step)
    writer.add_scalar('{}_H3'.format(split), metrics['hits@[1,3,10]'][1], global_step=step)
    writer.add_scalar('{}_H10'.format(split), metrics['hits@[1,3,10]'][2], global_step=step)


def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total
