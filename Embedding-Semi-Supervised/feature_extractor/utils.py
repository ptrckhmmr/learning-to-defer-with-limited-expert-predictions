import json
import os


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """Call in a loop to create terminal progress bar

    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: positive number of decimals in percent complete (Int)
    :param length: character length of bar (Int)
    :param fill: bar fill character (Str)
    :param printEnd: end character (e.g. "\r", "\r\n") (Str)
    """
    total = max(total, 1)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // (total+1e-5))
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def concat_args(args, mode):
    """Concatenate args to string

    :param args: Args
    :param mode: Mode
    :return: String
    """
    args_string = mode + '@'
    for key in args:
        if key != 'batch' and not (mode == 'emb_net' and key == 'lr'):
            args_string += str(key) + '-' + str(args[key]) + '-'
    return args_string[:-1]


def save_to_logs(train_dir, acc, loss):
    """Save accuracy and loss to training log

    :param train_dir: Training directory
    :param acc: Accuracy
    :param loss: Loss
    :return:
    """
    try:
        with open(train_dir + 'logs/exp_log.json', 'r') as f:
            log = json.load(f)
    except:
        log = {'valid_acc': [], 'loss': []}

    log['valid_acc'].append(acc)
    log['loss'].append(loss)

    with open(train_dir + 'logs/exp_log.json', 'w') as f:
        json.dump(log, f)


def get_train_dir(wkdir, args, mode):
    """Get or create training directory

    :param wkdir: Working directory
    :param mode: Mode
    :param args: Args
    """
    path = f'{wkdir}/{args["dataset"].upper()}/{concat_args(args, mode)}/'
    try:
        os.mkdir(path)
    except:
        pass
    try:
        os.mkdir(path+'args')
        os.mkdir(path+'checkpoints')
    except:
        pass
    return path
