import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value

    :ivar val: Value
    :ivar avg: Average
    :ivar sum: Sum
    :ivar count: Count
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average meter

        :param val: Value
        :param n: Number of values
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k

    :param output: Model output
    :param target: Targets
    :param topk: K-value

    :return: precision@k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def metrics_print(net, expert_fn, n_classes, loader, test=False):
    """Computes system metrics

    :param net: Model
    :param expert_fn: Expert labels
    :param n_classes: number of classes
    :param loader: Dataloader

    :return: Dict of metrics
    """
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    total_per_class = {c: 0 for c in range(n_classes)}
    real_total_per_class = {c: 0 for c in range(n_classes)}
    with torch.no_grad():
        for data in loader:
            images, labels, indices = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(indices, test=test)
            #print(f'DEBUG: \nlabels: {labels[:10]} \npredicted: {predicted[:10]} \nexp_predictions: {exp_prediction[:10]}')
            for i in range(0, batch_size):
                r = (predicted[i].item() == n_classes)
                prediction = predicted[i]
                if predicted[i] == n_classes:
                    max_idx = 0
                    # get second max
                    for j in range(0, n_classes):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    total_per_class[labels[i].item()] += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
                real_total_per_class[labels[i].item()] += 1
    cov = total/real_total
    cov_per_class = [total_per_class[c]/real_total_per_class[c] for c in range(n_classes)]
    to_print = {"coverage": cov,
                "cov per class": cov_per_class,
                "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    print(to_print)
    return to_print