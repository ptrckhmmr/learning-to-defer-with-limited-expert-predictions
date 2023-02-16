import sys, os
import json

from torch.utils.tensorboard import SummaryWriter
from absl import flags
from absl import app

from scripts.utils import get_train_dir, save_to_logs
from scripts.nn_ex_model_lib import NNExpertModel

wkdir = os.getcwd()
sys.path.append(wkdir)

FLAGS = flags.FLAGS


def main(argv):
    args = {
        'seed': FLAGS.seed,
        'ex_seed': FLAGS.ex_seed,
        'emb_model': FLAGS.emb_model,
        'ex_model': FLAGS.ex_model,
        'dataset': FLAGS.dataset,
        'num_classes': FLAGS.num_classes,
        'batch': FLAGS.batch,
        'lr': FLAGS.lr,
        'sched': FLAGS.sched,
        'opt': FLAGS.opt,
        'n_strengths': FLAGS.ex_strength,
        'binary': FLAGS.binary,
        'labels': FLAGS.n_labeled,
    }
    if args['dataset'] == 'nih':
        args['num_classes'] = 2
        args['binary'] = False
    else:
        args['num_classes'] = 20
    # get the training directory
    train_dir = get_train_dir(wkdir, args, 'expert_net')
    # initialize the summary writer for tensorboard
    writer = SummaryWriter(train_dir + 'logs/')
    # initiate the expert model
    ex_model = NNExpertModel(args, wkdir, writer)
    # try to load previous training
    start_epoch = ex_model.load_from_checkpoint(mode='latest')
    # train model
    if args['labels'] < 0:
        max_epochs = int(5000/args['labels'])
    else:
        max_epochs = 150
    for epoch in range(start_epoch, max_epochs):
        # train one epoch
        loss = ex_model.train_one_epoch(epoch)
        # get metrics
        test_acc = ex_model.get_metrics()
        # write data to tensorboard
        ex_model.writer.flush()
        # save logs to json
        save_to_logs(train_dir, test_acc['acc'], loss)
        # save model to checkpoint
        ex_model.save_to_checkpoint(epoch, loss, test_acc['acc'])

    preds = ex_model.predict()
    mode = 'binary' if args['binary'] else 'expert'
    mode2 = 'bin' if args['binary'] else 'mult'
    filename = f'EmbeddingNN_{mode2}_{args["dataset"]}_{mode}{args["n_strengths"]}.' \
               f'{args["seed"]}@{args["labels"]}_predictions.json'
    if not os.path.exists('./artificial_expert_labels/'):
        os.makedirs('./artificial_expert_labels/')
    with open(f'artificial_expert_labels/{filename}', 'w') as f:
        json.dump(preds, f)
    with open(os.getcwd()[
              :-len('Embedding-Supervised')] + f'Human-AI-Systems/artificial_expert_labels/{filename}','w') as f:
        json.dump(preds, f)


if __name__ == '__main__':
    flags.DEFINE_integer('seed', 123, 'Random seed for data splits')
    flags.DEFINE_integer('ex_seed', 123, 'Random seed for expert generation')
    flags.DEFINE_string('emb_model', 'efficientnet_b1', 'Type of base model')
    flags.DEFINE_string('ex_model', 'LinearNN_s', 'Type of expert model')
    flags.DEFINE_string('dataset', 'cifar100', 'Dataset')
    flags.DEFINE_integer('num_classes', 20, 'Number of classes')
    flags.DEFINE_integer('batch', 64, 'Batchsize')
    flags.DEFINE_float('lr', 4e-3, 'Learning-rate')
    flags.DEFINE_string('sched', 'step', 'Learningrate scheduling')
    flags.DEFINE_string('opt', 'sgd', 'Optimization algorithm')
    flags.DEFINE_integer('ex_strength', 60, 'Number of expert strengths')
    flags.DEFINE_boolean('binary', True, 'Flag for binary expert labels')
    flags.DEFINE_integer('n_labeled', 1000, 'Number of labeled images')

    app.run(main)
