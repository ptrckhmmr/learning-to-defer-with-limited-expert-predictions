import sys, os
import json

from absl import flags
from absl import app

from scripts.svm_ex_model_lib import SVMExpertModel

wkdir = os.getcwd()
sys.path.append(wkdir)

FLAGS = flags.FLAGS


def main(argv):
    args = {
        'seed': FLAGS.seed,
        'ex_seed': FLAGS.ex_seed,
        'dataset': FLAGS.dataset,
        'emb_model': FLAGS.emb_model,
        'num_classes': FLAGS.num_classes,
        'batch': FLAGS.batch,
        'n_strengths': FLAGS.ex_strength,
        'binary': FLAGS.binary,
        'labels': FLAGS.n_labeled,
        'kernel': FLAGS.kernel,
        'C': FLAGS.C,
        'class_weight': FLAGS.class_weight,
    }
    if args['dataset'] == 'nih':
        args['num_classes'] = 2
    else:
        args['num_classes'] = 20
    # initiate the expert model
    ex_model = SVMExpertModel(args, wkdir)
    ex_model.train_svc(args['kernel'], args['C'], args['class_weight'])
    ex_model.get_metrics()
    preds = ex_model.predict()
    mode = 'binary' if args['binary'] else 'expert'
    mode2 = 'bin' if args['binary'] else 'mult'
    if not os.path.exists('./artificial_expert_labels/'):
        os.makedirs('./artificial_expert_labels/')
    filename = f'EmbeddingSVM_{mode2}_{args["dataset"]}_{mode}{args["n_strengths"]}.' \
               f'{args["seed"]}@{args["labels"]}_predictions.json'
    with open(f'artificial_expert_labels/{filename}', 'w') as f:
        json.dump(preds, f)
    with open(os.getcwd()[
              :-len('Embedding-Supervised')] + f'Human-AI-Systems/artificial_expert_labels/{filename}','w') as f:
        json.dump(preds, f)


if __name__ == '__main__':
    flags.DEFINE_integer('seed', 123, 'random seed for data splits')
    flags.DEFINE_integer('ex_seed', 123, 'random seed for the expert')
    flags.DEFINE_string('emb_model', 'resnet18', 'Feature extractor model')
    flags.DEFINE_string('dataset', 'nih', 'Dataset')
    flags.DEFINE_integer('num_classes', 20, 'Number of classes')
    flags.DEFINE_integer('batch', 8, 'Batchsize')
    flags.DEFINE_integer('ex_strength', 4323195249, 'Number of expert strengths')
    flags.DEFINE_boolean('binary', True, 'Flag for binary expert labels')
    flags.DEFINE_integer('n_labeled', 1000, 'Number of labeled images')
    flags.DEFINE_string('kernel', 'rbf', 'Kernel function')
    flags.DEFINE_integer('C', 5, 'Regularization parameter')
    flags.DEFINE_string('class_weight', None, 'Class weights')

    app.run(main)
