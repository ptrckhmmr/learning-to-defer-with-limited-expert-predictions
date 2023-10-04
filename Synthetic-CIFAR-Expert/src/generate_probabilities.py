from efficientnet_pytorch import EfficientNet
import numpy as np
from scipy.special import softmax
import torch
from sklearn.metrics.pairwise import cosine_similarity

import src.preprocess_data as prep

BATCHSIZE = 10
EMB_MODEL = 'efficientnet-b1'


def generate_error_probs():
    """
    Generate the error probabilities for the synthetic experts.
    """
    # load train and test data for cifar100
    train_data, test_data = prep.get_train_test_data()

    # get train and test loader
    train_loader, test_loader, device = prep.get_data_loader(train_data, test_data, batch_size=BATCHSIZE)

    # initiate model
    model = EfficientNet.from_pretrained(EMB_MODEL)
    # load model to device
    model = prep.to_device(model, device)
    # generate feature vectors
    model.eval()
    # initiate array for the feature vectors
    feature_vecs = np.zeros((100, 100, 1000))
    # initiate array for counting the number of feature vectors by class
    class_counter = np.zeros(100, dtype=int)
    for ii, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            # get model output
            output = model(data).cpu().numpy()
        y = target.cpu().numpy()
        for i in range(len(y)):
            # save feature vector to array
            feature_vecs[y[i]][class_counter[y[i]]] = output[i]
            class_counter[y[i]] += 1

    # average over all feature vectors from the same class
    class_feature_vecs = np.mean(feature_vecs, axis=1)

    # initiate array for the class similarities
    class_sims = np.zeros((100, 100), dtype=float)

    # calculate the cosine similarity between all classes
    for c1 in range(100):
        for c2 in range(100):
            a = class_feature_vecs[int(c1)]
            b = class_feature_vecs[int(c2)]
            sim = cosine_similarity([a], [b])
            if c1 == c2:
                # set similarity to 0 between a class and itself
                class_sims[int(c1)][int(c2)] = 0
            else:
                class_sims[int(c1)][int(c2)] = sim[0]

    # save the calculated similarities
    np.save('data/class_sims.npy', class_sims)

    # calculate probabilities
    probs = np.zeros((100, 100))
    for i in range(100):
        # standardize similarities
        sim = (class_sims[i] - np.mean(class_sims[i])) / np.std(class_sims[i])
        # calculate probabilities via the softmax function
        probs[i] = softmax(sim)

    # save the calculated probabilities
    np.save('data/mistake_probs.npy', probs)


if __name__ == '__main__':
    generate_error_probs()