import torch

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def test(R, D, clf, test_dataloader, device):
    Pred = np.zeros(shape=(len(test_dataloader.dataset),), dtype=np.float32)
    Labels = np.zeros(shape=(len(test_dataloader.dataset),), dtype=np.long)
    Feat_vec = []
    total_steps = 0

    n = 0
    for samples in test_dataloader:
        imges = samples["img"]
        label = np.array(samples["cls_id"])
        mini_batch_size = imges.size()[0]
        total_steps = total_steps + mini_batch_size

        imges = imges.to(device)
        with torch.no_grad():
            _, feat_i, feat_o = R(imges)

        # For t-SNE visualization
        test_feat_vec = feat_i - feat_o
        test_feat_vec = test_feat_vec.to("cpu")
        for vec in test_feat_vec:
            vec = vec.reshape(-1)
            vec = vec.tolist()
            Feat_vec.append(vec)

        test_feat_vec = np.array(test_feat_vec)
        test_feat_vec = np.reshape(test_feat_vec, (mini_batch_size, -1))
        pred = clf.predict(test_feat_vec)

        Pred[n : n + mini_batch_size] = pred.reshape(mini_batch_size)
        Labels[n : n + mini_batch_size] = label.reshape(mini_batch_size)
        n += mini_batch_size

    acc = 0
    wrong = 0
    for i in range(len(Labels)):
        if Pred[i] == 1 and Labels[i] == 0:
            acc += 1
        elif Pred[i] == -1 and Labels[i] == 1:
            acc += 1
        else:
            wrong += 1
    Acc = acc / (acc + wrong)

    """
    T-SNE Visualization Latent Vector
    """
    digits3d_lat = TSNE(n_components=3).fit_transform(Feat_vec)
    fig_lat = plt.figure(figsize=(10, 10)).gca(projection="3d")
    for i in range(2):
        target = digits3d_lat[Labels == i]
        fig_lat.scatter(
            target[:, 0], target[:, 1], target[:, 2], label=str(i), alpha=0.5
        )
    fig_lat.legend(bbox_to_anchor=(1.02, 0.7), loc="upper left")
    plt.savefig("./result/t_SNE.png")

    return Acc
