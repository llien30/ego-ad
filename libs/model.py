import torch
import torch.nn as nn

from libs.meter import AverageMeter
from libs.flow_vis import visualize_flow
from libs.loss import l2_loss
from libs.test import test

from sklearn.svm import OneClassSVM
import numpy as np
from PIL import Image
import time

import wandb


def egocentric_ad(
    R, D, num_epochs, z_dim, lambdas, dataloader, test_dataloader, no_wandb
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lr_r = 0.002
    lr_d = 0.0002
    beta1, beta2 = 0.5, 0.999

    r_optimizer = torch.optim.Adam(R.parameters(), lr_r, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])

    R.to(device)
    D.to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        R.train()
        D.train()

        print(
            "----------------------(train {}/{})----------------------".format(
                epoch, num_epochs
            )
        )
        r_loss_meter = AverageMeter("R_Loss", ":.4e")
        d_loss_meter = AverageMeter("D_loss", ":.4e")

        for samples in dataloader:
            imges = samples["img"]
            imges = imges.to(device)

            # batch size of last sample will be smaller than the default
            mini_batch_size = imges.size()[0]
            real_label = torch.ones(
                size=(mini_batch_size,), dtype=torch.float32, device=device
            )
            fake_label = torch.zeros(
                size=(mini_batch_size,), dtype=torch.float32, device=device
            )

            """
            Forward Pass
            """
            train_fake_imges, feat_i, feat_o = R(imges)
            pred_real, feat_real = D(imges)
            pred_fake, feat_fake = D(train_fake_imges)

            """
            Loss Calculation
            """
            l1_loss = nn.L1Loss()
            d_loss = nn.BCEWithLogitsLoss(reduction="mean")

            loss_r_adv = l2_loss(feat_real, feat_fake)
            loss_r_con = l1_loss(train_fake_imges, imges)
            loss_r_enc = l2_loss(feat_i, feat_o)

            w_adv = lambdas[0]
            w_con = lambdas[1]
            w_enc = lambdas[2]

            r_loss = w_adv * loss_r_adv + w_con * loss_r_con + w_enc * loss_r_enc
            r_loss_meter.update(r_loss.item())

            loss_d_real = d_loss(pred_real, real_label)
            loss_d_fake = d_loss(pred_fake, fake_label)

            d_loss = (loss_d_real + loss_d_fake) * 0.5
            d_loss_meter.update(d_loss.item())

            """
            Backward Pass
            """
            r_optimizer.zero_grad()
            r_loss.backward(retain_graph=True)
            r_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        t_epoch_finish = time.time()

        # Visualize reconstruction image
        save_path_original = "./result/flow_vis/original"
        save_path_reconstruction = "./result/flow_vis/reconstruction"
        print(imges[0].shape)
        visualize_flow(imges[0], save_path_original, "flow_vis{}.png".format(epoch))
        visualize_flow(
            train_fake_imges[0],
            save_path_reconstruction,
            "flow_vis{}.png".format(epoch),
        )

        print(
            "D_Loss :{:.4f} || R_Loss :{:.4f}".format(
                d_loss_meter.avg, r_loss_meter.avg
            )
        )
        print("time : {:.4f} sec.".format(t_epoch_finish - t_epoch_start))

        print(
            "-----------------------(test {}/{})----------------------".format(
                epoch, num_epochs
            )
        )
        """
        Train One Class SVM
        """
        feature_vector = []
        R.eval()
        for samples in dataloader:

            imges = samples["img"]
            imges = imges.to(device)

            with torch.no_grad():
                _, feat_i, feat_o = R(imges)

            vectors = feat_i - feat_o

            for vect in vectors:
                vect = vect.reshape(-1)
                vect = vect.to("cpu")
                vect = vect.tolist()
                feature_vector.append(vect)

        feature_vector = np.array(feature_vector)
        feature_vector = np.reshape(feature_vector, (-1, z_dim))

        clf = OneClassSVM(kernel="rbf", gamma="auto")
        clf.fit(feature_vector)

        Acc = test(R, D, clf, test_dataloader, device)

        print("Accuracy :{}".format(Acc))
        print("-------------------------------------------------------")

        if not no_wandb:
            wandb.log(
                {
                    "train_time": t_epoch_finish - t_epoch_start,
                    "D_loss": d_loss_meter.avg,
                    "R_loss": r_loss_meter.avg,
                    "Accuracy": Acc,
                },
                step=epoch,
            )

            img = Image.open("./result/t_SNE.png")
            wandb.log({"image": [wandb.Image(img)]}, step=epoch)

        t_epoch_start = time.time()
    return R, D
