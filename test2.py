# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_model, get_data, get_img_list
from pylab import *
import cv2
import warnings

warnings.filterwarnings("ignore")
plt.switch_backend("agg")

# --------------------------------------------------------------------------------

models_list = ["UNet512", "M_Net", "FM_Net_SelfLearning", "SU_Net"]
dataset_list = ["DRIVE", "STARE", "CHASEDB1", "ORIGA", "REFUGE"]
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PyTorch ASOCT_Demo")
# ---------------------------
# params do not need to change
# ---------------------------
parser.add_argument("--epochs", type=int, default=250, help="the epochs of this run")
parser.add_argument(
    "--n_class",
    type=int,
    default=3,
    help="the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class",
)
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument(
    "--GroupNorm", type=bool, default=True, help="decide to use the GroupNorm"
)
parser.add_argument(
    "--BatchNorm", type=bool, default=False, help="decide to use the BatchNorm"
)
# ---------------------------
# model
# ---------------------------
parser.add_argument("--datasetID", type=int, default=3, help="dir of the all img")
parser.add_argument("--SubImageID", type=int, default=1, help="Only for Stare Dataset")
# parser.add_argument('--best_model', type=str,  default='D:\\FrangiNet\\FrangiNet\\models\\FM_Net_SelfLearning_M-Net\\75.pth',
#                    help='the pretrain model')
parser.add_argument(
    "--best_model",
    type=str,
    default="D:\\DIP\\DeepGuidance\\models\\SU_Net_CHASEDB_Train\\65.pth",
    help="the pretrain model",
)
parser.add_argument(
    "--model_id", type=int, default=3, help="the id of choice_model in models_list"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="the num of img in a batch"
)
parser.add_argument("--img_size", type=int, default=512, help="the train img size")
parser.add_argument(
    "--my_description",
    type=str,
    default="ORIGA_Test",
    help="some description define your train",
)
# ---------------------------
# GPU
# ---------------------------
parser.add_argument("--use_gpu", type=bool, default=True, help="dir of the all ori img")
parser.add_argument("--gpu_avaiable", type=str, default="0", help="the gpu used")

args = parser.parse_args()


def calculate_voe(predicted_mask, ground_truth, label):
    """Calculate the Volumetric Overlap Error (VOE) for a specific label. Parameters: predicted_mask (numpy.ndarray): Predicted segmentation mask. gt (numpy.ndarray): Ground truth segmentation mask. label (int): Label to calculate VOE for (1 for optic disc, 2 for optic cup). Returns: float: VOE for the specified label."""
    segm_label = predicted_mask == label
    gt_label = ground_truth == label
    intersection = np.logical_and(segm_label, gt_label).sum()
    union = np.logical_or(segm_label, gt_label).sum()
    if union == 0:
        raise ValueError(f"Union area for label {label} is zero, cannot calculate VOE.")
    voe = 1.0 - (intersection / float(union))
    return voe


def fast_test(model, args, img_list, model_name, epoch, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    softmax_2d = nn.Softmax2d()
    EPS = 1e-12

    Dataset = dataset_list[args.datasetID]
    SubID = args.SubImageID

    E_DISK = []
    E_CUP = []

    for i, path in enumerate(img_list):

        # img, imageGreys, gt, tmp_gt, img_shape, label_ori , mask_ori= get_data(args.data_path, [path], img_size=args.img_size, gpu=args.use_gpu,flag=True)
        img, imageGreys, gt, tmp_gt, img_shape, label_ori, mask_ori = get_data(
            Dataset, [path], img_size=args.img_size, gpu=args.use_gpu, flag="test"
        )

        # Save the image
        model.eval()
        start = time.time()
        # out, side_5, side_6, side_7, side_8, FeatureMap, NormalizedFilterResponse2, FeatureMap2 = model(img, imageGreys)
        out, side_5, side_6, side_7, side_8 = model(img, imageGreys)

        # BeforePath = os.path.join("D:\\FrangiNet\\AllResult", "Before" + path[:-5]+'.jpg')
        # ResponsePath = os.path.join("D:\\FrangiNet\\AllResult", "Response" + path[:-5]+'.jpg')
        # AfterPath = os.path.join("D:\\FrangiNet\\AllResult", "After" + path[:-5]+'.png')
        # cv2.imwrite(BeforePath, FeatureMap * 1000)
        # cv2.imwrite(ResponsePath, NormalizedFilterResponse2 * 10000)
        # cv2.imwrite(AfterPath, FeatureMap2 * 256)

        # out, side_5, side_6, side_7, side_8 = model(img, imageGreys)
        end = time.time()
        out = torch.log(softmax_2d(side_8) + EPS)
        # out = F.upsample(out, size=(img_shape[0][0], img_shape[0][1]), mode="bilinear")
        ppi = torch.argmax(out, 1).cpu().numpy().squeeze()
        # y_pred = out[:, 1, :, :]

        # plt.imsave(, y_pred[0], cmap="gray")

        # y_pred = y_pred.reshape([-1])
        # ppi = np.argmax(out, 1)

        # Output the prediction
        # ResultImage = "D:\\FrangiNet\\FrangiNet\\ResultImage"
        # ResultImage = "D:\\FrangiNet\\ResultImage\\SUNETDRIVE"
        # ResultImage = "D:\\FrangiNet\\ResultImage\\SUNCHASEDB1"

        # ImageName = path[:-4] + 'jpg'
        fileName = f"{result_path}/{epoch}_{i}.jpeg"
        ppi_temp = np.squeeze(ppi)

        tmp_out = ppi.reshape([-1])
        tmp_gt = label_ori.reshape([-1])

        Mask = mask_ori[0]
        # cv2.imshow('img', Mask*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        Mask = Mask.reshape([-1])

        # SelectOut = tmp_out[np.flatnonzero(Mask)]
        # SelectGT = tmp_gt[np.flatnonzero(Mask)]
        # my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        # my_confusion = metrics.confusion_matrix(SelectOut, SelectGT).astype(np.float32)
        # meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
        e_disk = calculate_voe(ppi, gt.cpu().numpy().squeeze(), 1)
        e_cup = calculate_voe(ppi, gt.cpu().numpy().squeeze(), 2)

        ppi_temp[ppi_temp == 0] = 255
        ppi_temp[ppi_temp == 1] = 128
        ppi_temp[ppi_temp == 2] = 0

        plt.imsave(fileName, ppi_temp, cmap="gray")

        E_DISK.append(e_disk)
        E_CUP.append(e_cup)

        # print(path+r'/'+str(len(img_list))+': '+'| Acc: {:.4f} | Se: {:.4f} | Sp: {:.4f} | Auc: {:.4f} |  Background_IOU: {:f}, vessel_IOU: {:f}'.format(Acc,Se,Sp,Auc,IU[0], IU[1])+'  |  time:%s'%(end-start))
        # For latex
        # print(Acc)
        # print(path + '&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(Acc, Auc, Se, Sp))

    print(
        "E_cup: %s  |  E_disk: %s"
        % (str(np.mean(np.stack(E_DISK))), str(np.mean(np.stack(E_CUP))))
    )

    # store test information
    with open(
        r"D:\\DIP\\DeepGuidance\\logs\\%s_%s.txt" % (model_name, args.my_description),
        "a+",
    ) as f:
        f.write(
            "E_cup: %s  |  E_disk: %s"
            % (str(np.mean(np.stack(E_DISK))), str(np.mean(np.stack(E_CUP))))
        )
        f.write("\n\n")

    # return np.mean(np.stack(Vessel_IOU))
    return np.mean(np.stack(E_DISK)), np.mean(np.stack(E_CUP))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_avaiable

    model_name = models_list[args.model_id]

    model = get_model(model_name)
    model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

    if args.use_gpu:
        model.cuda()
    if True:

        model_path = "D:\\DIP\\DeepGuidance\\models\\SU_Net_ORIGA_Train\\95.pth"
        model.load_state_dict(torch.load(model_path))
        print("success load models: %s_%s" % (model_name, args.my_description))

    print("This model is %s_%s_%s" % (model_name, args.n_class, args.img_size))
    Dataset = dataset_list[args.datasetID]
    SubID = args.SubImageID

    test_img_list = get_img_list(Dataset, SubID, flag="test")

    fast_test(
        model,
        args,
        test_img_list,
        model_name,
        0,
        f"D:\\DIP\\DeepGuidance\\results\\{Dataset}\\test",
    )
