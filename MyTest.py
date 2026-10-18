import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')

for _data_name in ['VisualInspection']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        with torch.no_grad():
            res5, res4, res3, res2 = model(image)
        
        #Prediction outputs at each stage
        intermediate_logits = {
            "Initial prediction": res5,
            "After RA4": res4,
            "After RA3": res3,
            "After RA2 (Final)": res2,
        }

        # Convert logits to probabilities at the ground-truth resolution
        intermediate_probabilities = {}

        for stage, logits in intermediate_logits.items():
            resized = F.interpolate(
                logits,
                size=gt.shape,
                mode="bilinear",
                align_corners=False
            )

            prob = torch.sigmoid(resized)
            intermediate_probabilities[stage] = (
                prob.detach().cpu().numpy().squeeze()
            )

        # Visualize only the first few images to avoid hundreds of plots
        if i < test_loader.size:
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))

            # Original image (undo ImageNet normalization)
            img = image[0].detach().cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(img * std + mean, 0, 1)

            axes[0, 0].imshow(img)
            axes[0, 0].set_title("Input image")

            axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[0, 1].set_title("Ground truth")

            positions = [
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            ]

            for (stage, prob), (r, c) in zip(
                intermediate_probabilities.items(), positions
            ):
                axes[r, c].imshow(
                    prob, cmap="gray", vmin=0, vmax=1
                )
                axes[r, c].set_title(stage)

            for ax in axes.flat:
                ax.axis("off")

            plt.tight_layout()
            comparison_dir = "./results/VisualInspection/comparisons"
            os.makedirs(comparison_dir, exist_ok=True)

            plt.savefig(
                os.path.join(comparison_dir, f"{os.path.splitext(name)[0]}_stages.png"),
                dpi=150,
                bbox_inches="tight"
            )
            plt.show()
            plt.close(fig)

        
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(
            os.path.join(save_path, name),
            (res * 255).astype(np.uint8)
        )