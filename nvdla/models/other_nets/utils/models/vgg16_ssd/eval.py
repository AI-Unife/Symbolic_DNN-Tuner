from vision_models_pytorch.models.vgg16_ssd.utils import *

from tqdm import tqdm
from pprint import PrettyPrinter

import torch

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_ssd(test_loader, model, class_names):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()
    model.to(device)

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = (
        list()
    )  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
            )
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            print(
                f"Len True Labels {len(true_labels)} (+{len(labels)}), Len True Boxes {len(true_boxes)} (+{len(boxes)})"
            )
            print(
                f"Labels {sum(lab.size(0) for lab in true_labels)} Boxes {sum(box.size(0) for box in true_boxes)}"
            )
        print()
        # Calculate mAP
        APs, mAP = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            true_difficulties,
            class_names,
        )

    # Print AP for each class
    pp.pprint(APs)

    print("\nMean Average Precision (mAP): %.3f" % mAP)
