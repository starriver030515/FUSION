import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E", "F"])
    return parser.parse_args()

def calculate_accuracy(data, source):
    total_count = len([x for x in data if x['source'] == source])
    accuracy = len([x for x in data if x['source'] == source and x['text'] == x['ground_truth']])
    return accuracy / total_count

if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir

    data = [json.loads(q) for q in open(os.path.expanduser(args.output_file), "r")]

    for pred in data:
        pred_text = pred['text']
        if pred_text in args.options:
            answer = pred_text
        else:
            pattern = re.compile(r'([A-F])\.|([A-F]),|([A-F])\n|\(([A-F])\)')
            res = pattern.findall(pred_text)
            if res:
                answer = res[0][0] or res[0][1] or res[0][2] or res[0][3]
            else:
                answer = "FAILED"
        pred['text'] = answer

    accuracy_2d_ade = calculate_accuracy(data, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(data, 'COCO')
    accuracy_3d_omni = calculate_accuracy(data, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    # Print the results
    print(f"CV-Bench Accuracy: {combined_accuracy:.4f}")
    print()
    print(f"Type Accuracies:")
    print(f"2D Accuracy: {accuracy_2d:.4f}")
    print(f"3D Accuracy: {accuracy_3d:.4f}")
    print()
    print(f"Source Accuracies:")
    print(f"ADE20K Accuracy: {accuracy_2d_ade:.4f}")
    print(f"COCO Accuracy: {accuracy_2d_coco:.4f}")
    print(f"Omni3D Accuracy: {accuracy_3d_omni:.4f}")
