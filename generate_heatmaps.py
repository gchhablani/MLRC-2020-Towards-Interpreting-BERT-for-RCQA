"""Script to generate heatmaps for token importances based on Jensen-Shannon
   Divergence across layers.

   Usage:
    $python generate_heatmaps.py --path ~/token_importances --name SQuAD
"""
import argparse
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


from dit.divergences import jensen_shannon_divergence as jsd
from dit.npscalardist import ScalarDistribution as dist


parser = argparse.ArgumentParser(
    prog="generate_heatmaps.py",
    description="Generate JSD heatmaps across layers for token importances.",
)

parser.add_argument(
    "--path",
    type=str,
    action="store",
    help="The path for token importances binary file.",
)
parser.add_argument(
    "--name",
    type=str,
    action="store",
    help="The name of the dataset to be used while storing heatmaps.",
    required=True,
)
parser.add_argument(
    "--topk",
    type=int,
    action="store",
    help="The topk token importances to be chosen.",
    required=True,
    default=2,
)
parser.add_argument(
    "--load_binary",
    action="store_true",
    help="Whether to load binary file heatmaps.",
    default=False,
)

args = parser.parse_args()

if not args.load_binary:
    with open(args.path, "rb") as f:
        token_importances = pkl.load(f)

    importances = []
    for sample in token_importances:
        importances.append([])
        for layers in sample:
            importances[-1].append(layers[1])

    NUM_SAMPLES = len(importances)
    NUM_LAYERS = len(importances[0])

    retained_hmap = np.zeros((NUM_SAMPLES, NUM_LAYERS, NUM_LAYERS))
    removed_hmap = np.zeros((NUM_SAMPLES, NUM_LAYERS, NUM_LAYERS))
    for sample in tqdm(range(NUM_SAMPLES)):
        for layer_i in range(NUM_LAYERS):
            for layer_j in range(layer_i + 1, NUM_LAYERS):
                ## divergence of layer_i and layer_i is always zero
                ## Layer I
                i_max_indices = importances[sample][layer_i].argsort()[-args.topk :]
                mask_i_retained = []
                mask_i_removed = []
                for i in range(len(importances[sample][layer_i])):
                    if i in i_max_indices:
                        mask_i_retained.append(1)
                        mask_i_removed.append(0)
                    else:
                        mask_i_retained.append(0)
                        mask_i_removed.append(1)

                ## Retain
                retained_i = np.array(mask_i_retained * importances[sample][layer_i])
                if np.sum(retained_i) != 0:
                    retained_i = retained_i / np.sum(retained_i)
                else:
                    retained_i = np.ones_like(retained_i) / retained_i.shape[0]

                ## Remove
                removed_i = np.array(mask_i_removed * importances[sample][layer_i])
                if np.sum(removed_i) != 0:
                    removed_i = removed_i / np.sum(removed_i)
                else:
                    removed_i = np.ones_like(removed_i) / removed_i.shape[0]

                ## Layer J
                j_max_indices = importances[sample][layer_j].argsort()[-args.topk :]
                mask_j_retained = []
                mask_j_removed = []
                for j in range(len(importances[sample][layer_j])):
                    if j in j_max_indices:
                        mask_j_retained.append(1)
                        mask_j_removed.append(0)
                    else:
                        mask_j_retained.append(0)
                        mask_j_removed.append(1)

                ## Retain
                retained_j = np.array(mask_j_retained * importances[sample][layer_j])
                if np.sum(retained_j) != 0:
                    retained_j = retained_j / np.sum(retained_j)
                else:
                    retained_j = np.ones_like(retained_j) / retained_j.shape[0]

                ## Remove
                removed_j = np.array(mask_j_removed * importances[sample][layer_j])
                if np.sum(removed_j) != 0:
                    removed_j = removed_j / np.sum(removed_j)
                else:
                    removed_j = np.ones_like(removed_j) / removed_j.shape[0]

                ## Retained Map
                dist_i_retained = dist(retained_i)
                dist_j_retained = dist(retained_j)
                retained_hmap[sample][layer_i][layer_j] = jsd(
                    [dist_i_retained, dist_j_retained]
                )
                retained_hmap[sample][layer_j][layer_i] = retained_hmap[sample][
                    layer_i
                ][layer_j]

                ## Removed Map
                dist_i_removed = dist(removed_i)
                dist_j_removed = dist(removed_j)
                removed_hmap[sample][layer_i][layer_j] = jsd(
                    [dist_i_removed, dist_j_removed]
                )
                removed_hmap[sample][layer_j][layer_i] = removed_hmap[sample][layer_i][
                    layer_j
                ]
else:
    with open(f"Retained Map {args.name} {args.topk}", "rb") as f:
        retained_hmap = pkl.load(f)

    with open(f"Removed Map {args.name} {args.topk}", "rb") as f:
        removed_hmap = pkl.load(f)

plt.style.use("seaborn-white")
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25
## Retained Heatmap
ax = sns.heatmap(
    np.mean(retained_hmap, axis=0)[:13, :13],
    cmap="Blues",
    vmin=0,
    vmax=1,
    annot=True,
    square=True,
    cbar=False,
    fmt=".2f",
    annot_kws={"fontweight": "black", "color": "black"},
)
ax.axhline(y=0, color="k", linewidth=2)
ax.axhline(y=13, color="k", linewidth=2)
ax.axvline(x=0, color="k", linewidth=2)
ax.axvline(x=13, color="k", linewidth=2)
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.title(f"BERT - {args.name} Integrated Gradients JSD\n Top {args.topk} Retained")
plt.savefig(f"JSD_{args.name}_{args.topk}_Heatmap_Retained.png", bbox_inches="tight")
print(
    "Retained Max, Min:\n",
    np.max(np.mean(retained_hmap, axis=0)),
    np.min(np.mean(np.where(retained_hmap > 0, retained_hmap, np.inf), axis=0)),
)
with open(f"Retained Map {args.name} {args.topk}", "wb") as f:
    pkl.dump(retained_hmap, f)

plt.clf()
## Retained Heatmap
ax = sns.heatmap(
    np.mean(removed_hmap, axis=0)[:13, :13],
    cmap="Blues",
    vmin=0,
    vmax=1,
    annot=True,
    square=True,
    cbar=False,
    fmt=".2f",
    annot_kws={"fontweight": "black", "color": "black"},
)
ax.axhline(y=0, color="k", linewidth=2)
ax.axhline(y=13, color="k", linewidth=2)
ax.axvline(x=0, color="k", linewidth=2)
ax.axvline(x=13, color="k", linewidth=2)
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.title(f"BERT - {args.name} Integrated Gradients JSD\n Top {args.topk} Removed")
plt.savefig(f"JSD_{args.name}_{args.topk}_Heatmap_Removed.png", bbox_inches="tight")
print(
    "Removed Max, Min:\n",
    np.max(np.mean(removed_hmap, axis=0)),
    np.min(np.mean(np.where(removed_hmap > 0, removed_hmap, np.inf), axis=0)),
)

with open(f"Removed Map {args.name} {args.topk}", "wb") as f:
    pkl.dump(removed_hmap, f)
