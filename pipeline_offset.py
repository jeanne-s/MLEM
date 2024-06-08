from MLEM import *
from tqdm import tqdm
#import torch
import multiprocessing

print("CPUs:")
print(multiprocessing.cpu_count())


pipeline = MLEMPipeline(dataset='svo_long_word_level_offset_simplif',
                        take_activation_from='last-token',
                        model='mamba')

pipeline.skip_existing = 0
pipeline.compute_features_distance()

ax = pipeline.plot_correlation(thresh=0.2)
ax.figure.savefig(f"figures/svo_long_offset/corr_svo_long_offset.png")

pipeline.skip_existing = 0 # Remove if recomputing is not necessary
pipeline.verbose = 0
for layer in tqdm(range(1, 25)):
    pipeline.layer = layer
    pipeline.compute_feature_importance()

ax = pipeline.plot_feature_importance()
ax.figure.savefig(f"figures/svo_long_offset/feature_importance_mamba_last-token.png")
