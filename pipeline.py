from MLEM import *
from tqdm import tqdm
#import torch
import multiprocessing


print("CPUs:")
print(multiprocessing.cpu_count())



for activation in tqdm(["mean", "last-token"], desc='Position'):
    for model in tqdm(['gpt2', 'bert-base-uncased'], desc='Model'):

        pipeline = MLEMPipeline(dataset='clause_type_word_level_simplif',
                                take_activation_from=activation,
                                model=model)


        pipeline.skip_existing = 0
        pipeline.compute_features_distance()

        ax = pipeline.plot_correlation(thresh=0.2)
        ax.figure.savefig(f"figures/clause_type/corr_clause_type.png")

        pipeline.skip_existing = 0 # Remove if recomputing is not necessary
        pipeline.verbose = 0
        for layer in tqdm(range(1, 13)):
            pipeline.layer = layer
            pipeline.compute_feature_importance()

        ax = pipeline.plot_feature_importance()
        ax.figure.savefig(f"figures/clause_type/feature_importance_{model}_{activation}.png")
