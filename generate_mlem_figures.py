from MLEM import *
import matplotlib.pyplot as plt

dataset = 'svo_long_posit0_word_level_simplif'

for token in ['mean', 'last-token']:
    for model in ['gpt2', 'bert-base-uncased']:

        pipeline = MLEMPipeline(dataset=dataset,
                                take_activation_from=token,
                                model=model)

        fig, ax = plt.subplots()
        ax = pipeline.plot_feature_importance()
        fig.savefig(f'figures/svo_long_posit0/feature_importance_{model}_{token}.png')