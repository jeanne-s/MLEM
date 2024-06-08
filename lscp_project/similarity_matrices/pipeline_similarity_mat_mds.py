
import pandas as pd
import os
import sys
from pathlib import Path
from scipy.stats import pearsonr
import plotly.express as px
from typing import List
#from sklearn.manifold import MDS
#import plotly.graph_objects as go
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel



class Model():

    def __init__(self,
                 model_name: str,
    ):
        self.model_name = model_name
        

    @property
    def model(self):
        if "pythia" in self.model_name:
            return self.get_pythia_model(self.model_name.replace("pythia-", ""))
        if "gpt2" in self.model_name:
            return self.get_gpt2_model(self.model_name.replace("gpt2-", ""))
        if "bert" in self.model_name:
            return self.get_bert_model(self.model_name)
        if "Mistral" in self.model_name:
            return self.get_mistral_model(self.model_name.replace("Mistral-", ""))
        if "mamba" in self.model_name:
            return self.get_mamba_model(self.model_name.replace("mamba-", ""))
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")
    
    @property
    def tokenizer(self):
        if "bert" in self.model_name:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers

    
    def get_layers_to_enumerate(self):
        if 'gpt' in self.model_name:
            return self.model.transformer.h
        elif 'pythia' in self.model_name:
            return self.model.gpt_neox.layers
        elif 'bert' in self.model_name:
            return self.model.encoder.layer
        elif 'Mistral' in self.model_name:
            return self.model.model.layers
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")


    def get_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        return model

    def get_pythia_model(self, size: str):
        assert size in ["14m", "31m", "70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
        model_name = f"EleutherAI/pythia-{size}"
        return self.get_model(model_name)

    def get_gpt2_model(self, size: str):
        assert size in ["gpt2", "small", "medium", "large", "xl"]
        if (size == "small" or size=="gpt2"):
            model_name = "gpt2"
        else:
            model_name = f"gpt2-{size}"
        return self.get_model(model_name)

    def get_bert_model(self, model_name: str):
        model = BertModel.from_pretrained(model_name)
        return model

    def get_mistral_model(self, size: str):
        assert size in ["7", "7x8"]
        model_name = f"mistralai/Mistral-{size}B-v0.1"
        return self.get_model(model_name)

    def get_mamba_model(self, size: str):
        assert size in ["130m", "370m", "790m", "1.4b", "2.8b"]
        model_name = f"state-spaces/mamba-{size}"
        return self.get_model(model_name)



class SimilarityMatrix():

    def __init__(self, 
                 model_names: List[str],
                 dataset: str,
                 take_activation_from: str = None,
                 fi_data_path: str = '/scratch2/jsalle/MLEM/experiments'
    ):
        self.model_names = model_names
        self.take_activation_from = take_activation_from
        self.dataset = dataset
        self.dataset_short = dataset.split('_word')[0]
        self.fi_data_path = fi_data_path

    @property
    def models(self):
        """ List of all models defined in self.model_names.
        """
        return [Model(model_name).model for model_name in self.model_names]


    def get_feature_importances(self):

        fi_df = pd.DataFrame()

        for model_name in self.model_names:
            n_layers = Model(model_name).n_layers

            for layer_id in range(1, n_layers+1):

                if model_name=='bert-base-uncased' and self.dataset_short=='clause_type':
                    model_name = 'bert_base_uncased'

                fi_current_layer = pd.read_csv(os.path.join(self.fi_data_path, 
                                            self.dataset, 
                                            model_name, 
                                            'spe_tok', 
                                            'analysis', 
                                            self.take_activation_from,
                                            'euclidean/zscore_False',
                                            f'layer_{layer_id}',
                                            'min_max_True',
                                            'feature_importance_conditional_False.csv'))
                fi_df = pd.concat([fi_df, fi_current_layer])

        return fi_df
    

    def get_feature_importances_correlations_across_layers(self):
        
        fi_df = self.get_feature_importances()

        layers_per_model_range = [[i for i in range(0, Model(m).n_layers)] for m in self.model_names] # layers are indexed from 1, not 0
        correlation_df = pd.DataFrame()

        a, b = 0, 0
        for i in range(0, len(self.model_names)):
            for j in range(0, len(self.model_names)):
                model1, model2 = self.model_names[i], self.model_names[j]
                for layer_id1 in layers_per_model_range[i]:
                    for layer_id2 in layers_per_model_range[j]:
                        corr = pearsonr(fi_df[(fi_df['layer']==layer_id1+1) & (fi_df['_model']==model1)]['importance'], 
                                        fi_df[(fi_df['layer']==layer_id2+1) & (fi_df['_model']==model2)]['importance'])[0] 
                        temp_dict = {'1': f'{model1}_l{str(layer_id1).zfill(2)}',
                                '2': f'{model2}_l{str(layer_id2).zfill(2)}',
                                'corr': corr}
                        correlation_df = pd.concat([correlation_df, pd.DataFrame([temp_dict])],  ignore_index=True)

        correlation_df.sort_values(by=['1', '2'], inplace=True)
        correlation_matrix = correlation_df.pivot(index='1', columns='2', values='corr')

        if len(self.model_names) == 1:
            correlation_df.to_csv(f'assets/{self.dataset_short}/correlation_fi_{self.model_names[0]}_{self.take_activation_from}.csv')

        else:
            str_models = "-".join(str(x) for x in self.model_names)
            correlation_df.to_csv(f'assets/{self.dataset_short}/correlation_fi_{str_models}_{self.take_activation_from}.csv')
        return correlation_matrix


    def plot_correlation_matrix(self, correlation_matrix):
        fig = px.imshow(correlation_matrix, 
                        #text_auto=True, 
                        color_continuous_scale='portland',
                        title=f'Correlation FI')
        fig.update_layout(xaxis_title="Layers",
                          yaxis_title="Layres")

        return fig


    def plot_mds(self, similarities_df):
        
        similarities_df = similarities_df.fillna(value=1.)

        dissimilarities_df = 1 - similarities_df
        embedding = MDS(n_components=2, 
                        normalized_stress='auto',
                        dissimilarity="precomputed") 
        mds_out = embedding.fit_transform(dissimilarities_df)

        dissimilarities_df['model'] = dissimilarities_df.index
        dissimilarities_df['MDS1'] = mds_out[:, 0]
        dissimilarities_df['MDS2'] = mds_out[:, 1]

        def process_model_info(label):
            parts = label.split('_')
            model_part = parts[0].split('-')[0]  # Assumes the model prefix is always before the first dash
            layer_number = int(parts[1].replace('l', ''))
            simplified_label = f"{model_part}_l{layer_number}"
            return simplified_label, layer_number

        dissimilarities_df[['simplified_model', 'layer_number']] = dissimilarities_df['model'].apply(
            lambda x: pd.Series(process_model_info(x))
        )        
        palette = px.colors.sequential.Plotly3_r
        max_layer = dissimilarities_df['layer_number'].max()
        color_map = {i: palette[i % len(palette)] for i in range(max_layer + 1)}
        dissimilarities_df['text_color'] = [color_map[x] for x in dissimilarities_df['layer_number']]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dissimilarities_df["MDS1"],
            y=dissimilarities_df["MDS2"],
            text=dissimilarities_df["simplified_model"],
            mode='text',
            textposition="top right",
            textfont=dict(
                family="sans serif",
                size=12,
                color=dissimilarities_df["text_color"]
            )
        ))

        if len(self.model_names) == 1:
            dissimilarities_df.to_csv(f'assets/{self.dataset_short}/fi_mds_{self.model_names[0]}_{self.take_activation_from}.csv')

        else:
            str_models = "-".join(str(x) for x in self.model_names)
            dissimilarities_df.to_csv(f'assets/{self.dataset_short}/fi_mds_{str_models}_{self.take_activation_from}.csv')

        return fig
    


for model_set in [['gpt2', 'bert-base-uncased'], ['gpt2'], ['bert-base-uncased']]:
    for dataset in ['svo_long_word_level_simplif', 'clause_type_word_level_simplif', 'relative_clause_word_level_simplif']:
        for activation in ['last-token', 'mean']:

            sim_matrix_one = SimilarityMatrix(model_names=model_set,
                                            dataset=dataset,
                                            take_activation_from=activation)

            correlation_matrix = sim_matrix_one.get_feature_importances_correlations_across_layers()
            _ = sim_matrix_one.plot_correlation_matrix(correlation_matrix=correlation_matrix)
            _ = sim_matrix_one.plot_mds(correlation_matrix)