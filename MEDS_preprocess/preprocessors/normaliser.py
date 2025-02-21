import pandas as pd
from os.path import join
from pathlib import Path
import os
from tqdm import tqdm
from azureml.core import Dataset
from preprocessors.azure import AzurePreprocessor
import torch
from azure_run import datastore
import numpy as np

class Normaliser():
    # load data in dask
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.data_store =  datastore(cfg.data.data_store)
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.firstRound = True
        self.normalisation_type = cfg.data['norm_type']
        self.azure_processor = AzurePreprocessor(cfg, logger)

        # Load distribution data
        if 'dist_path' not in cfg.data:
            dist = self.get_lab_dist()
            dist_save_path = join(cfg.paths.output_dir, 'lab_val_dict.pt')
            torch.save(dist, dist_save_path)
            self.logger.info(f'Saved lab distribution to {dist_save_path}')

        else:
            dist_path = join(self.cfg.data.dist_path)
            dist_dataset = Dataset.File.from_files(path=(self.data_store, dist_path))
            mount_context = dist_dataset.mount()
            mount_context.start()
            mount_point = mount_context.mount_point
            dist = torch.load(join(mount_point, 'lab_val_dict.pt'), weights_only=True)
            self.vocab = torch.load(join(mount_point, 'vocabulary.pt'), weights_only=True)        

        if self.normalisation_type == 'Min_max':
            # Gets the min max values for all concepts
            min_max_dict = {
                concept: (
                    np.percentile(dist[concept], 0.01 * 100) if len(dist[concept]) > 1 else dist[concept][0],
                    np.percentile(dist[concept], 0.99 * 100) if len(dist[concept]) > 1 else dist[concept][0]
                ) 
                for concept in dist 
                if dist[concept]
            }
            self.min_max_vals = min_max_dict

        elif self.normalisation_type == 'Categorise':
            self.quantiles = {}
            for concept in dist:
                sorted_dist = sorted(dist[concept])
                if len(sorted_dist) > 0:
                    q1 = np.percentile(sorted_dist, 25)
                    q2 = np.percentile(sorted_dist, 50)
                    q3 = np.percentile(sorted_dist, 75)
                    self.quantiles[concept] = (q1, q2, q3)
                else:
                    self.quantiles[concept] = (0, 0, 0)

        elif self.normalisation_type == 'Quantiles':
            self.n_quantiles = cfg.data['n_quantiles']
            self.quantiles = {}
            for concept in dist:
                sorted_dist = sorted(dist[concept])
                if len(sorted_dist) > 0:
                    # Calculate percentiles for 10 quantiles (10, 20, ..., 100)
                    quantiles = [np.percentile(sorted_dist, i) for i in np.linspace(100/self.n_quantiles, 100, self.n_quantiles)]
                    self.quantiles[concept] = quantiles
                else:
                    self.quantiles[concept] = [0] * self.n_quantiles
        else:
            raise ValueError('Invalid type of normalisation')

    def __call__(self):
        cfg = self.cfg
        save_name = cfg.data.save_name
        if not Path(join(cfg.paths.output_dir, save_name)).exists():
            counter = 0
            # Iterate over chunks of the CSV file
            for chunk in tqdm(self.azure_processor.load_chunks(cfg.data), desc='Chunks'):
                if 'Column1' in chunk.columns:
                    chunk = chunk.drop(columns='Column1')
                chunk = chunk.reset_index(drop=True)
                self.logger.info(f'Loaded {cfg.data.chunksize*counter}')
                chunk_processed = self.process_chunk(chunk)
                if counter == 0:
                    self.azure_processor.save(chunk_processed, cfg.data,  f'concept.{save_name}', mode='w')
                else:
                    self.azure_processor.save(chunk_processed, cfg.data, f'concept.{save_name}', mode='a')
                
                counter += 1

    def get_lab_dist(self):
        self.logger.info('Getting lab distribution')
        cfg = self.cfg
        save_name = 'lab_val_dict.pt'
        lab_val_dict = {}
        counter = 0
        for chunk in tqdm(self.azure_processor.load_chunks(cfg.data), desc='Chunks'):
            self.logger.info(f'Loaded {cfg.data.chunksize*counter}')
            chunk['RESULT'] = pd.to_numeric(chunk['RESULT'], errors='coerce')
            chunk = chunk.dropna(subset=['RESULT'])
            grouped = chunk.groupby('CONCEPT')['RESULT'].apply(list).to_dict()

            for key, values in grouped.items():
                if key in lab_val_dict:
                    lab_val_dict[key].extend(values)
                else:
                    lab_val_dict[key] = values
            
            counter += 1
        return lab_val_dict

    def process_chunk(self, chunk):
        chunk['RESULT'] = chunk.apply(self.normalise, axis=1)
        return chunk
        
    def normalise(self, row):
        concept = row['CONCEPT']
        value = row['RESULT']        
        # Returns value if it is not numerical
        if not pd.notnull(pd.to_numeric(value, errors='coerce')):
            return value
        else: 
            value = pd.to_numeric(value)

        # Normalises numerical values
        if self.normalisation_type == 'Min_max':
            return self.min_max_normalise(concept, value)
        elif self.normalisation_type == 'Quantiles':
            return self.quantile(concept, value)
        else:
            Warning(f"Normalisation type {self.normalisation_type} not implemented")
        
    def min_max_normalise(self, concept, value):
        if concept in self.min_max_vals:
            (min_val, max_val) = self.min_max_vals[concept]
            if max_val != min_val: 
                normalised_value = (value - min_val) / (max_val - min_val)
                return round(max(0, min(1, normalised_value)),3)
            else:
                return "UNIQUE"
        else:
            return "N/A"
        
    def quantile(self, concept, value):
        if concept not in self.quantiles:
            return 'N/A'
        else:
            quantile_values = self.quantiles[concept]
            # Ensure there are exactly 12 quantiles
            if len(quantile_values) != self.n_quantiles:
                raise ValueError(f"Expected {self.n_quantiles} quantiles for concept '{concept}'")            
            for i, q in enumerate(quantile_values, start=1):
                if value <= q:
                    return 'Q{}'.format(i)
            return 'Q{}'.format(self.n_quantiles)
