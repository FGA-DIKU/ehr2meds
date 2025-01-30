import pandas as pd
import os
from azureml.core import Dataset
from tqdm import tqdm
import hashlib
from azureml.core import Workspace, Dataset, Datastore
from os.path import join
from datetime import timedelta
import torch 

def setup_azure(run_name, datastore_name='workspaceblobstore', dataset_name='BREAST_CANCER'):
    from azure_run import datastore
    from azure_run.run import Run
    from azureml.core import Dataset
    
    run = Run
    run.name(run_name)
    ds = datastore(datastore_name)
    dataset = Dataset.File.from_files(path=(ds, dataset_name))
    mount_context = dataset.mount()
    mount_context.start()  # this will mount the file streams
    return run, mount_context


class MEDSPreprocessor():
    def __init__(self, cfg, logger, datastore, dump_path, file_datastore=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.datastore =  datastore
        if 'file_path' in cfg.combine:
            _, mount_context = setup_azure(run_name=self.cfg.run_name, datastore_name='workspaceblobstore', dataset_name='BREAST_CANCER')
            self.mount_point = mount_context.mount_point
            self.data_path = join(self.mount_point, cfg.combine.file_path)
        else:
            self.data_path = False
        self.dump_path = dump_path if dump_path is not None else None
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.removed_concepts = {k:0 for k in self.cfg.concepts.keys()} # count concepts that are removed
        self.initial_patients = set()
        self.formatted_patients = set()

    def __call__(self):
        if self.data_path is False:
            subject_id_mapping = self.patients_info()
            self.format_concepts(subject_id_mapping)        

    def filter_dates(self):
        self.logger.info("Filter dates")
        for concept_type, concept_config in tqdm(self.cfg.concepts.items(), desc="Concepts"):
            if concept_type not in ['diagnosis', 'medication', 'labtest', 'procedure', 'admissions']:
                raise ValueError(f'{concept_type} not implemented yet')
            self.logger.info(f"INFO: Filter {concept_type}")            
            df = self.load_pandas(concept_config)
            if self.test:
                df = df.sample(100000)
            df = self.select_columns(df, concept_config)
            df = self.change_dtype(df, concept_config)
            df = self.filter_dates_pipeline(df, concept_config)
            self.save(df, concept_config, f'{concept_type}')

    def iterate_through_file(self, concept_type, concept_config, subject_id_mapping, first=True):
        for chunk in tqdm(self.load_chunks(concept_config), desc='Chunks'):
            # process each chunk here.
            chunk_processed = self.concepts_process_pipeline(chunk, concept_type, subject_id_mapping)
            if first:
                self.save(chunk_processed, concept_config, f'{concept_type}', mode='w')
                first = False
            else:
                self.save(chunk_processed, concept_config, f'{concept_type}', mode='a')

    def format_concepts(self, subject_id_mapping):
        """Loop over all top-level concepts (diagnosis, medication, procedures, etc.) and call processing"""
        for concept_type, concept_config in tqdm(self.cfg.concepts.items(), desc="Concepts"):
            if concept_type not in ['diagnosis', 'medication', 'labtest', 'procedure', 'admissions']:
                raise ValueError(f'{concept_type} not implemented yet')
            self.logger.info(f"INFO: Preprocess {concept_type}")
            first = True

            if type(concept_config.filename) == list:
                for file_name in concept_config.filename:
                    concept_config.filename = file_name
                    self.iterate_through_file(concept_type, concept_config, subject_id_mapping, first=first)
                    first=False
            else:
                self.iterate_through_file(concept_type, concept_config, subject_id_mapping)

    def concepts_process_pipeline(self, concepts, concept_type, subject_id_mapping):
        if isinstance(self.cfg.filtering, dict) and 'filter_date' in self.cfg.filtering:
            filter_date = self.cfg.filtering['filter_date']
        else:
            filter_date = False
        """Process concepts"""
        formatter = getattr(self, f"format_{concept_type}")
        concepts = formatter(concepts, subject_id_mapping)
        self.initial_patients = self.initial_patients | set(concepts.subject_id.unique())
        self.logger.info(f"{len(self.initial_patients)} before cleaning")
        self.logger.info(f"{len(concepts)} concepts")
        concepts = concepts.dropna()
        self.logger.info(f"{len(concepts)} concepts after removing nans")
        concepts = concepts.drop_duplicates()
        self.logger.info(f"{len(concepts)} concepts after dropping duplicates nans")
        if filter_date:
            concepts = self.filter_dates_pipeline(concepts, filter_date)
        self.logger.info(f"{len(concepts)} concepts after filtering on date")
        self.formatted_patients = self.formatted_patients | set(concepts.subject_id.unique())
        self.logger.info(f"{len(self.formatted_patients)} after cleaning")
        return concepts

    def filter_dates_pipeline(self, chunk, filter_date):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', infer_datetime_format=True)
        filter_date_dt = pd.to_datetime(filter_date)
        filtered_chunk = chunk[chunk['timestamp'] < filter_date_dt]
        return filtered_chunk

    @staticmethod
    def format_diagnosis(diag, cfg):
        diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
        if 'fill_diags' in cfg and cfg['fill_diags']:
            diag['code'] = diag['code'].fillna(diag['Diagnose'])
        diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
        diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
        diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
        return diag

    @staticmethod
    def format_procedure(proc, subject_id_mapping):
        proc['code'] = proc['ProcedureCode'].str.replace(' ', '')
        proc = proc.drop(['ProcedureCode'], axis=1)
        proc = proc.rename(columns={'CPR_hash':'subject_id', 'ServiceDatetime':'timestamp', 'ProcedureName':'description'})
        proc['code'] = proc['code'].map(lambda x: 'P'+x)
        proc['subject_id'] = proc['subject_id'].map(subject_id_mapping)
        return proc

    @staticmethod
    def format_labtest(labs, subject_id_mapping):
        labs = labs.rename(columns={'CPR_hash':'subject_id', 'BestOrd':'code', 'Prøvetagningstidspunkt': 'timestamp', 'Resultatværdi':'numeric_value'})
        labs['code'] = labs['code'].map(lambda x: 'LAB_'+x)
        labs['numeric_value'] = pd.to_numeric(labs['numeric_value'], errors='coerce')
        labs = labs.dropna(subset=['numeric_value'])
        labs['subject_id'] = labs['subject_id'].map(subject_id_mapping)
        return labs
    
    @staticmethod
    def format_medication(med, subject_id_mapping):
        med.loc[:, 'CONCEPT'] = med.ATC.fillna('Ordineret_lægemiddel')
        med.loc[:, 'TIMESTAMP'] = med.Administrationstidspunkt.fillna("Bestillingsdato")
        med = med.rename(columns={'CPR_hash':'PID'})
        med = med[['PID','CONCEPT','TIMESTAMP']]
        med['CONCEPT'] = med['CONCEPT'].map(lambda x: 'M'+x)
        med = med.rename(columns={'PID':'subject_id', 'TIMESTAMP':'timestamp', 'CONCEPT':'code'})
        med['subject_id'] = med['subject_id'].map(subject_id_mapping)
        return med
    
    @staticmethod
    def format_admissions(adm, subject_id_mapping):
        adm['Flyt_ind'] = pd.to_datetime(adm['Flyt_ind'])
        adm['Flyt_ud'] = pd.to_datetime(adm['Flyt_ud'])
        
        adm.sort_values(by=['CPR_hash', 'Flyt_ind'], inplace=True)
        merged_admissions = []
        current_row = None
        for index, row in adm.iterrows():
            if current_row is None:
                current_row = row
                continue

            # Check for overlap or if next admission is within 24 hours after the current admission's discharge
            if row['Flyt_ind'] <= current_row['Flyt_ud'] + timedelta(hours=24):
                # Extend the current admission's discharge time if the next admission's discharge time is later
                current_row['Flyt_ud'] = max(current_row['Flyt_ud'], row['Flyt_ud'])
            else:
                # No overlap within 24 hours, add the current admission to merged_admissions and start a new current admission
                merged_admissions.append(current_row.to_dict())
                current_row = row
        merged_admissions.append(current_row.to_dict())

        events = []
        for admission in merged_admissions:
            events.append({'subject_id': admission['CPR_hash'], 
                           'admission': admission['Flyt_ind'], 
                           'discharge': admission['Flyt_ud'],
                           'timestamp': admission['Flyt_ind']})
        final_df = pd.DataFrame(events)
        final_df['subject_id'] = final_df['subject_id'].map(subject_id_mapping)
        return final_df

    def patients_info(self):
        """Load patients info and rename columns"""
        self.logger.info("Load patients info")
        df = self.load_pandas(self.cfg.patients_info)
        if self.test:
            df = df.sample(100000)
        df = self.select_columns(df, self.cfg.patients_info)
        self.logger.info(f"Number of patients after selecting columns: {len(df)}")
        df = df.rename(columns={'PID':'subject_id'})

        df['integer_id'], unique_values = pd.factorize(df['subject_id'])
        hash_to_integer_map = dict(zip(unique_values, range(len(unique_values))))

        torch.save(hash_to_integer_map, f'{self.cfg.paths.output_dir}/hash_to_integer_map.pt')

        df['subject_id'] = df['integer_id']
        df = df.drop(columns=['integer_id'])

        self.logger.info(f"Number of patients before saving: {len(df)}")

        # Convert info dict to dataframe
        self.save(df, self.cfg.patients_info, 'subject')
        return hash_to_integer_map

    @staticmethod
    def combine_dataframes(df1, df2):
        """
        Combine two dataframes, removing unnecessary columns from the one within admissions.
        """
        df2 = df2.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END'])
        return pd.concat([df1, df2])
    
    @staticmethod
    def assign_hash(df):
        return df.apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest(), axis=1)

    def change_dtype(self, df, cfg):
        """Change column dtype"""
        if 'dtypes' in cfg:
            for col, dtype in cfg.dtypes.items():
                df[col] = df[col].astype(dtype)
        return df

    def select_columns(self, df, cfg):
        """Select and Rename columns"""
        columns = df.columns.tolist()
        selected_columns = [columns[i] for i in cfg.usecols]
        df = df[selected_columns]
        df = df.rename(columns={old: new for old, new in zip(selected_columns, cfg.names)})
        return df
        
    def load_pandas(self, cfg: dict):
        ds = self.get_dataset(cfg)
        df = ds.to_pandas_dataframe()
        return df
    
    def load_dask(self, cfg: dict):
        ds = self.get_dataset(cfg)
        df = ds.to_dask_dataframe()
        return df

    def load_chunks(self, cfg: dict, pandas=True):
        """Generate chunks of the dataset and convert to pandas/dask df"""
        ds = self.get_dataset(cfg)
        if 'start_chunk' in cfg:
            i = cfg.start_chunk
        else:
            i = 0
        while True:
            self.logger.info(f"chunk {i}")
            chunk = ds.skip(i * cfg.chunksize)
            chunk = chunk.take(cfg.chunksize)
            if pandas:
                df = chunk.to_pandas_dataframe()
            else:
                df = chunk.to_dask_dataframe()
            if len(df.index) == 0:
                self.logger.info("empty")
                break
            i += 1
            yield df
            
    def get_dataset(self, cfg: dict):
        file_path = join(self.dump_path, cfg.filename) if self.dump_path is not None else cfg.filename
        ds = Dataset.Tabular.from_parquet_files(path=(self.datastore,file_path))
        if 'keep_cols' in cfg:
            ds = ds.keep_columns(columns=cfg.keep_cols)
        if self.test:
            ds = ds.take(100000)
        return ds
    
    def save(self, df, cfg, filename, mode='w'):
        self.logger.info(f"Save {filename}")
        out = self.cfg.paths.output_dir
        if 'file_type' in cfg:
            file_type = cfg.file_type
        else:
            file_type = self.cfg.file_type
        if not os.path.exists(out):
            os.makedirs(out)
        if file_type == 'parquet':
            path = os.path.join(out, f'{filename}.parquet')
            df.to_parquet(path)
        elif file_type == 'csv':
            path = os.path.join(out, f'{filename}.csv')
            if mode == 'w':
                df.to_csv(path, index=False, mode=mode)
            else: 
                df.to_csv(path, index=False, mode=mode, header=False)
        else:
            raise ValueError(f"Filetype {file_type} not implemented yet")