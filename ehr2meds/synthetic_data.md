python ehr2meds/generate_synthetic_raw_data.py --config synthetic_generation/fetal_SP.yaml
python ehr2meds/convert_raw_to_premeds.py --config preMEDS/fetal_SP.yaml     

bash ehr2meds/convert_premeds_to_meds.sh \
  ehr2meds/data/preMEDS/fetal_data/SP \
  ehr2meds/configs/MEDS/default_pipeline.yaml \
  ehr2meds/configs/MEDS/default_event.yaml \
  ehr2meds/data/MEDS/SP