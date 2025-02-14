python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/none \
  --input_json o/evaluation-single/pubchem-else-rdkit/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/pubchem-else-rdkit_none.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smote \
  --input_json o/evaluation-single/pubchem-else-rdkit/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/pubchem-else-rdkit_smote.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smoteenn \
  --input_json o/evaluation-single/pubchem-else-rdkit/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/pubchem-else-rdkit_smoteenn.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smotetomek \
  --input_json o/evaluation-single/pubchem-else-rdkit/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/pubchem-else-rdkit_smotetomek.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/none \
  --input_json o/evaluation-single/pubchem-else-rdkit/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/knn_pubchem-else-rdkit_none.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smote \
  --input_json o/evaluation-single/pubchem-else-rdkit/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/knn_pubchem-else-rdkit_smote.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smoteenn \
  --input_json o/evaluation-single/pubchem-else-rdkit/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/knn_pubchem-else-rdkit_smoteenn.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smotetomek \
  --input_json o/evaluation-single/pubchem-else-rdkit/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/knn_pubchem-else-rdkit_smotetomek.csv \
  --use_knn y \
  && \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n