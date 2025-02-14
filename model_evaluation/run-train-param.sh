python3 src/test/train.rdkitonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/test/train.rdkitonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/test/train.rdkitonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/test/train.rdkitonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/rdkit-only/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/test/train.pubchemonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/test/train.pubchemonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/test/train.pubchemonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/test/train.pubchemonly.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-only/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/test/train.avg.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/test/train.avg.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/test/train.avg.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/test/train.avg.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/avg/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/test/train.both.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/test/train.both.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/test/train.both.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/test/train.both.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/both/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n \
  && \
python3 src/test/train.pubchemelserdkit.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-else-rdkit/none \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 1 \
  --use_gpu n \
&& \
python3 src/test/train.pubchemelserdkit.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-else-rdkit/smote \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n \
&& \
python3 src/test/train.pubchemelserdkit.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-else-rdkit/smoteenn \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 3 \
  --use_gpu n \
&& \
python3 src/test/train.pubchemelserdkit.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/evaluation-single/pubchem-else-rdkit/smotetomek \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 4 \
  --use_gpu n