python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/none \
  --input_json o/evaluation-single/pubchem-else-rdkit/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-else-rdkit_none.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smote \
  --input_json o/evaluation-single/pubchem-else-rdkit/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-else-rdkit_smote.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smoteenn \
  --input_json o/evaluation-single/pubchem-else-rdkit/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-else-rdkit_smoteenn.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smotetomek \
  --input_json o/evaluation-single/pubchem-else-rdkit/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-else-rdkit_smotetomek.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/none \
  --input_json o/evaluation-single/pubchem-else-rdkit/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-else-rdkit_none.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smote \
  --input_json o/evaluation-single/pubchem-else-rdkit/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-else-rdkit_smote.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smoteenn \
  --input_json o/evaluation-single/pubchem-else-rdkit/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-else-rdkit_smoteenn.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-else-rdkit/smotetomek \
  --input_json o/evaluation-single/pubchem-else-rdkit/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-else-rdkit_smotetomek.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/none \
  --input_json o/evaluation-single/rdkit-only/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_rdkit-only_none.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/smote \
  --input_json o/evaluation-single/rdkit-only/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_rdkit-only_smote.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/smoteenn \
  --input_json o/evaluation-single/rdkit-only/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_rdkit-only_smoteenn.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/smotetomek \
  --input_json o/evaluation-single/rdkit-only/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_rdkit-only_smotetomek.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/none \
  --input_json o/evaluation-single/rdkit-only/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_rdkit-only_none.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/smote \
  --input_json o/evaluation-single/rdkit-only/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_rdkit-only_smote.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/smoteenn \
  --input_json o/evaluation-single/rdkit-only/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_rdkit-only_smoteenn.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/rdkit-only/smotetomek \
  --input_json o/evaluation-single/rdkit-only/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_rdkit-only_smotetomek.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/none \
  --input_json o/evaluation-single/pubchem-only/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-only_none.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/smote \
  --input_json o/evaluation-single/pubchem-only/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-only_smote.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/smoteenn \
  --input_json o/evaluation-single/pubchem-only/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-only_smoteenn.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/smotetomek \
  --input_json o/evaluation-single/pubchem-only/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_pubchem-only_smotetomek.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/none \
  --input_json o/evaluation-single/pubchem-only/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-only_none.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/smote \
  --input_json o/evaluation-single/pubchem-only/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-only_smote.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/smoteenn \
  --input_json o/evaluation-single/pubchem-only/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-only_smoteenn.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/pubchem-only/smotetomek \
  --input_json o/evaluation-single/pubchem-only/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_pubchem-only_smotetomek.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/none \
  --input_json o/evaluation-single/avg/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_avg_none.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/smote \
  --input_json o/evaluation-single/avg/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_avg_smote.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/smoteenn \
  --input_json o/evaluation-single/avg/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_avg_smoteenn.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/smotetomek \
  --input_json o/evaluation-single/avg/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_avg_smotetomek.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/none \
  --input_json o/evaluation-single/avg/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_avg_none.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/smote \
  --input_json o/evaluation-single/avg/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_avg_smote.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/smoteenn \
  --input_json o/evaluation-single/avg/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_avg_smoteenn.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/avg/smotetomek \
  --input_json o/evaluation-single/avg/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_avg_smotetomek.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/none \
  --input_json o/evaluation-single/both/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_both_none.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/smote \
  --input_json o/evaluation-single/both/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_both_smote.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/smoteenn \
  --input_json o/evaluation-single/both/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_both_smoteenn.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/smotetomek \
  --input_json o/evaluation-single/both/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_both_smotetomek.csv \
  --use_knn n \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/none \
  --input_json o/evaluation-single/both/none/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_both_none.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/smote \
  --input_json o/evaluation-single/both/smote/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_both_smote.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/smoteenn \
  --input_json o/evaluation-single/both/smoteenn/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_both_smoteenn.csv \
  --use_knn y \
  && \
python3 src/predict.py \
  --model_dir o/evaluation-single/both/smotetomek \
  --input_json o/evaluation-single/both/smotetomek/validation_data_original.json \
  --output_csv predictions/evaluation-single/rdkit-only-pred_knn_both_smotetomek.csv \
  --use_knn y 