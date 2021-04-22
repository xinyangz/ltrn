PRETRAINED=bert-base-uncased
DATA_DIR=/path/to/parent/dir/of/books
DATASET=books
EXP_NAME=joint-train-books
# if more than one GPU given, first GPU will be for GNN, the rest for BERT
GPU=0,1,2,3
OUTPUT_DIR=/path/to/output

GPU0=$(echo $GPU | awk -F ',' '{print $1}')

# generate initial bert embedding
if [ ! -f ${DATA_DIR}/${DATASET}/train.emb.tsv.gz ]; then
    echo "Generate initial BERT embedding for train dataset!"
    CUDA_VISIBLE_DEVICES=$GPU0 python bert_unsup_embedding.py \
      $DATA_DIR $DATASET $PRETRAINED train
fi
if [ ! -f ${DATA_DIR}/${DATASET}/dev.emb.tsv.gz ]; then
    echo "Generate initial BERT embedding for dev dataset!"
    CUDA_VISIBLE_DEVICES=$GPU0 python bert_unsup_embedding.py \
      $DATA_DIR $DATASET $PRETRAINED dev
fi

python joint_training.py \
 --exp_name $EXP_NAME \
 --gpu $GPU \
 --output_dir $OUTPUT_DIR \
 --master_port 10040 \
 --data_dir $DATA_DIR \
 --dataset $DATASET \
 --bert_model_name_or_path $PRETRAINED \
 --bert_max_steps 400 \
 --bert_eval_steps 200 \
 --gnn_max_steps 300 \
 --gnn_eval_steps 20 \
 --overwrite_output_dir \
 --topk 500 \
 --conf_threshold_text 0.9 \
 --conf_threshold_graph 0.95 \
 --cotrain_iter 3

