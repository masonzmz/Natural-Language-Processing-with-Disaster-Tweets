###
 # @Author       : Mingzhe Zhang (s4566656)
 # @Date         : 2022-09-29 17:11:53
 # @LastEditors  : Mingzhe Zhang (s4566656)
 # @LastEditTime : 2022-09-30 12:02:05
 # @FilePath     : /s4566656/anaconda3/envs/mason/Kaggle_Disaster/src/run.sh
 # @Description  : Tuning parameters [pre-trained model, threshold, batch size, dropout, number of layers]. You
 # may need to change the file path. Run this sh file may take a long time (about 105hrs).
###


for model_name in "bert_base" "roberta_base"; do
    for t in 0.5 0.6 0.7; do
        for bs in 4 8 16 32; do
            for dp in 0.0 0.1 0.2 0.3 0.4 0.5; do
                for l in 1 2 3; do
                    python3 "anaconda3/envs/mason/Kaggle_Disaster/src/train.py" \
                    --model_name $model_name \
                    --threshold $t \
                    --batchsize $bs \
                    --dropout $dp \
                    --layer $l 
                done
            done
        done
    done
done
