echo "step: $1"
dataset_root=/data/dataset_root_more/
task_name=laiye_pretrain

case $1 in

enc_pre)
    nohup python encoder_preprocess.py ${dataset_root} -d aidatatang_200zh >${dataset_root}/enc_pre.log 2>&1 & 
    ;;
enc_train)
    nohup python encoder_train.py ${task_name}  ${dataset_root}/SV2TTS/encoder --no_visdom >${dataset_root}/enc_train.log 2>&1 &
    ;;
syn_pre)
    nohup python pre.py ${dataset_root} -d aidatatang_200h -n 8 >${dataset_root}/syn_pre.log 2>&1 &
    ;;
syn_train)
    export CUDA_VISIBLE_DEVICES=1
    nohup python synthesizer_train.py  ${task_name}  ${dataset_root}/SV2TTS/synthesizer >${dataset_root}/syn.train.log 2>&1 &
    ;;
scp_data)
    scp -r ./* works@52.130.89.3://home/works/hdavid/data/dataset_root/data_aishell/
    ;;
voc_pre)
    # 使用enc_pre 生成的mel spectrograms, the wavs 作为训练数据
    #nohup python vocoder_preprocess.py ${dataset_root} --cpu -m /data/app/synthesizer/saved_models/85k >/data/voc_pre.log 2>&1 &
    nohup python vocoder_preprocess.py ${dataset_root}  -m /data/app/synthesizer/saved_models/guanchun_85k >${dataset_root}/voc_pre.log 2>&1 &
    ;;
voc_train_wavernn)
    export CUDA_VISIBLE_DEVICES=1
    python vocoder_train.py  ${task_name}_wavernn ${dataset_root} wavernn >${dataset_root}/voc_train_wavernn.log 2>&1 &
    ;;
voc_train_hifigan)
    python vocoder_train.py ${task_name}_hifigan ${dataset_root} hifigan >${dataset_root}/voc_train_hifigan.log 2>&1 &
    ;;
esac



