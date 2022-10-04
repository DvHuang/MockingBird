echo "step: $1"
dataset_root=/data/dataset_root/

case $1 in

syn_pre)
    nohup python pre.py ${dataset_root} -d data_aishell -n 8 >/data/syn_pre.log 2>&1 &
    ;;
syn_train)
    export CUDA_VISIBLE_DEVICES=1
    nohup python synthesizer_train.py guanchun ${dataset_root}/SV2TTS/synthesizer >/data/syn.train.log 2>&1 &
    ;;
scp_data)
    scp -r ./* works@52.130.89.3://home/works/hdavid/data/dataset_root/data_aishell/
    ;;
enc_pre)
    nohup python encoder_preprocess.py ${dataset_root} -d aidatatang_200zh >/data/enc_pre.log 2>&1 & 
    ;;
enc_train)
    nohup python encoder_train.py my_run ${dataset_root}/SV2TTS/encoder --no_visdom >/data/enc_train.log 2>&1 &
    ;;
voc_pre)
    # 使用enc_pre 生成的mel spectrograms, the wavs 作为训练数据
    #nohup python vocoder_preprocess.py ${dataset_root} --cpu -m /data/app/synthesizer/saved_models/85k >/data/voc_pre.log 2>&1 &
    nohup python vocoder_preprocess.py ${dataset_root}  -m /data/app/synthesizer/saved_models/guanchun_85k >/data/voc_pre.log 2>&1 &
    ;;
voc_train_wavernn)
    export CUDA_VISIBLE_DEVICES=1
    python vocoder_train.py guanchun_wavernn ${dataset_root} wavernn >/data/voc_train_wavernn.log 2>&1 &
    ;;
voc_train_hifigan)
    python vocoder_train.py guanchun_hifigan ${dataset_root} hifigan >/data/voc_train_hifigan.log 2>&1 &
    ;;
esac



