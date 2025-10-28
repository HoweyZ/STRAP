# Energy-Wind, the same for the other two datasets.

#--------------------------------------------------------------Retrain------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 24 --logname "retrain_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 622 --logname "retrain_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/retrain.json --gpuid 2 --seed 100 --logname "retrain_st" --backbone "stgnn"

#--------------------------------------------------------------Pretrain------------------------------------------------------------------------------#

python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st-24/0/3.0127.pkl" --gpuid 2 --seed 24 --logname "pre_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st-100/0/2.0238.pkl" --gpuid 2 --seed 100 --logname "pre_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st-622/0/3.0058.pkl" --gpuid 2 --seed 622 --logname "pre_st" --backbone "stgnn"


python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_t-24/0/2.7608.pkl" --gpuid 2 --seed 24 --logname "pre_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_t-100/0/2.9586.pkl" --gpuid 2 --seed 100 --logname "pre_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_t-622/0/2.5731.pkl" --gpuid 2 --seed 622 --logname "pre_t" --backbone "tgcn"


python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_dc-24/0/2.6751.pkl" --gpuid 2 --seed 24 --logname "pre_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_dc-100/0/2.7599.pkl" --gpuid 2 --seed 100 --logname "pre_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_dc-622/0/2.5705.pkl" --gpuid 2 --seed 622 --logname "pre_dc" --backbone "dcrnn"


python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_ast-24/0/2.9481.pkl" --gpuid 2 --seed 24 --logname "pre_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_ast-100/0/2.8455.pkl" --gpuid 2 --seed 100 --logname "pre_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pre.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_ast-622/0/3.085.pkl" --gpuid 2 --seed 622 --logname "pre_ast" --backbone "astgnn"


python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st-24/2016/18.7955.pkl" --gpuid 2 --seed 24 --logname "pre_st" --backbone "stgnn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st-100/2016/19.8465.pkl" --gpuid 2 --seed 100 --logname "pre_st" --backbone "stgnn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st-622/2016/15.9219.pkl" --gpuid 2 --seed 622 --logname "pre_st" --backbone "stgnn"


python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_t-24/2016/19.7186.pkl" --gpuid 2 --seed 24 --logname "pre_t" --backbone "tgcn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_t-100/2016/18.6676.pkl" --gpuid 2 --seed 100 --logname "pre_t" --backbone "tgcn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_t-622/2016/18.9347.pkl" --gpuid 2 --seed 622 --logname "pre_t" --backbone "tgcn"


python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_dc-24/2016/17.1791.pkl" --gpuid 2 --seed 24 --logname "pre_dc" --backbone "dcrnn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_dc-100/2016/18.4592.pkl" --gpuid 2 --seed 100 --logname "pre_dc" --backbone "dcrnn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_dc-622/2016/17.8702.pkl" --gpuid 2 --seed 622 --logname "pre_dc" --backbone "dcrnn"


python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_ast-24/2016/20.4989.pkl" --gpuid 2 --seed 24 --logname "pre_ast" --backbone "astgnn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_ast-100/2016/19.3166.pkl" --gpuid 2 --seed 100 --logname "pre_ast" --backbone "astgnn"
python main.py --conf conf/AIR/pre.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_ast-622/2016/20.1651.pkl" --gpuid 2 --seed 622 --logname "pre_ast" --backbone "astgnn"




python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st-24/2011/16.3006.pkl" --gpuid 2 --seed 24 --logname "pre_st" --backbone "stgnn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st-100/2011/16.915.pkl" --gpuid 2 --seed 100 --logname "pre_st" --backbone "stgnn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st-622/2011/16.6973.pkl" --gpuid 2 --seed 622 --logname "pre_st" --backbone "stgnn"


python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_t-24/2011/17.0331.pkl" --gpuid 2 --seed 24 --logname "pre_t" --backbone "tgcn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_t-100/2011/16.2856.pkl" --gpuid 2 --seed 100 --logname "pre_t" --backbone "tgcn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_t-622/2011/17.0128.pkl" --gpuid 2 --seed 622 --logname "pre_t" --backbone "tgcn"


python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_dc-24/2011/17.1.pkl" --gpuid 2 --seed 24 --logname "pre_dc" --backbone "dcrnn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_dc-100/2011/17.401.pkl" --gpuid 2 --seed 100 --logname "pre_dc" --backbone "dcrnn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_dc-622/2011/17.78.pkl" --gpuid 2 --seed 622 --logname "pre_dc" --backbone "dcrnn"


python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_ast-24/2011/16.966.pkl" --gpuid 2 --seed 24 --logname "pre_ast" --backbone "astgnn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_ast-100/2011/16.9295.pkl" --gpuid 2 --seed 100 --logname "pre_ast" --backbone "astgnn"
python main.py --conf conf/PEMS/pre.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_ast-622/2011/16.917.pkl" --gpuid 2 --seed 622 --logname "pre_ast" --backbone "astgnn"



#--------------------------------------------------------------Trafficstream------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 24 --logname "trafficstream_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 622 --logname "trafficstream_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 2 --seed 100 --logname "trafficstream_st" --backbone "stgnn"

#--------------------------------------------------------------ST-LoRA------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 24 --logname "stlora_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 622 --logname "stlora_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stlora.json --gpuid 2 --seed 100 --logname "stlora_st" --backbone "stgnn"

#--------------------------------------------------------------stkec------------------------------------------------------------------------------#
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "tgcn" --logname "stkec_t"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "tgcn" --logname "stkec_t"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "tgcn" --logname "stkec_t"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "astgnn" --logname "stkec_ast"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "astgnn" --logname "stkec_ast"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "astgnn" --logname "stkec_ast"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "dcrnn" --logname "stkec_dc"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "dcrnn" --logname "stkec_dc"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "dcrnn" --logname "stkec_dc"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "stgnn" --logname "stkec_st"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "stgnn" --logname "stkec_st"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "stgnn" --logname "stkec_st"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "tgcn" --logname "stkec_t"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "tgcn" --logname "stkec_t"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "tgcn" --logname "stkec_t"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "astgnn" --logname "stkec_ast"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "astgnn" --logname "stkec_ast"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "astgnn" --logname "stkec_ast"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "dcrnn" --logname "stkec_dc"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "dcrnn" --logname "stkec_dc"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "dcrnn" --logname "stkec_dc"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "stgnn" --logname "stkec_st"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "stgnn" --logname "stkec_st"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "stgnn" --logname "stkec_st"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "tgcn" --logname "stkec_t"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "tgcn" --logname "stkec_t"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "tgcn" --logname "stkec_t"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "astgnn" --logname "stkec_ast"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "astgnn" --logname "stkec_ast"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "astgnn" --logname "stkec_ast"


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "dcrnn" --logname "stkec_dc"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "dcrnn" --logname "stkec_dc"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "dcrnn" --logname "stkec_dc"

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 24 --backbone "stgnn" --logname "stkec_st"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 622 --backbone "stgnn" --logname "stkec_st"
python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 2 --seed 100 --backbone "stgnn" --logname "stkec_st"

#-------------------------------------------------------------EAC----------------------------------------------------------------------------------#
#----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 24 --logname "eac_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 622 --logname "eac_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 2 --seed 100 --logname "eac_st" --backbone "stgnn"

#--------------------------------------------------------------EWC------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 24 --logname "ewc_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 622 --logname "ewc_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/ewc.json --gpuid 2 --seed 100 --logname "ewc_st" --backbone "stgnn"

#--------------------------------------------------------------Replay------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 24 --logname "replay_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 622 --logname "replay_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/replay.json --gpuid 2 --seed 100 --logname "replay_st" --backbone "stgnn"

#--------------------------------------------------------------stadapter------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 24 --logname "stadapter_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 622 --logname "stadapter_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/stadapter.json --gpuid 2 --seed 100 --logname "stadapter_st" --backbone "stgnn"

#--------------------------------------------------------------Graphpro------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 24 --logname "graphpro_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 622 --logname "graphpro_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/graphpro.json --gpuid 2 --seed 100 --logname "graphpro_st" --backbone "stgnn"

#--------------------------------------------------------------pecpm------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 24 --logname "pecpm_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 622 --logname "pecpm_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/pecpm.json --gpuid 2 --seed 100 --logname "pecpm_st" --backbone "stgnn"


#--------------------------------------------------------------STRAP------------------------------------------------------------------------------#
#-----------------------TGCN--------------------------#
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_t" --backbone "tgcn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_t" --backbone "tgcn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_t" --backbone "tgcn"

#-----------------------ASTGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_ast" --backbone "astgnn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_ast" --backbone "astgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_ast" --backbone "astgnn"

#-----------------------DCRNN--------------------------#
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_dc" --backbone "dcrnn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_dc" --backbone "dcrnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_dc" --backbone "dcrnn"

#-----------------------STGNN--------------------------#
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 24 --logname "rap_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 622 --logname "rap_st" --backbone "stgnn"

python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_st" --backbone "stgnn"
python main.py --conf conf/ENERGY-Wind/rap.json --gpuid 2 --seed 100 --logname "rap_st" --backbone "stgnn"
