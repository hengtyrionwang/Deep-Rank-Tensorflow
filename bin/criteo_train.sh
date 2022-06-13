###
 # @Descripttion: 
 # @version: 
 # @Author: Heng Tyrion Wang
 # @Date: 2022-06-01 09:05:28
 # @LastEditors: Heng Tyrion Wang
 # @Email: hengtyrionwang@gmail.com
 # @LastEditTime: 2022-06-13 14:54:58
### 

path=$(pwd)
#model="DeepFM"
log_path=${path}/logs/
data_path=${path}"/tfdata"
saved_model_path=${path}"/saved_model"
learning_rate=0.001
batch_size=10000
buffer_size=30000
epochs=100
feature_dim=1000000
embedding_dim=32
field_dim=39
field_sub_dim=1
dropout_rate=0.5
l2_reg=0.00001
decay_steps=3668
decay_rate=0.3
num_cross=4
hidden_units="[1024,1024,1024]"
version=`date +%Y%m%d`
verbose=1

if [ ! -d ${log_path} ]; then
    mkdir ${log_path}
fi

model_type=( "DeepCross" "DeepCrossV2" "DeepCrossMix" "AutoInt" "DeepFM" "NFM" "FM" "LR")

for model in ${model_type[*]}
do
    python ${path}/main.py\
     --model_name ${model}\
     --data_path ${data_path}\
     --saved_model_path ${saved_model_path}\
     --learning_rate ${learning_rate}\
     --batch_size ${batch_size}\
     --epochs ${epochs}\
     --feature_dim ${feature_dim}\
     --embedding_dim ${embedding_dim}\
     --field_dim ${field_dim}\
     --field_sub_dim ${field_sub_dim}\
     --dropout_rate ${dropout_rate}\
     --l2_reg ${l2_reg}\
     --decay_steps=${decay_steps}\
     --decay_rate=${decay_rate}\
     --version ${version}\
     --hidden_units ${hidden_units}\
     --num_cross ${num_cross}\
     --verbose ${verbose}\
     &> ${log_path}/${model}_${version}.log
done