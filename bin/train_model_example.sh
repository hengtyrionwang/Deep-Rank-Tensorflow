###
 # @Descripttion: 
 # @version: 
 # @Author: Heng Tyrion Wang
 # @Date: 2022-04-01 09:05:28
 # @LastEditors: Heng Tyrion Wang
 # @Email: hengtyrionwang@gmail.com
 # @LastEditTime: 2022-06-13 15:02:42
### 

path=$(pwd)
model="DeepFM"
log_path=${path}/logs
data_path=${path}"/tfdata"
saved_model_path=${path}"/saved_model"
learning_rate=0.001
batch_size=1000
buffer_size=40000
epochs=100
feature_dim=1000000
embedding_dim=16
field_dim=39
field_sub_dim=1
dropout_rate=0.2
l2_reg=0.0001
hidden_units="[512,512,512]"
version=`date +%Y%m%d`

if [ ! -d ${log_path} ]; then
    mkdir ${log_path}
fi

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
 --version ${version}\
 --hidden_units ${hidden_units}\
 &> ${log_path}/${model}_${version}.log