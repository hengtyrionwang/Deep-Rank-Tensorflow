###
 # @Descripttion: 
 # @version: 
 # @Author: Heng Tyrion Wang
 # @Date: 2022-04-01 10:08:05
 # @LastEditors: Heng Tyrion Wang
 # @Email: hengtyrionwang@gmail.com
 # @LastEditTime: 2022-04-01 10:45:17
### 

path=$(pwd)
tfdata_path=${path}/tfdata
version=`date +%Y%m%d`

if [ ! -d ${tfdata_path} ]; then
    mkdir ${tfdata_path}
else
    rm -rf ${tfdata_path}/*
fi

data_type=("train" "test" "valid")

for type in ${data_type[*]}
do
    python ${path}/script/converter.py ${path}/data/${type}.txt ${path}/tfdata/${type}.tfrecord
done
