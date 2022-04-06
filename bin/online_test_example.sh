###
 # @Descripttion: 
 # @version: 
 # @Author: Heng Tyrion Wang
 # @Date: 2022-04-01 10:26:13
 # @LastEditors: Heng Tyrion Wang
 # @Email: hengtyrionwang@gmail.com
 # @LastEditTime: 2022-04-01 10:27:52
### 

path=$(pwd)
model="DeepFM"
batch_size=50

python ${path}/script/online_test.py ${path}/data/test_example.txt ${batch_size} ${model}
