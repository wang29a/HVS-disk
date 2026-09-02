project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output


cd $project_path/build

make -j
data_name=sift100w #  gist, sift100w, deep, msong
pq_dim=128
# data=${data_path}/${data_name}/${data_name}_base.fbin500000
data=${data_path}/${data_name}/${data_name}_base.fbin
mkdir ${output_path}/${data_name}
mkdir ${output_path}/${data_name}/disk_init_pq$pq_dim
data_type=float
R=32
B=64
M=64
Lbuild=75
metric=l2
single_file_index=0
index_path_prefix=${output_path}/${data_name}/disk_init_pq$pq_dim/${data_name}
#/home/loujh/yq_code/Greator/src/aux_utils.cpp 987行控制pq大小
# 搜索size_t num_pq_chunks =
cmd="./tests/build_disk_index \
    ${data_type}  \
    ${data} \
    ${index_path_prefix} \
    ${R} \
    ${Lbuild} \
    ${B} \
    ${M} \
    64 \
    ${metric} \
    ${single_file_index}
    "
echo "BUILD cmd:"
echo $cmd
eval $cmd
