cd build
make 
cd ..

data_name=sift100w
data_type=float
pq_dim=64
output_path=/gauss_yusong/ljh_data/freshdiskann_output$data_name

index_path=$output_path/$data_name"_disk.index"
topo_path=$output_path/dram_index_graph
coord_path=$output_path/disk_index_data
# topo_path=$output_path/$data_name"_memory_topo.index"
# coord_path=$output_path/$data_name"_disk_coord.index"

/home/loujh/yq_code/DiskANN_Baseline/tool/build/split_index $index_path $topo_path $coord_path $data_type
