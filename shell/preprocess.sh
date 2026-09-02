# data_name=sift100w
# data_name=msong
# data_name=gist
for type in doublepq  #error_refine 512  original
do
    aux_utils_cpp_path=$PROJECT_PATH/src/utils/aux_utils.cpp
    
    if [ "$type" = "doublepq" ]; then
        sed -i 's/int type = [0-9];/int type = 1;/' $aux_utils_cpp_path
    elif [ "$type" = "error_refine" ]; then
        sed -i 's/int type = [0-9];/int type = 2;/' $aux_utils_cpp_path
    elif [ "$type" = "512" ]; then
        sed -i 's/int type = [0-9];/int type = 3;/' $aux_utils_cpp_path
    else
        sed -i 's/int type = [0-9];/int type = 0;/' $aux_utils_cpp_path
    fi

    for data_name in deep100m
    do
        npts=100000000
        project_path=/root/code/GreatorPlus

        batch_npts=80000000
        pq_dim=64
        ndim=512

        if [ "$data_name" = "sift100w" ] || ["$data_name" = "sift200w"]; then
            ndim=512
        elif [ "$data_name" = "msong" ]; then
            ndim=1680
        elif [ "$data_name" = "gist" ]; then
            ndim=3840
        elif [ "$data_name" = "deep100m" ]; then
            ndim=384
        fi
        data_path=/root/dataset
        output_path=/root/index_ljh/$data_name

        full_data=${data_path}/${data_name}/${data_name}_base.fbin
        query=${data_path}/${data_name}/${data_name}_query.fbin
        topk=10000
        gt=${data_path}/${data_name}/${data_name}_gt_K${topk}
        target_topk=10
        data_type=float

        cd $project_path/build
        cmake  ..
        make -j
        cd $project_path

        # ${project_path}/build/tests/utils/compute_groundtruth $data_type $full_data $query $topk $gt
        # for update_ratio in 0.001 0.002 0.004 0.008 0.01 0.02 0.04 0.08
        for update_ratio in 0.00125
        do
            truthset_prefix=${data_path}/${data_name}/${data_name}_gt_${update_ratio}
            ${project_path}/build/tests/gt_update $gt $npts $batch_npts $target_topk $truthset_prefix $update_ratio 0
        done
        # continue
        ${project_path}/build/tests/change_pts $data_type $full_data $batch_npts
        # exit 0
        disk_init_path=${output_path}/disk_init_pq${pq_dim}_${type}
        # disk_init_path=${output_path}/disk_init_pq${pq_dim}
        # disk_init_path=${output_path}/disk_init_pq${pq_dim}_doublepq
        # disk_init_path=${output_path}/disk_init_pq${pq_dim}_error_refine
        mkdir ${output_path}
        mkdir ${disk_init_path}
        data=$full_data$batch_npts
        index_path_prefix=${disk_init_path}/${data_name}

        $project_path/build/tests/utils/gen_random_slice $data_type $data ${index_path_prefix}_SAMPLE_RATE_0.01 0.01

        $project_path/build/tests/build_memory_index $data_type ${index_path_prefix}_SAMPLE_RATE_0.01_data.bin ${index_path_prefix}_SAMPLE_RATE_0.01_ids.bin ${index_path_prefix}_mem.index 0 0 32 64 1.2 24 l2
        continue

        R=32
        M=64
        Lbuild=75
        metric=l2
        single_file_index=0
        #/home/loujh/yq_code/Greator/src/aux_utils.cpp 987行控制pq大小
        # 搜索size_t num_pq_chunks =
        cmd="$project_path/build/tests/build_disk_index \
            ${data_type}  \
            ${data} \
            ${index_path_prefix} \
            ${R} \
            ${Lbuild} \
            ${pq_dim} \
            ${M} \
            32 \
            ${metric} \
            ${single_file_index}
            "
        echo "BUILD cmd:"
        echo $cmd
        eval $cmd


        index_path=$disk_init_path/$data_name"_disk.index"
        topo_path=$disk_init_path/dram_index_graph
        aligned_topo_path=$disk_init_path/disk_index_graph
        coord_path=$disk_init_path/disk_index_data

        $project_path/build/tests/split_index $index_path $topo_path $aligned_topo_path $coord_path $data_type
        
        $project_path/build/tests/reorder_by_map $batch_npts 132 $aligned_topo_path $disk_init_path/reorder_map_graph_2 $disk_init_path/reordered_disk_index_graph_2
        $project_path/build/tests/reorder_by_map $batch_npts $ndim $coord_path $disk_init_path/reorder_map_data_2 $disk_init_path/reordered_disk_index_data_2
        
    done
done