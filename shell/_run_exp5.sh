project_path=/home/loujh/yq_code/PipeANN-main

cd "$project_path"
data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

log_path=/home/loujh/yq_code/PipeANN-main/log/fig4

for sys in odinann dgai
do

    if [ "$sys" = "dgai" ]; then
        flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK  -DUSE_NHOOD_CACHE"
        L_mem=0
        num_threads=8
        search_mode=3 #0 base, 1 starling, 2 pipeann 3 rerank
        pipeline_width=32
        strategy=63
    else
        flag="-DREAD_ONLY_TESTS -DNO_MAPPING"
        L_mem=10
        num_threads=8
        search_mode=0 
        pipeline_width=4
        strategy=0
    fi

    cd $project_path/build
    export ADDITIONAL_DEFINITIONS="$flag"
    cmake ..
    make -j8

    for strategy in 1 2 4 8 16 32
    do
        for data_name in sift100w msong gist
        do

            # data_name=sift1b # sift, gist, sift100w
            pq_dim=64
            K=10

            # query=${data_path}/${data_name}/${data_name}_query.bin
            # gt=${data_path}/${data_name}/${data_name}_gt_K10.bin
            # data_type=uint8
            query=${data_path}/${data_name}/${data_name}_query.fbin
            gt=${data_path}/${data_name}/${data_name}_gt_0..00125/gt_0.bin 
            data_type=float

            disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim

            if [ "$data_name" = "sift100w" ]; then
                L="10 20 30 40 50 60 70 80 90 100"
            elif [ "$data_name" = "msong" ]; then
                L="10 40 70 100 130 160 190 210 240 270"
            elif [ "$data_name" = "gist" ]; then
                L="50 100 150 200 250 300 350 400 450 500"
            fi

            index_path_prefix=${output_path}/${data_name}/${data_name}

            rm "$output_path"/${data_name}/${data_name}_disk.index.tags
            cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_mem.index "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_mem.index.data "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_mem.index.tags "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_compressed.bin "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_pivots.bin "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_compressed_2.bin "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_pivots_2.bin "$output_path"/${data_name}/
            cp $disk_init_path/disk_index_graph "$output_path"/${data_name}/
            cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
            cp $disk_init_path/dram_index_graph "$output_path"/${data_name}/
            cp $disk_init_path/reordered_disk_index_data "$output_path"/${data_name}/
            cp $disk_init_path/reordered_disk_index_graph "$output_path"/${data_name}/
            cp $disk_init_path/reorder_map_data "$output_path"/${data_name}/
            cp $disk_init_path/reorder_map_graph "$output_path"/${data_name}/


            if [ "$sys" = "odinann" ]; then
                sys='OdinANN'
            elif [ "$sys" = "dgai" ]; then
                sys='DGAI'
            fi

            if [ "$data_name" = "sift1b" ]; then
                data_name='SIFT1B'
            elif [ "$data_name" = "sift100w" ]; then
                data_name='SIFT1M'
            elif [ "$data_name" = "gist" ]; then
                data_name='GIST'
            elif [ "$data_name" = "msong" ]; then
                data_name='MSONG'
            fi

            log_file_path=${log_path}/${sys}_${data_name}_${num_threads}.log

            cmd="${project_path}/build/tests/search_disk_index  \
                ${data_type} \
                ${index_path_prefix} \
                ${num_threads} \
                ${pipeline_width} \
                ${query} \
                ${gt} \
                ${K} \
                l2 \
                ${search_mode} \
                ${L_mem} \
                ${strategy} \
                ${L} 2>&1 | tee ${log_file_path}
                "
            echo ${cmd}
            eval ${cmd}

        done
    done
done
