project_path=/root/code/GreatorPlus

data_path=/root/dataset

output_path=/root/work_folder

log_path=/root/code/GreatorPlus/log/64dim_256

linux_aligned_file_reader=${project_path}/src/utils/linux_aligned_file_reader.cpp
mkdir -p $log_path

K=10
num_threads=8
type=doublepq # error_refine 512 original doublepq
# for strategy in 1 5 65 129 187
# for strategy in 37 59 63
for strategy in 1 # 1 是普通的rerank， 5 rerank+doublePQ， 15 打开全部rerank+doublePQ+reorder
do
    for sys in dgai
    do

        if [ "$sys" = "dgai" ]; then
            flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_NHOOD_CACHE"
            # flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
            L_mem=0
            search_mode=3
            pipeline_width=4
            strategy=$strategy
        else
            flag="-DREAD_ONLY_TESTS -DNO_MAPPING"
            L_mem=0
            search_mode=2
            pipeline_width=32
            strategy=0
        fi

        # cd $project_path/build
        # export ADDITIONAL_DEFINITIONS="$flag"
        # cmake ..
        # make -j8
        # cd "$project_path"

        for data_name in sift100w_shuffled
        # for data_name in sift100w
        do

            # data_name=sift1b # sift, gist, sift100w
            pq_dim=64

            # query=${data_path}/${data_name}/${data_name}_query.bin
            # gt=${data_path}/${data_name}/${data_name}_gt_K10.bin
            # data_type=uint8
            query=${data_path}/${data_name}/${data_name}_query.fbin
            gt=${data_path}/${data_name}/${data_name}_gt_0.00125/gt_0.bin 
            # gt=${data_path}/${data_name}/${data_name}_gt_0.01/gt_0.bin 
            # gt=${data_path}/${data_name}/${data_name}_gt_K10
            data_type=float

            trunc_len=10000
            if [ "$data_name" = "sift100w_shuffled" ] || [ "$data_name" = "sift200w" ]; then
                L="10 20 30 40 50 60 70 80 90 100"
                # L="100 120 140 160 180 200 240 280 300 350 400 450 500"
                # L="400"
                # L="90 91 92 93 94 95 96 97 98 99 100"
                # L="60 61 62 63 64 65 66 67 68 69 70"
                # L="60 62 64 66 68 70 72 74 76 78 80" # recall 0.98
                # L="90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105" # recall 0.98
                # L="100 200 300 400 500" # recall 0.98
                # L="57 58 59 60 61 62"
                # L="300 340 380 420 450 500 550 600 650 700 750 800 850 900 950 1000 1200 1400 1600 1800 2000"
                # L="1000 1200 1400 1600 1800 2000"
                # L="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300"
                trunc_len=48 # trunc_64 = 22, trunc_32 = 48
            elif [ "$data_name" = "msong" ]; then
                L="20 30 40 50 60 70 90 110 135 160 210 250 300 360 450"
                L="215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240" # 99recall
                trunc_len=168
            elif [ "$data_name" = "gist" ]; then
                # L="50 60 70 80 90 100 125 150 175 200 250 300 400 500 600 800 1200 1600"
                L="160 200 230 260 300 360 420 500 600 700 800 900 1100 1300 1500 1700"
                # L="900 1000 1100 1200 1300 1400"
                # L="50 100 150 200 250 300 350 400 450 500 600"
                # if [ "$sys" = "dgai" ]; then
                #     L="1235 1240 1245 1250 1255 1260 1265 1270"  # 99recall
                # elif [ "$sys" = "odinann" ]; then
                #     L="1170 1171 1172 1173 1174 1175 1176 1177 1178 1179 1180 1181"  # 99recall
                # fi
                trunc_len=511
            elif [ "$data_name" = "deep100m" ]; then
                L="100 120 140 160 200 250 300 360 420 500 600 700 900 1100 1500"
                # if [ "$strategy" = 5 ]; then
                # L="81 91 92 105 106 122 123 145 146 177 178 226"
                # L="410 415 420 425 430 435 "
                # L="700"
                # L="200 "
                # if [ "$sys" = "dgai" ]; then
                # L="800 805 810 815 820 825 830 835 840 845 850 855 860 865 870 875 880 885 890 895 900" # 99recall
                # L="420 425 430 435 440 445 450"
                # fi
                trunc_len=123 # trunc_64 = 23, trunc_32 = 123
            elif [ "$data_name" = "deep10m" ]; then
                # L="10 30 70 100 130 160 190 210 240 270 300 350 400"
                L="162 164 166 168 170 172 174 176 178 180 182 184 186 188 190 192 194 196 198 200" # 98recall
                trunc_len=23
            elif [ "$data_name" = "deep1m" ] || [ "$data_name" = "deep50w" ]; then
                # L="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
                # L="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
                L=" 80 81 82 83 84 85 "
                trunc_len=23
            fi
            
            # if [ "$data_name" = "sift100w" ]; then
            #     L="10 15 20 25 30 40 50 60 70 80 90 100"
            #     trunc_len=18
            # elif [ "$data_name" = "msong" ]; then
            #     L="10 30 50 75 100 130 160 190 210 240 300 400"
            #     trunc_len=168
            # elif [ "$data_name" = "gist" ]; then
            #     L="50 100 150 200 250 300 350 400 450 500 550 600 700 800 900 1000"
            #     # L="50 100 150 200 250 300 350 400 450 500 600"
            #     trunc_len=511
            # elif [ "$data_name" = "deep100m" ]; then
            #     L="40 70 100 130 160 200 250 300 350 400 450 500 550 600 650 700"
            #     trunc_len=21
            # elif [ "$data_name" = "deep1m" ] || [ "$data_name" = "deep50w" ]; then
            #     L="40 70 100 130 160 200 250 300 350 400"
            # fi


            sed -i 's/\(trunc_len *= *\)[0-9]\+/\1'"$trunc_len"'/' $linux_aligned_file_reader 
            # exit 0
            cd $project_path/build
            export ADDITIONAL_DEFINITIONS="$flag"
            cmake ..
            make -j8
            cd "$project_path"

            # disk_init_path=${output_path}/${data_name}/disk_init_pq${pq_dim}_error_refine
            disk_init_path=${output_path}/${data_name}/disk_init_pq${pq_dim}_${type}
            # disk_init_path=${output_path}/${data_name}/disk_init_pq${pq_dim}_original

            index_path_prefix=${output_path}/${data_name}/${data_name}

            rm "$output_path"/${data_name}/${data_name}_disk.index.tags
            cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
            if [ "$sys" = "odinann" ]; then
                cp $disk_init_path/${data_name}_mem.index "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_mem.index.data "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_mem.index.tags "$output_path"/${data_name}/
            fi
            cp $disk_init_path/${data_name}_pq_compressed.bin "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_pivots.bin "$output_path"/${data_name}/

            cp $disk_init_path/disk_index_graph "$output_path"/${data_name}/
            cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
            # cp $disk_init_path/dram_index_graph "$output_path"/${data_name}/
            if [ "$sys" = "dgai" ]; then
                cp $disk_init_path/${data_name}_pq_compressed_2.bin "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_pq_pivots_2.bin "$output_path"/${data_name}/
                # cp $disk_init_path/${data_name}_pq_compressed_3.bin "$output_path"/${data_name}/
                # cp $disk_init_path/${data_name}_pq_pivots_3.bin "$output_path"/${data_name}/
                # cp $disk_init_path/${data_name}_pq_compressed_4.bin "$output_path"/${data_name}/
                # cp $disk_init_path/${data_name}_pq_pivots_4.bin "$output_path"/${data_name}/
                # cp $disk_init_path/${data_name}_pq_compressed_5.bin "$output_path"/${data_name}/
                # cp $disk_init_path/${data_name}_pq_pivots_5.bin "$output_path"/${data_name}/
                # cp $disk_init_path/reordered_disk_index_data "$output_path"/${data_name}/
                cp $disk_init_path/reordered_disk_index_data_2 "$output_path"/${data_name}/
                if [ "$data_name" = "gist" ]; then
                    cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
                fi

                cp $disk_init_path/reordered_disk_index_graph_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reordered_disk_index_graph "$output_path"/${data_name}/
                # cp $disk_init_path/reorder_map_data "$output_path"/${data_name}/
                cp $disk_init_path/reorder_map_data_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reorder_map_graph "$output_path"/${data_name}/
                cp $disk_init_path/reorder_map_graph_2 "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_map.bin "$output_path"/${data_name}/
            fi
            # exit 0

            if [ "$sys" = "odinann" ]; then
                sys_name='OdinANN'
            elif [ "$sys" = "dgai" ]; then
                sys_name='DGAI'
            fi

            if [ "$data_name" = "sift1b" ]; then
                dataset_name='SIFT1B'
            elif [ "$data_name" = "sift100w" ]; then
                dataset_name='SIFT1M'
            elif [ "$data_name" = "gist" ]; then
                dataset_name='GIST'
            elif [ "$data_name" = "msong" ]; then
                dataset_name='MSONG'
            elif [ "$data_name" = "deep100m" ]; then
                dataset_name='DEEP100M'
            elif [ "$data_name" = "deep10m" ]; then
                dataset_name='DEEP10M'
            elif [ "$data_name" = "deep1m" ]; then
                dataset_name='DEEP1M'
            fi

            # log_file_path=${log_path}/_.log
            # log_file_path=${log_path}/${data_name}_${pq_dim}dim_rerank_doublePQ.log
            # log_file_path=${log_path}/${data_name}_${pq_dim}dim_rerank_doublePQ_reorder.log
            log_file_path=${log_path}/${data_name}_${pq_dim}dim_${strategy}_${type}_tau.log
            # log_file_path=${log_path}/${data_name}_${pq_dim}dim_${strategy}_original.log
            # log_file_path=${log_path}/${data_name}_${pq_dim}dim_${strategy}_error.log

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
