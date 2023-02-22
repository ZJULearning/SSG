#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
MAX_THREADS=`nproc --all`
THREAD=(1)
#THREAD=(${MAX_THREADS})
K=(10)
L_SIZE=(33) #(100 110 120 130 140 150 160 170 180 190)
#L_SIZE=(11 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 250 300 350 400 450 500)

ssg_sift1M() {
  # Build a proximity graph
  if [ ! -f "sift1M.ssg" ]; then
    echo "Converting sift1M_200nn.graph kNN graph to sift1M.ssg"
    if [ -f "sift1M_200nn.graph" ]; then
      ./test_ssg_index sift1M/sift_base.fvecs sift1M_200nn.graph 100 50 60 sift1M.ssg > sift1M_index_${TIME}.log
    else
      echo "ERROR: sift1M_200nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (sift1M_L${l}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.ssg ${1} ${2} sift1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift1M/sift_groundtruth.ivecs 0.25 512 ${4} 2> sift1M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_sift10M() {
  # Build a proximity graph
  if [ ! -f "sift10M.ssg" ]; then
    echo "Converting sift10M_200nn.graph kNN graph to sift10M.ssg"
    if [ -f "sift10M_200nn.graph" ]; then
      ./test_ssg_index sift10M/sift10m_base.fvecs sift10M_200nn.graph 100 50 60 sift10M.ssg > sift10M_index_${TIME}.log
    else
      echo "ERROR: sift10M_200nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (sift10M_L${l}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS sift10M/sift10m_base.fvecs sift10M/sift10m_query.fvecs sift10M.ssg ${1} ${2} sift10M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift10M/sift10m_groundtruth.ivecs 0.2 512 ${4} 2> sift10M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_gist1M() {
  # Build a proximity graph
  if [ ! -f "gist1M.ssg" ]; then
    echo "Converting gist1M_400nn.graph kNN graph to gist1M.ssg"
    if [ -f "gist1M_400nn.graph" ]; then
      ./test_ssg_index gist1M/gist_base.fvecs gist1M_400nn.graph 500 70 60 gist1M.ssg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist1M_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (gist1M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.ssg ${1} ${2} gist1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    gist1M/gist_groundtruth.ivecs 0.3 1024 ${4} 2> gist1M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_crawl() {
  # Build a proximity graph
  if [ ! -f "crawl.ssg" ]; then
    echo "Converting crawl_400nn.graph kNN graph to crawl.ssg"
    if [ -f "crawl_400nn.graph" ]; then
      ./test_ssg_index crawl/crawl_base.fvecs crawl_400nn.graph 500 40 60 crawl.ssg > crawl_index_${TIME}.log
    else
      echo "ERROR: crawl_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (crawl_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.ssg ${1} ${2} crawl_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    crawl/crawl_groundtruth.ivecs 0.3 512 ${4} 2> crawl_search_L${1}K${2}_${3}_T${4}.log
}

ssg_glove-100() {
  # Build a proximity graph
  if [ ! -f "glove-100.ssg" ]; then
    echo "Converting glove-100_400nn.graph kNN graph to glove-100.ssg"
    if [ -f "glove-100_400nn.graph" ]; then
      ./test_ssg_index glove-100/glove-100_base.fvecs glove-100_400nn.graph 500 50 60 glove-100.ssg > glove-100_index_${TIME}.log
    else
      echo "ERROR: glove-100_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (glove-100_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS glove-100/glove-100_base.fvecs glove-100/glove-100_query.fvecs glove-100.ssg ${1} ${2} glove-100_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    glove-100/glove-100_groundtruth.ivecs 0.3 512 ${4} 2> glove-100_search_L${1}K${2}_${3}_T${4}.log
}

ssg_uqv() {
  # Build a proximity graph
  if [ ! -f "uqv.ssg" ]; then
    echo "Converting uqv_400nn.graph kNN graph to uqv.ssg"
    if [ -f "uqv_400nn.graph" ]; then
      ./test_ssg_index uqv/uqv_base.fvecs uqv_400nn.graph 500 40 60 uqv.ssg > uqv_index_${TIME}.log
    else
      echo "ERROR: uqv_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (uqv_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS uqv/uqv_base.fvecs uqv/uqv_query.fvecs uqv.ssg ${1} ${2} uqv_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    uqv/uqv_groundtruth.ivecs 0.3 512 ${4} 2> uqv_search_L${1}K${2}_${3}_T${4}.log
}

ssg_msong() {
  # Build a proximity graph
  if [ ! -f "msong.ssg" ]; then
    echo "Converting msong_400nn.graph kNN graph to msong.ssg"
    if [ -f "msong_400nn.graph" ]; then
      ./test_ssg_index msong/deep1m_base.fvecs msong_400nn.graph 500 40 60 msong.ssg > msong_index_${TIME}.log
    else
      echo "ERROR: msong_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (msong_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS msong/msong_base.fvecs msong/msong_query.fvecs msong.ssg ${1} ${2} msong_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    msong/msong_groundtruth.ivecs 0.3 1024 ${4} 2> msong_search_L${1}K${2}_${3}_T${4}.log
}

ssg_deep1M() {
  # Build a proximity graph
  if [ ! -f "deep1M.ssg" ]; then
    echo "Converting deep1M_400nn.graph kNN graph to deep1M.ssg"
    if [ -f "deep1M_400nn.graph" ]; then
      ./test_ssg_index deep1M/deep1m_base.fvecs deep1M_400nn.graph 500 40 60 deep1M.ssg > deep1M_index_${TIME}.log
    else
      echo "ERROR: deep1M_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (deep1M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.ssg ${1} ${2} deep1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep1M/deep1m_groundtruth.ivecs 0.3 512 ${4} 2> deep1M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_deep100M_1T() {
  # Build a proximity graph
  if [ ! -f "deep100M.ssg" ]; then
    echo "Converting deep100M_400nn.graph kNN graph to deep100M.ssg"
    if [ -f "deep100M_400nn.graph" ]; then
      ./test_ssg_index deep100M/deep100M_base.fvecs deep100M_400nn.graph 500 40 60 deep100M.ssg > deep100M_index_${TIME}.log
    else
      echo "ERROR: deep100M_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (deep100M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS deep100M/deep100M_base.fvecs deep100M/deep100M_query.fvecs deep100M.ssg ${1} ${2} deep100M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep100M/deep100M_groundtruth.ivecs 0.25 512 ${4} 2> deep100M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_deep100M_16T() {
  # Build a proximity graph
  export sub_num=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
  for id in ${sub_num[@]}; do
    if [ ! -f "deep100M_${id}.ssg" ]; then
      echo "Converting deep100M_400nn_${id}.graph kNN graph to deep100M_${id}.ssg"
      if [ -f "deep100M_400nn_${id}.graph" ]; then
        ./test_ssg_index deep100M/deep100M_base_${id}.fvecs deep100M_400nn_${id}.graph 500 40 60 deep100M_${id}.ssg
      else
        echo "ERROR: deep100M_400nn_${id}.graph does not exist"
        exit 1
      fi
    fi
  done

  # Perform search
  echo "Perform kNN searching using SSG index (deep100M_L${1}K${2}T16)"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  for id in ${sub_num[@]}; do
    ./test_ssg_optimized_search_ADA_NNS deep100M/deep100M_base_${id}.fvecs deep100M/deep100M_query.fvecs deep100M_${id}.ssg ${1} ${2} deep100M_ssg_result_L${1}K${2}_${3}_T16_${id}.ivecs \
      deep100M/deep100M_groundtruth.ivecs 0.25 512 ${4} ${id} 2> deep100M_search_L${1}K${2}_${3}_T16_${id}.log &
  done
  wait
  rm -rf deep100M_search_L${1}K${2}_${3}_T16.log
  for id in ${sub_num[@]}; do
    awk 'NR==15{ printf "%s ", $2; exit }' deep100M_search_L${1}K${2}_${3}_T16_${id}.log >> deep100M_search_L${1}K${2}_${3}_T16.log
    awk 'NR==16{ print substr($0, 1, length($0) - 1); exit }' deep100M_search_L${1}K${2}_${3}_T16_${id}.log >> deep100M_search_L${1}K${2}_${3}_T16.log
  done
  cat deep100M_search_L${1}K${2}_${3}_T16.log | awk '{sum += $2;} {if(NR==1) min = $1} {if($1 < min) min = $1} END { print "min_qps: " min; print "recall: " sum; }' >> deep100M_search_L${1}K${2}_${3}_T16.log 
}

if [[ ${#} -eq 1 ]]; then
  if [ "${1}" == "sift1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_sift1M ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "sift10M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_sift10M ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "gist1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_gist1M ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "crawl" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_crawl ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "glove-100" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_glove-100 ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "deep1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_deep1M ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "uqv" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_uqv ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "msong" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_msong ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "deep100M_1T" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_deep100M_1T ${l} ${k} ADA-NNS ${t}
        done
      done
    done
  elif [ "${1}" == "deep100M_16T" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        ssg_deep100M_16T ${l} ${k} ADA-NNS 1
      done
    done
  elif [ "${1}" == "all" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
#          ssg_sift1M ${l} ${k} ADA-NNS ${t}
#          ssg_gist1M ${l} ${k} ADA-NNS ${t}
#          ssg_crawl ${l} ${k} ADA-NNS ${t}
          ssg_uqv ${l} ${k} ADA-NNS ${t}
          ssg_msong ${l} ${k} ADA-NNS ${t}
#          ssg_deep1M ${l} ${k} ADA-NNS ${t}
#          ssg_deep100M_1T ${l} ${k} ADA-NNS ${t}
        done
#        ssg_deep100M_16T ${l} ${k} ADA-NNS 1
      done
    done
  else
    echo "Usage: ./evalulate_ADA-NNS.sh [dataset]"
  fi
else
  echo "Usage: ./evalulate_ADA-NNS.sh [dataset]"
fi
