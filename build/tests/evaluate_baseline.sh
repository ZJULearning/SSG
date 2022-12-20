#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
MAX_THREADS=`nproc --all`
THREAD=(1)
K=(1)
L_SIZE=(35)

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
  ./test_ssg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.ssg ${1} ${2} sift1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift1M/sift_groundtruth.ivecs ${4} 2> sift1M_search_L${1}K${2}_${3}_T${4}.log
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
  ./test_ssg_optimized_search sift10M/sift10m_base.fvecs sift10M/sift10m_query.fvecs sift10M.ssg ${1} ${2} sift10M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift10M/sift10m_groundtruth.ivecs ${4} 2> sift10M_search_L${1}K${2}_${3}_T${4}.log
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
  ./test_ssg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.ssg ${1} ${2} gist1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    gist1M/gist_groundtruth.ivecs ${4} 2> gist1M_search_L${1}K${2}_${3}_T${4}.log
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
  ./test_ssg_optimized_search crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.ssg ${1} ${2} crawl_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    crawl/crawl_groundtruth.ivecs ${4} 2> crawl_search_L${1}K${2}_${3}_T${4}.log
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
  ./test_ssg_optimized_search deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.ssg ${1} ${2} deep1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep1M/deep1m_groundtruth.ivecs ${4} 2> deep1M_search_L${1}K${2}_${3}_T${4}.log
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
  ./test_ssg_optimized_search deep100M/deep100M_base.fvecs deep100M/deep100M_query.fvecs deep100M.ssg ${1} ${2} deep100M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep100M/deep100M_groundtruth.ivecs ${4} 2> deep100M_search_L${1}K${2}_${3}_T${4}.log
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
    ./test_ssg_optimized_search deep100M/deep100M_base_${id}.fvecs deep100M/deep100M_query.fvecs deep100M_${id}.ssg ${1} ${2} deep100M_ssg_result_L${1}K${2}_${3}_T16_${id}.ivecs \
      deep100M/deep100M_groundtruth.ivecs ${id} 2> deep100M_search_L${1}K${2}_${3}_T16_${id}.log &
  done
  wait
  rm -rf deep100M_search_L${1}K${2}_${3}_T16.log
  for id in ${sub_num[@]}; do
    awk 'NR==8{ printf "%s ", $2; exit }' deep100M_search_L${1}K${2}_${3}_T16_${id}.log >> deep100M_search_L${1}K${2}_${3}_T16.log
    awk 'NR==9{ print substr($0, 1, length($0) - 1); exit }' deep100M_search_L${1}K${2}_${3}_T16_${id}.log >> deep100M_search_L${1}K${2}_${3}_T16.log
  done
  cat deep100M_search_L${1}K${2}_${3}_T16.log | awk '{sum += $2;} {if(NR==1) min = $1} {if($1 < min) min = $1} END { print "min_qps: " min; print "recall: " sum; }' >> deep100M_search_L${1}K${2}_${3}_T16.log 
}

if [[ ${#} -eq 1 ]]; then
  if [ "${1}" == "sift1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_sift1M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "sift10M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_sift10M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "gist1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_gist1M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "crawl" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_crawl ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "deep1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_deep1M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "deep100M_1T" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_deep100M_1T ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "deep100M_16T" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        ssg_deep100M_16T ${l} ${k} baseline 1
      done
    done
  elif [ "${1}" == "all" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          ssg_sift1M ${l} ${k} baseline ${t}
          ssg_gist1M ${l} ${k} baseline ${t}
          ssg_crawl ${l} ${k} baseline ${t}
          ssg_deep1M ${l} ${k} baseline ${t}
          ssg_deep100M_1T ${l} ${k} baseline ${t}
        done
        ssg_deep100M_16T ${l} ${k} baseline 1
      done
    done
  else
    echo "Usage: ./evalulate_baseline.sh [dataset]"
  fi
else
  echo "Usage: ./evalulate_baseline.sh [dataset]"
fi
