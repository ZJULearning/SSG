#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
MAX_THREADS=`nproc --all`
#THREAD=(16)
THREAD=(${MAX_THREADS})
#THREAD=(1 2 4 8 10 12 14 16 18 20 22 ${MAX_THREADS})
K=(1)
L_SIZE=(50)
#L_SIZE=(20 22 24 30 35 38 40 50 55 58 60 63 70 80 84 90 91 98 100 110 117 120 123 130 135 140 150 160 170 180 190 200 250 300 350 400 450 500 550 600)
#L_SIZE=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000) # 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 400)

ssg_sift1M() {
  if [ ! -f "sift1M.ssg" ]; then
    echo "Converting sift1M_200nn.graph kNN graph to sift1M.ssg"
    if [ -f "sift1M_200nn.graph" ]; then
      ./test_ssg_index sift1M/sift_base.fvecs sift1M_200nn.graph 100 50 60 sift1M.ssg > sift1M_index_${TIME}.log
    else
      echo "ERROR: sift1M_200nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using SSG index (sift1M_L${l}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.ssg ${1} ${2} sift1M_ssg_result.ivecs \
    sift1M/sift_groundtruth.ivecs 512 0.25 ${4} 2> sift1M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_sift10M() {
  if [ ! -f "sift10M.ssg" ]; then
    echo "Converting sift10M_200nn.graph kNN graph to sift10M.ssg"
    if [ -f "sift10M_200nn.graph" ]; then
      ./test_ssg_index sift10M/sift10m_base.fvecs sift10M_200nn.graph 100 50 60 sift10M.ssg > sift10M_index_${TIME}.log
    else
      echo "ERROR: sift10M_200nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using SSG index (sift10M_L${l}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search sift10M/sift10m_base.fvecs sift10M/sift10m_query.fvecs sift10M.ssg ${1} ${2} sift10M_ssg_result.ivecs \
    sift10M/sift10m_groundtruth.ivecs 512 0.25 ${4} 2> sift10M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_gist1M() {
  if [ ! -f "gist1M.ssg" ]; then
    echo "Converting gist1M_400nn.graph kNN graph to gist1M.ssg"
    if [ -f "gist1M_400nn.graph" ]; then
      ./test_ssg_index gist1M/gist_base.fvecs gist1M_400nn.graph 500 70 60 gist1M.ssg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist1M_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using SSG index (gist1M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.ssg ${1} ${2} gist1M_ssg_result.ivecs \
    gist1M/gist_groundtruth.ivecs 1024 0.3 ${4} 2> gist1M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_deep1M() {
  if [ ! -f "deep1M.ssg" ]; then
    echo "Converting deep1M_400nn.graph kNN graph to deep1M.ssg"
    if [ -f "deep1M_400nn.graph" ]; then
      ./test_ssg_index deep1M/deep1m_base.fvecs deep1M_400nn.graph 500 40 60 deep1M.ssg > deep1M_index_${TIME}.log
    else
      echo "ERROR: deep1M_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using SSG index (deep1M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.ssg ${1} ${2} deep1M_ssg_result.ivecs \
    deep1M/deep1m_groundtruth.ivecs 512 0.35 ${4} 2> deep1M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_deep100M() {
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
  echo "Perform kNN searching using SSG index (deep100M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_deep100M deep100M/deep100M_base.fvecs deep100M/deep100M_query.fvecs deep100M.ssg ${1} ${2} deep100M_ssg_result.ivecs \
    deep100M/deep100M_groundtruth.ivecs 512 0.3 ${4} 2> deep100M_search_L${1}K${2}_${3}_T${4}.log
}

ssg_glove-100() {
  if [ ! -f "glove-100.ssg" ]; then
    echo "Converting glove-100_400nn.graph kNN graph to glove-100.ssg"
    if [ -f "glove-100_400nn.graph" ]; then
      ./test_ssg_index glove-100/glove-100_base.fvecs glove-100_400nn.graph 500 50 60 glove-100.ssg > glove-100_index_${TIME}.log
    else
      echo "ERROR: glove-100_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using SSG index (glove-100_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search glove-100/glove-100_base.fvecs glove-100/glove-100_query.fvecs glove-100.ssg ${1} ${2} glove-100_ssg_result.ivecs \
    glove-100/glove-100_groundtruth.ivecs 512 0.3 ${4} 2> glove-100_search_L${1}K${2}_${3}_T${4}.log
}

ssg_crawl() {
  if [ ! -f "crawl.ssg" ]; then
    echo "Converting crawl_400nn.graph kNN graph to crawl.ssg"
    if [ -f "crawl_400nn.graph" ]; then
      ./test_ssg_index crawl/crawl_base.fvecs crawl_400nn.graph 500 40 60 crawl.ssg > crawl_index_${TIME}.log
    else
      echo "ERROR: crawl_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using SSG index (crawl_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.ssg ${1} ${2} crawl_ssg_result.ivecs \
    crawl/crawl_groundtruth.ivecs 512 0.3 ${4} 2> crawl_search_L${1}K${2}_${3}_T${4}.log
}

if [ "${1}" == "sift1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_sift1M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "sift10M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_sift10M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "gist1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_gist1M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "deep1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_deep1M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "deep100M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_deep100M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "glove-100" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_glove-100 ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "crawl" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_crawl ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "all" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${THREAD[@]}; do
        ssg_sift1M ${l} ${k} ${2} ${t}
        ssg_gist1M ${l} ${k} ${2} ${t}
        ssg_deep1M ${l} ${k} ${2} ${t}
#        ssg_glove-100 ${l} ${k} ${2} ${t}
        ssg_crawl ${l} ${k} ${2} ${t}
      done
    done
  done
else
  echo "Please use either 'sift1M' or 'gist1M' or 'deep1M' or 'glove-100' or 'crawl' or 'all' as an argument"
fi
