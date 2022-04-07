#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
#K=(1)
#L_SIZE=(10 20 30 40 50 60 70 80 90 100)

K=(10)
#L_SIZE=(20)
L_SIZE=(20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
#L_SIZE=(250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)

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
  echo "Perform kNN searching using NSG index (sift1M_L${l}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.ssg ${1} ${2} sift1M_ssg_result.ivecs \
  sift1M/sift_groundtruth.ivecs 2> sift1M_search_L${1}K${2}_${3}.log
#  ./test_ssg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.ssg ${1} ${2} sift1M_ssg_result.ivecs \
#    sift1M/sift_groundtruth.ivecs 512 0.25 > sift1M_search_L${1}K${2}_${3}.log
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
  echo "Perform kNN searching using NSG index (gist1M_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.ssg ${1} ${2} gist1M_ssg_result.ivecs \
    gist1M/gist_groundtruth.ivecs 1024 0.3 2> gist1M_search_L${1}K${2}_${3}.log
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
  echo "Perform kNN searching using NSG index (deep1M_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.ssg ${1} ${2} deep1M_ssg_result.ivecs \
    deep1M/deep1m_groundtruth.ivecs 512 0.3 2> deep1M_search_L${1}K${2}_${3}.log
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
  echo "Perform kNN searching using NSG index (glove-100_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search glove-100/glove-100_base.fvecs glove-100/glove-100_query.fvecs glove-100.ssg ${1} ${2} glove-100_ssg_result.ivecs \
    glove-100/glove-100_groundtruth.ivecs 512 0.3 2> glove-100_search_L${1}K${2}_${3}.log
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
  echo "Perform kNN searching using NSG index (crawl_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.ssg ${1} ${2} crawl_ssg_result.ivecs \
    crawl/crawl_groundtruth.ivecs 512 0.3 2> crawl_search_L${1}K${2}_${3}.log
}

if [ "${1}" == "sift1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      ssg_sift1M ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "gist1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      ssg_gist1M ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "deep1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      ssg_deep1M ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "glove-100" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      ssg_glove-100 ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "crawl" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      ssg_crawl ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "all" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      ssg_sift1M ${l} ${k} ${2}
      ssg_gist1M ${l} ${k} ${2}
      ssg_deep1M ${l} ${k} ${2}
#      ssg_glove-100 ${l} ${k} ${2}
      ssg_crawl ${l} ${k} ${2}
    done
  done
else
  echo "Please use either 'sift1M' or 'gist1M' or 'deep1M' or 'glove-100' or 'crawl' or 'all' as an argument"
fi
