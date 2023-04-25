#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
MAX_THREADS=`nproc --all`
TARGET_RECALL=95

l_start=10
l_end=200
l_step=10

for (( l=l_start; l<=l_end; l=l+l_step )); do
	L_SIZE+=($l)
done

l_start=250
l_end=500
l_step=50

for (( l=l_start; l<=l_end; l=l+l_step )); do
	L_SIZE+=($l)
done

l_start=1000
l_end=2500
l_step=500

for (( l=l_start; l<=l_end; l=l+l_step )); do
	L_SIZE+=($l)
done


THREAD=(1)
K=(1 10)
TAU=(0.3)
HASH=(512)

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
  echo "Perform kNN searching using SSG index (sift1M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.ssg ${1} ${2} sift1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift1M/sift_groundtruth.ivecs ${5} ${6} ${4} 2> sift1M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
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
  echo "Perform kNN searching using SSG index (gist1M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.ssg ${1} ${2} gist1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    gist1M/gist_groundtruth.ivecs ${5} ${6} ${4} 2> gist1M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
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
  echo "Perform kNN searching using SSG index (crawl_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.ssg ${1} ${2} crawl_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    crawl/crawl_groundtruth.ivecs ${5} ${6} ${4} 2> crawl_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
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
  echo "Perform kNN searching using SSG index (deep1M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.ssg ${1} ${2} deep1M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep1M/deep1m_groundtruth.ivecs ${5} ${6} ${4} 2> deep1M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}


ssg_msong() {
  # Build a proximity graph
  if [ ! -f "msong.ssg" ]; then
    echo "Converting msong_400nn.graph kNN graph to msong.ssg"
    if [ -f "msong_400nn.graph" ]; then
      ./test_ssg_index msong/msong_base.fvecs msong_400nn.graph 500 40 60 msong.ssg > msong_index_${TIME}.log
    else
      echo "ERROR: msong_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (msong_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS msong/msong_base.fvecs msong/msong_query.fvecs msong.ssg ${1} ${2} msong_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    msong/msong_groundtruth.ivecs ${5} ${6} ${4} 2> msong_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}

ssg_glove-100() {
  # Build a proximity graph
  if [ ! -f "glove-100.ssg" ]; then
    echo "Converting glove-100_400nn.graph kNN graph to glove-100.ssg"
    if [ -f "glove-100_400nn.graph" ]; then
      ./test_ssg_index glove-100/glove-100_base.fvecs glove-100_400nn.graph 400 420 12 glove-100.ssg > glove-100_index_${TIME}.log
    else
      echo "ERROR: glove-100_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform kNN searching using SSG index (glove-100_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS glove-100/glove-100_base.fvecs glove-100/glove-100_query.fvecs glove-100.ssg ${1} ${2} glove-100_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    glove-100/glove-100_groundtruth.ivecs ${5} ${6} ${4} 2> glove-100_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}

ssg_deep100M() {
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
  echo "Perform kNN searching using SSG index (deep100M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_ssg_optimized_search_ADA_NNS deep100M/deep100M_base.fvecs deep100M/deep100M_query.fvecs deep100M.ssg ${1} ${2} deep100M_ssg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep100M/deep100M_groundtruth.ivecs ${5} ${6} ${4} 2> deep100M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}

if [[ ${#} -eq 1 ]]; then
  if [ "${1}" == "sift1M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
          ssg_sift1M ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "gist1M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
          ssg_gist1M ${l} ${k} ADA-NNS ${t} ${tt} 1024
          done
        done
      done
    done
  elif [ "${1}" == "crawl" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
          ssg_crawl ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "deep1M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
          ssg_deep1M ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "msong" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            ssg_msong ${l} ${k} ADA-NNS ${t} ${tt} 768
          done
        done
      done
    done
  elif [ "${1}" == "glove-100" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            ssg_glove-100 ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "deep100M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
          ssg_deep100M ${l} ${k} ADA-NNS ${t} ${tt}
          done
        done
      done
    done
  elif [ "${1}" == "all" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for h in ${HASH[@]}; do
            for l_size in ${L_SIZE[@]}; do
              declare -i l=l_size
              ssg_sift1M ${l} ${k} ADA-NNS ${t} ${tt} 512
              ssg_gist1M ${l} ${k} ADA-NNS ${t} ${tt} 1024
              ssg_crawl ${l} ${k} ADA-NNS ${t} ${tt} 512
              ssg_deep1M ${l} ${k} ADA-NNS ${t} ${tt} 512
              ssg_msong ${l} ${k} ADA-NNS ${t} ${tt} 768
              ssg_glove-100 ${l} ${k} ADA-NNS ${t} ${tt} 512
              ssg_deep100M ${l} ${k} ADA-NNS ${t} ${tt} 512
            done
          done
        done
      done
    done
  else
    echo "Usage: ./evalulate_ADA-NNS.sh [dataset]"
  fi
elif [[ ${#} -eq 6 ]]; then
  ssg_$1 $2 $3 ADA-NNS $4 $5 $6
  
else
  echo "Usage: ./evalulate_ADA-NNS.sh [dataset]"
fi
