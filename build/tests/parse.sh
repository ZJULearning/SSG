#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
export all_data=(sift1M gist1M crawl deep1M)

parse_log() {
  export dataset=${1} # SIFT1M, GIST1M, CRAWL, DEEP1M
  export tag=${2}
  export log_list=`find . -name "${dataset}_search_L*${tag}.log" | sort -V`
  # Basic parsing
  export QPS=`cat ${log_list} | grep "QPS" | awk '{printf "%s\n", $2}'`
  echo ${QPS} > "${dataset}_${tag}_QPS.summary"
  export precision=`cat ${log_list} | grep "\%" | awk '{printf "%s\n", $1}'`
  echo ${precision} > "${dataset}_${tag}_precision.summary"
  # Latency report parsing
  export query_hash_time=`cat ${log_list} | grep "query_hash" | awk '{printf "%s\n", $3}'`
  export hash_approx_time=`cat ${log_list} | grep "hash_approx" | awk '{printf "%s\n", $3}'`
  export dist_time=`cat ${log_list} | grep "dist" | awk '{printf "%s\n", $3}'`
  echo ${query_hash_time} > "${dataset}_${tag}_query_hash_time.summary"
  echo ${hash_approx_time} > "${dataset}_${tag}_hash_approx_time.summary"
  echo ${dist_time} > "${dataset}_${tag}_dist_time.summary"
}

if [ $# == 2 ]; then
  if [ "${1}" == "all" ]; then
    for data in ${all_data[@]}; do
      echo "Parsing ${data}_${2} summary..."
      parse_log ${data} ${2}
    done
  else
    parse_log ${1} ${2}
  fi
else
  echo "Usage: ./parse.sh [dataset] [K(top-k)_baseline_T(thread) or K(top-k)_aid_by_approx_theta_T(thread)"
fi
