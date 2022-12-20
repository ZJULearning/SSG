//
// Created by 付聪 on 2017/6/21.
//

#include <chrono>

#include "index_random.h"
#include "index_ssg.h"
#include "util.h"
#include <omp.h>

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  int sub_id = -1;
#ifdef ADA_NNS
  if (argc < 11) {
    std::cout << " data_file query_file ssg_path L K result_path ground_truth_path tau hash_bitwidth num_threads [sub_id] [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc < 14) {
    sub_id = (int)atoi(argv[11]);
    if (argc == 13) {
      unsigned seed = (unsigned)atoi(argv[12]);
      srand(seed);
      std::cerr << "Using Seed " << seed << std::endl;
    }
  }
#else
  if (argc < 9) {
    std::cout << " data_file query_file ssg_path L K result_path ground_truth_path num_threads [sub_id] [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc < 12) {
    sub_id = (int)atoi(argv[9]);
    if (argc == 11) {
      unsigned seed = (unsigned)atoi(argv[10]);
      srand(seed);
      std::cerr << "Using Seed " << seed << std::endl;
    }
  }
#endif

  std::cerr << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim;
  float* data_load = nullptr;
  data_load = efanna2e::load_data(argv[1], points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);

  std::cerr << "Query Path: " << argv[2] << std::endl;

  unsigned query_num, query_dim;
  float* query_load = nullptr;
  query_load = efanna2e::load_data(argv[2], query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);

  assert(dim == query_dim);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexSSG index(dim, points_num, efanna2e::FAST_L2,
                           (efanna2e::Index*)(&init_index));

  std::cerr << "SSG Path: " << argv[3] << std::endl;
  std::cerr << "Result Path: " << argv[6] << std::endl;

#ifdef ADA_NNS
  float tau = (float)atof(argv[8]);
  uint64_t hash_bitwidth = (uint64_t)atoi(argv[9]);
  uint32_t num_threads = atoi(argv[10]);
  index.SetHashBitwidth(hash_bitwidth);
  index.SetTau(tau);
#else
  uint32_t num_threads = atoi(argv[8]);
#endif
  omp_set_num_threads(num_threads);

  index.Load(argv[3]);
  index.OptimizeGraph(data_load);

#ifdef ADA_NNS
  char* hash_function_name = new char[strlen(argv[3]) + strlen(".hash_function_") + strlen(argv[9]) + 1];
  char* hashed_set_name = new char[strlen(argv[3]) + strlen(".hashed_set") + strlen(argv[9]) + 1];
  strcpy(hash_function_name, argv[3]);
  strcat(hash_function_name, ".hash_function_");
  strcat(hash_function_name, argv[9]);
  strcat(hash_function_name, "b");
  strcpy(hashed_set_name, argv[3]);
  strcat(hashed_set_name, ".hashed_set_");
  strcat(hashed_set_name, argv[9]);
  strcat(hashed_set_name, "b");
  std::cerr << "hash_function_name: " << hash_function_name << std::endl;
  std::cerr << "hashed_set_name: " << hashed_set_name  << std::endl;

  if (index.ReadHashFunction(hash_function_name)) {
    if (!index.ReadHashedSet(hashed_set_name))
      index.GenerateHashedSet(hashed_set_name);
  }
  else {
    index.GenerateHashFunction(hash_function_name);
    index.GenerateHashedSet(hashed_set_name);
  }
  delete[] hash_function_name;
  delete[] hashed_set_name;
#endif

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

#ifdef THREAD_LATENCY
  std::vector<double> latency_stats(query_num, 0);
#endif
#ifdef PROFILE
  index.SetTimer(num_threads);
#endif
  // [ARC-SJ]: Minor optimization of greedy search 
  //           Allocate visited list once
  //           For large-scale dataset (e.g., DEEP100M),
  //           repeated allocation is a huge overhead
  boost::dynamic_bitset<> flags{index.Get_nd(), 0};
  // Warm up
  for (int loop = 0; loop < 3; ++loop) {
    for (unsigned i = 0; i < 10; ++i) {
      index.SearchWithOptGraph(query_load + i * dim, flags, K, paras, res[i].data());
    }
  }

  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned i = 0; i < query_num; i++) {
#ifdef THREAD_LATENCY
    auto query_start = std::chrono::high_resolution_clock::now();
#endif
    index.SearchWithOptGraph(query_load + i * dim, flags, K, paras, res[i].data());
#ifdef THREAD_LATENCY
   auto query_end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> query_diff = query_end - query_start;
   latency_stats[i] = query_diff.count() * 1000000;
#endif
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

// Print result
  std::cerr << "Search Time: " << diff.count() << std::endl;
  std::cerr << "QPS: " << query_num / diff.count() << std::endl;

  save_result(argv[6], res);

#ifdef EVAL_RECALL
  unsigned int* ground_truth_load = NULL;
  unsigned ground_truth_num, ground_truth_dim;
  ground_truth_load = efanna2e::load_data_ivecs(argv[7], ground_truth_num, ground_truth_dim);
  
  unsigned int topk_hit = 0;
  for (unsigned int i = 0; i < query_num; i++) {
    for (unsigned int j = 0; j < K; j++) {
      for (unsigned int k = 0; k < K; k++) {
        if (sub_id == -1) {
          if (res[i][j] == *(ground_truth_load + i * ground_truth_dim + k)) {
            topk_hit++;
            break;
          }
        }
        else { // [ARC-SJ] Recall for deep100M_16T
          if (res[i][j] + (6250000 * sub_id) == *(ground_truth_load + i * ground_truth_dim + k)) {
            topk_hit++;
            break;
          }
        }
      }
    }
  }
  std::cerr << (float)topk_hit / (query_num * K) * 100 << "%" << std::endl;
#endif
#ifdef THREAD_LATENCY
  std::sort(latency_stats.begin(), latency_stats.end());
  double mean_latency = 0;
  for (uint64_t q = 0; q < query_num; q++) {
    mean_latency += latency_stats[q];
  }
  mean_latency /= query_num;
  std::cerr << "mean_latency: " << mean_latency << "ms" << std::endl;
  std::cerr << "99% latency: " << latency_stats[(unsigned long long)(0.999 * query_num)] << "ms"<< std::endl;
#endif
#ifdef GET_DIST_COMP
  std::cerr << "========Distance Compute Report========" << std::endl;
  std::cerr << "# of distance compute: " << index.GetTotalDistComp() << std::endl;
  std::cerr << "# of missed distance compute: " << index.GetTotalDistCompMiss() << std::endl;
  std::cerr << "Ratio: " << (float)index.GetTotalDistCompMiss() / index.GetTotalDistComp()  * 100 << " %" << std::endl;
  std::cerr << "Speedup: " << (float)(index.Get_nd()) * query_num / index.GetTotalDistComp() << std::endl;
  std::cerr << "=====================================" << std::endl;
#endif
#ifdef PROFILE
  std::cerr << "=======Profile Report========" << std::endl;
  double* timer = (double*)calloc(4, sizeof(double));
  for (unsigned int tid = 0; tid < num_threads; tid++) {
    timer[0] += index.GetTimer(tid * 4); // visited list init time
    timer[1] += index.GetTimer(tid * 4 + 1); // query hash stage time
    timer[2] += index.GetTimer(tid * 4 + 2); // candidate selection stage time
    timer[3] += index.GetTimer(tid * 4 + 3); // fast L2 distance compute time
  }
#ifdef ADA_NNS
  std::cerr << "visited_init time: " << timer[0] / query_num << "ms" << std::endl;
  std::cerr << "query_hash time: " << timer[1] / query_num << "ms" << std::endl;
  std::cerr << "cand_select time: " << timer[2] / query_num << "ms" << std::endl;
  std::cerr << "dist time: " << timer[3] / query_num << "ms" << std::endl;
#else
  std::cerr << "visited_init time: " << timer[0] / query_num << "ms" << std::endl;
  std::cerr << "dist time: " << timer[3] / query_num << "ms" << std::endl;
#endif
  std::cerr << "=====================================" << std::endl;
#endif

  return 0;
}
