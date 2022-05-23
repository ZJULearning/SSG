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
  if (argc < 11) {
    std::cout << "./run data_file query_file ssg_path L K result_path ground_truth_path hash_bitwidth threshold_percent num_threads [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc == 12) {
    unsigned seed = (unsigned)atoi(argv[11]);
    srand(seed);
    std::cerr << "Using Seed " << seed << std::endl;
  }

  std::cerr << "Query Path: " << argv[2] << std::endl;

  unsigned query_num, query_dim;
  float* query_load = nullptr;
  query_load = efanna2e::load_data(argv[2], query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K * 16);

  unsigned int num_threads = atoi(argv[10]);

  double global_search_time = 0.0;
#ifdef THREAD_LATENCY
  std::vector<double> latency_stats(query_num, 0);
#endif
#ifdef PROFILE
  unsigned num_timer = 3;
  std::vector<double> global_timer;
  global_timer.resize(num_timer, 0.0);
#endif

#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned int iter = 0; iter < 16; iter++) {
    unsigned points_num, dim;
    float* data_load = nullptr;
    char iter_char[3];
    std::sprintf(iter_char, "%d", iter);
    unsigned int dataname_len = strlen(argv[1]);
    char* sub_dataname = new char[dataname_len + 5];
    strncpy(sub_dataname, argv[1], dataname_len - 6);
    sub_dataname[dataname_len - 6] = '\0';
    strcat(sub_dataname, "_");
    strcat(sub_dataname, iter_char);
    strcat(sub_dataname, &argv[1][dataname_len - 6]);
    std::cerr << "Data Path: " << sub_dataname << std::endl;

    data_load = efanna2e::load_data(sub_dataname, points_num, dim);
    data_load = efanna2e::data_align(data_load, points_num, dim);

    assert(dim == query_dim);

    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexSSG index(dim, points_num, efanna2e::FAST_L2,
        (efanna2e::Index*)(&init_index));

    unsigned int indexname_len = strlen(argv[3]);
    char* sub_indexname = new char[indexname_len + 5];
    strncpy(sub_indexname, argv[3], indexname_len - 4);
    sub_indexname[indexname_len - 4] = '\0';
    strcat(sub_indexname, "_");
    strcat(sub_indexname, iter_char);
    strcat(sub_indexname, &argv[3][indexname_len - 4]);
    std::cerr << "SSG Path: " << sub_indexname << std::endl;
    std::cerr << "Result Path: " << argv[6] << std::endl;

#ifdef THETA_GUIDED_SEARCH
    index.hash_bitwidth = (unsigned)atoi(argv[8]);
    index.threshold_percent = (float)atof(argv[9]);
#endif

    index.Load(sub_indexname);
    index.OptimizeGraph(data_load);

#ifdef THETA_GUIDED_SEARCH
    // SJ: For profile, related with #THETA_GUIDED_SEARCH flag
    char* hash_function_name = new char[strlen(sub_indexname) + strlen(".hash_function_") + strlen(argv[9]) + 1];
    char* hash_vector_name = new char[strlen(sub_indexname) + strlen(".hash_vector") + strlen(argv[9]) + 1];
    strcpy(hash_function_name, sub_indexname);
    strcat(hash_function_name, ".hash_function_");
    strcat(hash_function_name, argv[8]);
    strcat(hash_function_name, "b");
    strcpy(hash_vector_name, sub_indexname);
    strcat(hash_vector_name, ".hash_vector_");
    strcat(hash_vector_name, argv[8]);
    strcat(hash_vector_name, "b");

    if (index.LoadHashFunction(hash_function_name)) {
      if (!index.LoadHashValue(hash_vector_name))
        index.GenerateHashValue(hash_vector_name);
    }
    else {
      index.GenerateHashFunction(hash_function_name);
      index.GenerateHashValue(hash_vector_name);
    }
#endif
#ifdef PROFILE
    index.num_timer = num_timer;
    index.profile_time.resize(num_threads * index.num_timer, 0.0);
#endif
    // Warm up
    for (int loop = 0; loop < 3; ++loop) {
      for (unsigned i = 0; i < 10; ++i) {
        index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data() + K * iter);
      }
    }

    omp_set_num_threads(num_threads);
    auto s = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned i = 0; i < query_num; i++) {
#ifdef THREAD_LATENCY
      auto query_start = std::chrono::high_resolution_clock::now();
#endif
      index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data() + K * iter);
#ifdef THREAD_LATENCY
      auto query_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> query_diff = query_end - query_start;
      latency_stats[i] += query_diff.count() * 1000000;
#endif
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    global_search_time += diff.count();

    // SJ: free dynamic alloc arrays
    delete[] data_load;
    delete[] sub_dataname;
    delete[] sub_indexname;
#ifdef THETA_GUIDED_SEARCH
    delete[] hash_function_name;
    delete[] hash_vector_name;
#endif
  }

  std::cerr << "Search Time: " << global_search_time << std::endl;
  std::cerr << "QPS: " << query_num / global_search_time << std::endl;

#ifdef EVAL_RECALL
  unsigned int* ground_truth_load = NULL;
  unsigned ground_truth_num, ground_truth_dim;
  ground_truth_load = efanna2e::load_data_ivecs(argv[7], ground_truth_num, ground_truth_dim);
  
  unsigned int topk_hit = 0;
  for (unsigned int i = 0; i < query_num; i++) {
    unsigned int topk_local_hit = 0;
    for (unsigned int j = 0; j < K * 16; j++) {
      for (unsigned int k = 0; k < K; k++) {
//        std::cerr << res[i][j] << ", " << *(ground_truth_load + i * ground_truth_dim + k) << std::endl;
        if (res[i][j] + (6250000 * (j / K)) == *(ground_truth_load + i * ground_truth_dim + k)) {
          topk_hit++;
          break;
        }
      }
    }
//    std::cerr << std::endl;
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
#ifdef PROFILE
  std::cerr << "========Thread Latency Report========" << std::endl;
  double* timer = (double*)calloc(index.num_timer, sizeof(double));
  for (unsigned int tid = 0; tid < num_threads; tid++) {
    timer[0] += index.profile_time[tid * index.num_timer];
    timer[1] += index.profile_time[tid * index.num_timer + 1];
    timer[2] += index.profile_time[tid * index.num_timer + 2];
  }
#ifdef THETA_GUIDED_SEARCH
  std::cerr << "query_hash time: " << timer[0] / query_num << "ms" << std::endl;
  std::cerr << "hash_approx time: " << timer[1] / query_num << "ms" << std::endl;
  std::cerr << "dist time: " << timer[2] / query_num << "ms" << std::endl;
#else
  std::cerr << "dist time: " << timer[2] / query_num << "ms" << std::endl;
#endif
  std::cerr << "=====================================" << std::endl;
#endif

  save_result(argv[6], res);

  return 0;
}
