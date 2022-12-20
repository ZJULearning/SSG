#include "index_ssg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <queue>
#include <boost/dynamic_bitset.hpp>

#include "exceptions.h"
#include "parameters.h"

#include <sys/mman.h>

constexpr double kPi = 3.14159265358979323846264;

namespace efanna2e {

#define _CONTROL_NUM 100

IndexSSG::IndexSSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexSSG::~IndexSSG() {}

void IndexSSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  unsigned n_ep=eps_.size();
  out.write((char *)&n_ep, sizeof(unsigned));
  out.write((char *)eps_.data(), n_ep*sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexSSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  unsigned n_ep=0;
  in.read((char *)&n_ep, sizeof(unsigned));
  eps_.resize(n_ep);
  in.read((char *)eps_.data(), n_ep*sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  std::cerr << "Average Degree = " << cc << std::endl;
}

void IndexSSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexSSG::get_neighbors(const unsigned q, const Parameters &parameter,
                             std::vector<Neighbor> &pool) {
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned L = parameter.Get<unsigned>("L");
  flags[q] = true;
  for (unsigned i = 0; i < final_graph_[q].size(); i++) {
    unsigned nid = final_graph_[q][i];
    for (unsigned nn = 0; nn < final_graph_[nid].size(); nn++) {
      unsigned nnid = final_graph_[nid][nn];
      if (flags[nnid]) continue;
      flags[nnid] = true;
      float dist = distance_->compare(data_ + dimension_ * q,
                                      data_ + dimension_ * nnid, dimension_);
      pool.push_back(Neighbor(nnid, dist, true));
      if (pool.size() >= L) break;
    }
    if (pool.size() >= L) break;
  }
}

void IndexSSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
  std::mt19937 rng(rand());
  GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexSSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  // ep_ = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;  // For Compatibility
}

void IndexSSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameters, float threshold,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameters.Get<unsigned>("R");
  width = range;
  unsigned start = 0;

  boost::dynamic_bitset<> flags{nd_, 0};
  for (unsigned i = 0; i < pool.size(); ++i) {
    flags[pool[i].id] = 1;
  }
  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)q,
                                    data_ + dimension_ * (size_t)id,
                                    (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size()) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                     sqrt(p.distance * result[t].distance);
      if (cos_ij > threshold) {
        occlude = true;
        break;
      }
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexSSG::InterInsert(unsigned n, unsigned range, float threshold,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(
              data_ + dimension_ * (size_t)result[t].id,
              data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);
          float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                         sqrt(p.distance * result[t].distance);
          if (cos_ij > threshold) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
        if (result.size() < range) {
          des_pool[result.size()].distance = -1;
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

void IndexSSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cerr << "Graph Link" << std::endl;
  unsigned progress = 0;
  unsigned percent = 100;
  unsigned step_size = nd_ / percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);

  float angle = parameters.Get<float>("A");
  float threshold = std::cos(angle / 180 * kPi);

#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      get_neighbors(n, parameters, pool);
      sync_prune(n, pool, parameters, threshold, cut_graph_);
      /*
      cnt++;
      if (cnt % step_size == 0) {
        LockGuard g(progress_lock);
        std::cout << progress++ << "/" << percent << " completed" << std::endl;
      }
      */
    }

#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      InterInsert(n, range, threshold, locks, cut_graph_);
    }
  }
}

void IndexSSG::Build(size_t n, const float *data,
                     const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) {
        break;
      }
      pool_size = j;
    }
    ++pool_size;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  DFS_expand(parameters);

  unsigned max, min, avg;
  max = 0;
  min = nd_;
  avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n",
         max, min, avg);

  /* Buggy!
  strong_connect(parameters);

  max = 0;
  min = nd_;
  avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics(After TreeGrow): Max = %d, Min = %d, Avg = %d\n",
         max, min, avg);
  */

  has_built = true;
}

void IndexSSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  std::mt19937 rng(rand());
  GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
  assert(eps_.size() < L);
  for(unsigned i=0; i<eps_.size(); i++){
    init_ids[i] = eps_[i];
  }

  for (unsigned i = 0; i < L; i++) {
    unsigned id = init_ids[i];
    float dist = distance_->compare(data_ + dimension_ * id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist = distance_->compare(query, data_ + dimension_ * id,
                                        (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexSSG::SearchWithOptGraph(const float *query, boost::dynamic_bitset<>& flags, size_t K,
                                  const Parameters &parameters,
                                  unsigned *indices) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  std::mt19937 rng(rand());
  GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
  assert(eps_.size() < L);
  for(unsigned i=0; i<eps_.size(); i++){
    init_ids[i] = eps_[i];
  }
#ifdef PROFILE
  unsigned int tid = omp_get_thread_num();
  auto visited_list_init_start = std::chrono::high_resolution_clock::now();
#endif
//  [ARC-SJ] Initialize visited list, allocation moved to main module
//  boost::dynamic_bitset<> flags{nd_, 0};
  flags.reset();
#ifdef PROFILE
  auto visited_list_init_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> visited_list_init_diff = visited_list_init_end - visited_list_init_start;
  profile_time[tid * 4] += visited_list_init_diff.count() * 1000000;
#endif
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  // std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + L);
#ifdef ADA_NNS
#ifdef PROFILE
  auto query_hash_start = std::chrono::high_resolution_clock::now();
#endif
  std::vector<HashNeighbor> selected_pool(100);
  unsigned int hash_size = hash_bitwidth_ >> 5;
  unsigned int* hashed_query = new unsigned int[hash_size];
  QueryHash(query, hashed_query, hash_size); 
#ifdef __AVX__
  unsigned int hash_avx_size = hash_size >> 3;
  __m256i hashed_query_avx[hash_avx_size];
  for (unsigned int m = 0; m < hash_avx_size; m++) {
    hashed_query_avx[m] = _mm256_loadu_si256((__m256i*)&hashed_query[m << 3]);
  }
#endif
#ifdef PROFILE
  auto query_hash_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> query_hash_diff = query_hash_end - query_hash_start;
  profile_time[tid * 4 + 1] += query_hash_diff.count() * 1000000;
#endif
#endif

  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
#ifdef ADA_NNS
#ifdef PROFILE
      auto cand_select_start = std::chrono::high_resolution_clock::now();
#endif
      unsigned int selected_pool_size = CandidateSelection(hashed_query_avx, selected_pool, neighbors, MaxM, hash_size);
#ifdef PROFILE
      auto cand_select_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> cand_select_diff = cand_select_end - cand_select_start;
      profile_time[tid * 4 + 2] += cand_select_diff.count() * 1000000;
#endif
#endif
#ifdef PROFILE
      auto dist_start = std::chrono::high_resolution_clock::now();
#endif
#ifdef ADA_NNS
      for (unsigned m = 0; m < selected_pool_size; ++m)
        _mm_prefetch(opt_graph_ + node_size * selected_pool[m].id, _MM_HINT_T0);
      for (unsigned int m = 0; m < selected_pool_size; m++) {
        unsigned int id = selected_pool[m].id;
#else
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
#endif
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist =
            dist_fast->compare(query, data, norm, (unsigned)dimension_);
#ifdef GET_DIST_COMP
        total_dist_comp_++;
        if (dist >= retset[L - 1].distance){
          total_dist_comp_miss_++;
          continue;
        }
#else
        if (dist >= retset[L - 1].distance){
          continue;
        }
#endif
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
#ifdef PROFILE
      auto dist_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> dist_diff = dist_end - dist_start;
      profile_time[tid * 4 + 3] += dist_diff.count() * 1000000;
#endif
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexSSG::OptimizeGraph(const float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;
#ifdef ADA_NNS
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  uint64_t hash_function_size = dimension_ * hash_bitwidth_ * sizeof(float);
#ifdef MMAP_HUGETLB
  opt_graph_ = (char *) mmap(NULL, node_size * nd_ + hash_len * nd_ + hash_function_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_POPULATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#else
  opt_graph_ = (char *)malloc(node_size * nd_ + hash_len * nd_ + hash_function_size);
#endif
#else
#ifdef MMAP_HUGETLB
  opt_graph_ = (char *) mmap(NULL, node_size * nd_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_POPULATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#else
  opt_graph_ = (char *)malloc(node_size * nd_);
#endif
#endif
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  CompactGraph().swap(final_graph_);
}

void IndexSSG::DFS(boost::dynamic_bitset<> &flag,
                   std::vector<std::pair<unsigned, unsigned>> &edges,
                   unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      unsigned head = s.top();
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      unsigned tail = tmp;
      if (check_edge(head, tail)) {
        edges.push_back(std::make_pair(head, tail));
      }
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexSSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  bool found = false;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = true;
      break;
    }
  }
  if (!found) {
    for (int retry = 0; retry < 1000; ++retry) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}

bool IndexSSG::check_edge(unsigned h, unsigned t) {
  bool flag = true;
  for (unsigned i = 0; i < final_graph_[h].size(); i++) {
    if (t == final_graph_[h][i]) flag = false;
  }
  return flag;
}

void IndexSSG::strong_connect(const Parameters &parameter) {
  unsigned n_try = parameter.Get<unsigned>("n_try");
  std::vector<std::pair<unsigned, unsigned>> edges_all;
  std::mutex edge_lock;

#pragma omp parallel for
  for (unsigned nt = 0; nt < n_try; nt++) {
    unsigned root = rand() % nd_;
    boost::dynamic_bitset<> flags{nd_, 0};
    unsigned unlinked_cnt = 0;
    std::vector<std::pair<unsigned, unsigned>> edges;

    while (unlinked_cnt < nd_) {
      DFS(flags, edges, root, unlinked_cnt);
      // std::cout << unlinked_cnt << '\n';
      if (unlinked_cnt >= nd_) break;
      findroot(flags, root, parameter);
      // std::cout << "new root"<<":"<<root << '\n';
    }

    LockGuard guard(edge_lock);

    for (unsigned i = 0; i < edges.size(); i++) {
      edges_all.push_back(edges[i]);
    }
  }
  unsigned ecnt = 0;
  for (unsigned e = 0; e < edges_all.size(); e++) {
    unsigned start = edges_all[e].first;
    unsigned end = edges_all[e].second;
    unsigned flag = 1;
    for (unsigned j = 0; j < final_graph_[start].size(); j++) {
      if (end == final_graph_[start][j]) {
        flag = 0;
      }
    }
    if (flag) {
      final_graph_[start].push_back(end);
      ecnt++;
    }
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}

void IndexSSG::DFS_expand(const Parameters &parameter) {
  unsigned n_try = parameter.Get<unsigned>("n_try");
  unsigned range = parameter.Get<unsigned>("R");

  std::vector<unsigned> ids(nd_);
  for(unsigned i=0; i<nd_; i++){
    ids[i]=i;
  }
  std::random_shuffle(ids.begin(), ids.end());
  for(unsigned i=0; i<n_try; i++){
    eps_.push_back(ids[i]);
    //std::cout << eps_[i] << '\n';
  }
#pragma omp parallel for
  for(unsigned i=0; i<n_try; i++){
    unsigned rootid = eps_[i];
    boost::dynamic_bitset<> flags{nd_, 0};
    std::queue<unsigned> myqueue;
    myqueue.push(rootid);
    flags[rootid]=true;

    std::vector<unsigned> uncheck_set(1);

    while(uncheck_set.size() >0){
      while(!myqueue.empty()){
        unsigned q_front=myqueue.front();
        myqueue.pop();

        for(unsigned j=0; j<final_graph_[q_front].size(); j++){
          unsigned child = final_graph_[q_front][j];
          if(flags[child])continue;
          flags[child] = true;
          myqueue.push(child);
        }
      }

      uncheck_set.clear();
      for(unsigned j=0; j<nd_; j++){
        if(flags[j])continue;
        uncheck_set.push_back(j);
      }
      //std::cout <<i<<":"<< uncheck_set.size() << '\n';
      if(uncheck_set.size()>0){
        for(unsigned j=0; j<nd_; j++){
          if(flags[j] && final_graph_[j].size()<range){
            final_graph_[j].push_back(uncheck_set[0]);
            break;
          }
        }
        myqueue.push(uncheck_set[0]);
        flags[uncheck_set[0]]=true;
      }
    }
  }
}

#ifdef ADA_NNS 
void IndexSSG::GenerateHashFunction (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;
  std::normal_distribution<float> norm_dist (0.0, 1.0);
  std::mt19937 gen(rand());
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  hash_function_ = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
  float hash_function_norm[hash_bitwidth_ - 1];

  std::cerr << "GenerateHashFunction" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned int dim = 0; dim < dimension_; dim++) { // Random generated vector
    hash_function_[dim] = norm_dist(gen);
  }
  hash_function_norm[0] = dist_fast->norm(hash_function_, dimension_);

  for (unsigned int hash_col = 1; hash_col < hash_bitwidth_; hash_col++) { // Iterate to generate vectors orthogonal to 0th column
    for (unsigned int dim = 0; dim < dimension_; dim++) { // Random generated vector
       hash_function_[hash_col * dimension_ + dim] = norm_dist(gen);
    }

    // Gram-schmidt process
    for (unsigned int compare_col = 0; compare_col < hash_col; compare_col++) {
      float inner_product_between_hash = dist_fast->DistanceInnerProduct::compare(&hash_function_[hash_col * dimension_], &hash_function_[compare_col * dimension_], (unsigned)dimension_);
      for (unsigned int dim = 0; dim < dimension_; dim++) {
        hash_function_[hash_col * dimension_ + dim] -= (inner_product_between_hash / hash_function_norm[compare_col] * hash_function_[compare_col * dimension_ + dim]);
      }
    }
    hash_function_norm[hash_col] = dist_fast->norm(&hash_function_[hash_col * dimension_], dimension_);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
//    std::cout << "HashFunction generation time: " << diff.count() * 1000 << std::endl;;

  std::ofstream file_hash_function(file_name, std::ios::binary | std::ios::out);
  file_hash_function.write((char*)&hash_bitwidth_, sizeof(unsigned int));
  file_hash_function.write((char*)hash_function_, dimension_ * hash_bitwidth_ * sizeof(float));
  file_hash_function.close();
}
void IndexSSG::GenerateHashedSet (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;
  uint64_t hash_len = (hash_bitwidth_ >> 3);

  std::cerr << "GenerateHashedSet" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned int i = 0; i < nd_; i++) {
    hashed_set_ = (unsigned int*)(opt_graph_ + node_size * nd_ + hash_len * i);
    float* vertex = (float *)(opt_graph_ + node_size * i + sizeof(float));
    for (unsigned int num_integer = 0; num_integer < (hash_bitwidth_ >> 5); num_integer++) {
      std::bitset<32> temp_bool;
      for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
        temp_bool.set(bit_count, (dist_fast->DistanceInnerProduct::compare(vertex, &hash_function_[dimension_ * (32 * num_integer + bit_count)], (unsigned)dimension_)) > 0);
      }
      for (unsigned bit_count = 0; bit_count < 32; bit_count++) {
        hashed_set_[num_integer] = (unsigned)(temp_bool.to_ulong());
      }
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
//    std::cout << "HashedSet generation time: " << diff.count() * 1000 << std::endl;;

  std::ofstream file_hashed_set(file_name, std::ios::binary | std::ios::out);
  hashed_set_ = (unsigned int*)(opt_graph_ + node_size * nd_); 
  for (unsigned int i = 0; i < nd_; i++) {
    for (unsigned int j = 0; j < (hash_len >> 2); j++) { 
      file_hashed_set.write((char*)(hashed_set_ + (hash_len >> 2) * i + j), 4);
    }
  }
  file_hashed_set.close();
}
bool IndexSSG::ReadHashFunction (char* file_name) {
  std::ifstream file_hash_function(file_name, std::ios::binary);
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  if (file_hash_function.is_open()) {
    std::cerr << "ReadHashFunction" << std::endl;
    unsigned int hash_bitwidth_temp;
    file_hash_function.read((char*)&hash_bitwidth_temp, sizeof(unsigned int));
    if (hash_bitwidth_ != hash_bitwidth_temp) {
      file_hash_function.close();
      return false;
    }

    hash_function_ = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
    file_hash_function.read((char*)hash_function_, dimension_ * hash_bitwidth_ * sizeof(float));
    file_hash_function.close();
    return true;
  }
  else {
    return false;
  }
}
bool IndexSSG::ReadHashedSet (char* file_name) {
  std::ifstream file_hashed_set(file_name, std::ios::binary);
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  if (file_hashed_set.is_open()) {
    std::cerr << "ReadHashedSet" << std::endl;
    hashed_set_ = (unsigned int*)(opt_graph_ + node_size * nd_);
    for (unsigned int i = 0; i < nd_; i++) {
      for (unsigned int j = 0; j < (hash_len >> 2); j++) {
        file_hashed_set.read((char*)(hashed_set_ + (hash_len >> 2) * i + j), 4);
      }
    }
    file_hashed_set.close();
    
    return true;
  }
  else {
    return false;
  }
}

void IndexSSG::QueryHash (const float* query, unsigned* hashed_query, unsigned hash_size) {
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned int num_integer = 0; num_integer < hash_size; num_integer++) {
    std::bitset<32> temp_bool;
    for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
      temp_bool.set(bit_count, (dist_fast->DistanceInnerProduct::compare(query, &hash_function_[dimension_ * (32 * num_integer + bit_count)], dimension_) > 0));
    }
    for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
      hashed_query[num_integer] = (unsigned)(temp_bool.to_ulong());
    }
  }
}

unsigned int IndexSSG::CandidateSelection (const __m256i* hashed_query_avx, std::vector<HashNeighbor>& selected_pool, const unsigned* neighbors, const unsigned MaxM, const unsigned hash_size) {
  unsigned int prefetch_counter = 0;
  for (; prefetch_counter < (MaxM >> 1); prefetch_counter++) {
    unsigned int id = neighbors[prefetch_counter];
    for (unsigned n = 0; n < hash_size; n += 1)
      _mm_prefetch(hashed_set_ + hash_size * id + n, _MM_HINT_T0);
  }

  unsigned long long hamming_result[4];
  unsigned int selected_pool_size = 0;
  unsigned int selected_pool_size_limit = (unsigned int)ceil(MaxM * tau_);
  HashNeighbor hamming_distance_max(0, 0);
  std::vector<HashNeighbor>::iterator index;

  for (unsigned m = 0; m < MaxM; ++m) {
    if (prefetch_counter < MaxM) {
      unsigned int id = neighbors[prefetch_counter];
      for (unsigned n = 0; n < hash_size; n += 1)
        _mm_prefetch(hashed_set_ + hash_size * id + n, _MM_HINT_T0);
      prefetch_counter++;
    }
    unsigned int id = neighbors[m];
    unsigned int hamming_distance = 0;
    unsigned int* hashed_set_address = hashed_set_ + hash_size * id;
#ifdef __AVX__
  for (unsigned int i = 0; i < (hash_size >> 3); i++) {
    __m256i hashed_set_avx, hamming_result_avx;
    hashed_set_avx = _mm256_loadu_si256((__m256i*)(hashed_set_address));
    hamming_result_avx = _mm256_xor_si256(hashed_query_avx[i], hashed_set_avx);
#ifdef __AVX512VPOPCNTDQ__
    hamming_result_avx = _mm256_popcnt_epi64(hamming_result_avx);
    _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
    for (unsigned int j = 0; j < 4; j++)
      hamming_distance += hamming_result[j];
#else
    _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
    for (unsigned int j = 0; j < 4; j++)
      hamming_distance += _popcnt64(hamming_result[j]);
#endif
    hashed_set_address += 8;
  }
#else
  for (unsigned int num_integer = 0; num_integer < hash_bitwidth / 32; num_integer++) {
    unsigned int hamming_result = hashed_query[num_integer] ^ hashed_set_address[num_integer]; 
    hamming_distance += __builtin_popcount(hamming_result);
  }
#endif
  HashNeighbor cat_hamming_id(id, hamming_distance);
  if ((selected_pool_size_limit < selected_pool_size) && (hamming_distance < hamming_distance_max.distance)) {
    selected_pool[selected_pool_size] = selected_pool[hamming_distance_max.id];
    selected_pool[hamming_distance_max.id] = cat_hamming_id;
    index = std::max_element(selected_pool.begin(), selected_pool.begin() + selected_pool_size_limit);
    hamming_distance_max.id = std::distance(selected_pool.begin(), index);
    hamming_distance_max.distance = selected_pool[hamming_distance_max.id].distance;
    selected_pool_size++;
  }
  else {
    selected_pool[selected_pool_size] = cat_hamming_id;
    selected_pool_size++;
    if (selected_pool_size == selected_pool_size_limit) {
      index = std::max_element(selected_pool.begin(), selected_pool.begin() + selected_pool_size);
      hamming_distance_max.id = std::distance(selected_pool.begin(), index);
      hamming_distance_max.distance = selected_pool[hamming_distance_max.id].distance;
    }
  }
}
  return selected_pool_size_limit;
}
#endif

}  // namespace efanna2e
