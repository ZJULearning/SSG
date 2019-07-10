//
// Created by 付聪 on 2017/6/21.
//

#include "index_random.h"
#include "index_ssg.h"
#include "util.h"

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cout << "./run data_file nn_graph_path L R Angle save_graph_file [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc == 8) {
    unsigned seed = (unsigned)atoi(argv[7]);
    srand(seed);
    std::cout << "Using Seed " << seed << std::endl;
  }

  std::cerr << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim;
  float* data_load = nullptr;
  data_load = efanna2e::load_data(argv[1], points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);

  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  float A = (float)atof(argv[5]);

  std::cout << "L = " << L << ", ";
  std::cout << "R = " << R << ", ";
  std::cout << "Angle = " << A << std::endl;
  std::cout << "KNNG = " << nn_graph_path << std::endl;

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexSSG index(dim, points_num, efanna2e::L2,
                           (efanna2e::Index*)(&init_index));

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<float>("A", A);
  paras.Set<unsigned>("n_try", 10);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);

  std::cerr << "Output SSG Path: " << argv[6] << std::endl;

  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "Build Time: " << diff.count() << "\n";

  index.Save(argv[6]);

  return 0;
}
