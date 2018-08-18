#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <vector>

namespace perceptron {
  class MultiLayerPerceptron {
      public:
        int n_hidden, l2, eta, minibatch_size, seed;
        bool shuffle;

        MultiLayerPerceptron(int n_hidden, int l2, int eta, bool shuffle, int minibatch_size, int seed);
        std::vector<int> oneHot(std::vector<int>, int);

  };
}
