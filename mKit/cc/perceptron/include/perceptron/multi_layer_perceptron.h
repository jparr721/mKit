#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;
namespace perceptron {
  class MultiLayerPerceptron {
      public:

        int n_hidden, l2, eta, minibatch_size, seed;
        bool shuffle;

        MultiLayerPerceptron(int n_hidden, int l2, int eta, bool shuffle, int minibatch_size, int seed);
        np::ndarray onehot(np::ndarray, int);
        ~MultiLayerPerceptron();
  };
}
