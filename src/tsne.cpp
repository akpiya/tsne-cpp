#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using std::vector;

int main() {}

// helper function to determine the gaussian
double exp_plus(vector<double> x, vector<double> y, double sigma) {
  double squared_sum = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    squared_sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return exp(-1.0 * squared_sum / (2 * sigma * sigma));
}

// helper function for equation (4)
double recip_plus(vector<double> x, vector<double> y) {
  double squared_sum = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    squared_sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return (1) / (1 + squared_sum);
}

// For a given sigma, find p_ij
vector<double> apply_sigmai(vector<vector<double>> d, double beta, int index) {
  vector<double> p_i(d.size());

  // Computing denom of eq(1) i.e. normalizing factor
  double normalization = 0.0;
  for (int j = 0; j < d.size(); j++) {
    if (j != index) {
      normalization += exp_plus(d[index], d[j], beta);
    }
  }

  // Populate p_i
  for (int j = 0; j < d.size(); j++) {
    if (j != index) {
      auto val = exp_plus(d[j], d[index], beta);
      p_i[j] = val / normalization;
    } else {
      p_i[j] = 0;
    }
  }
  return p_i;
}

double calculate_entropy(vector<double> p_i) {
  double ans = 0.0;
  for (const auto &p : p_i) {
    ans += -p * log2(p);
  }
  return ans;
}

vector<vector<double>> find_sigmas(vector<vector<double>> x, double tolerance,
                                   double perplexity) {
  auto n = x.size();
  auto d = x[0].size();

  vector<vector<double>> p(n, vector<double>(n));

  auto target_entropy = log2(perplexity);
  auto H = 0;

  for (int i = 0; i < n; ++i) {
    double betamin = std::numeric_limits<double>::min();
    double betamax = std::numeric_limits<double>::max();

    auto iterations = 0;
    vector<double> p_i;

    while (abs(H - target_entropy) > tolerance && iterations < 50) {
      auto midpoint = (betamin + betamax) / 2.0;
      p_i = apply_sigmai(x, midpoint, i);
      H = calculate_entropy(p_i);
      std::cout << "Entropy: " << H << std::endl;

      if (H - target_entropy > 0) {
        betamin = midpoint;
      } else {
        betamax = midpoint;
      }
      iterations += 1;
    }
    p[i] = p_i;
  }
  return p;
}

vector<vector<double>> calculate_q(vector<vector<double>> y) {
  auto n = y.size();
  vector<vector<double>> q(n, vector<double>(n));
  for (int i = 0; i < y.size(); ++i) {
    for (int j = 0; j <= i; ++j) {
      if (i == j) {
        q[i][j] = 0.0;
      } else {
        auto num = recip_plus(y[i], y[j]);
        q[i][j] = num;
        q[j][i] = num;
      }
    }
  }
  double normalization = 0.0;
  for (int i = 0; i < q.size(); ++i) {
    for (int j = 0; j < q.size(); ++j) {
      normalization += q[i][j];
    }
  }
  for (int i = 0; i < q.size(); ++i) {
    for (int j = 0; j < q.size(); ++j) {
      if (i == j) {
        q[i][j] = 0.0000001;
      } else {
        q[i][j] /= normalization;
      }
    }
  }
  return q;
}

vector<vector<double>> tsne(vector<vector<double>> x, int output_dims,
                            double perplexity) {
  auto n = x.size();
  auto d = x[0].size();
  auto max_iterations = 2000;
  auto init_momentum = 0.5;
  auto final_momentum = 0.8;
  auto eta = 500;
  vector<vector<double>> y(n, vector<double>(output_dims));
  vector<vector<double>> prev_y(n, vector<double>(output_dims));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distribution(0, 0.0001);

  // Determining the p_ij matrix
  auto p = find_sigmas(x, 0.000001, perplexity);
  for (int i = 0; i < p.size(); ++i) {
    for (int j = 0; j <= i; ++j) {
      if (i == j) {
        p[i][j] = 0.00000000001;
      } else {
        auto num = (p[i][j] + p[j][i]) / (2 * n);
        p[i][j] = num;
        p[j][i] = num;
      }
    }
  }

  // Populating y matrix
  for (int i = 0; i < y.size(); ++i) {
    for (int j = 0; j < y[i].size(); ++j) {
      y[i][j] = distribution(gen);
    }
  }

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    auto q = calculate_q(y);
    vector<vector<double>> gradient(n, vector<double>(output_dims));
    // Several optimizations can be made here.
    // loop over each y_i
    for (int i = 0; i < n; ++i) {
      // iterate through each component in y_i
      for (int k = 0; k < output_dims; ++k) {
        // iterate through sum in equaiton (5)
        double grad_sum = 0.0;
        for (int j = 0; j < n; ++j) {
          grad_sum += (p[i][j] - q[i][j]) * (y[i][k] - y[j][k]) *
                      recip_plus(y[i], y[j]);
        }
        gradient[i][k] = grad_sum;
      }
    }

    double momentum;
    // update momentum
    if (iteration < 20) {
      momentum = init_momentum;
    } else {
      momentum = final_momentum;
    }

    // y_prev = y_{t-2}
    vector<vector<double>> temp_y(y);

    // Use the gradient to update weights
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < output_dims; ++k) {
        y[i][k] = y[i][k] + eta * gradient[i][k] +
                  momentum * (y[i][k] - prev_y[i][k]);
      }
    }
    prev_y = temp_y;
  }
  return y;
}
