/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef LINEAR_TESTS_H
#define LINEAR_TESTS_H

#include "../tests.h"

using namespace OpenKalman;

struct linear_tests : public ::testing::Test
{
  linear_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~linear_tests() override {}

protected:
  template<typename Jacobians, typename Covs, std::size_t...ints>
  constexpr auto sumprod(const Jacobians& j, const Covs& c, std::index_sequence<ints...>) const
  {
    return strict(((std::get<ints>(j) * std::get<ints>(c) * adjoint(std::get<ints>(j))) + ...));
  }

public:
  template<typename Transformation, typename Transform, typename InputDist, typename ... Noise>
  ::testing::AssertionResult run_linear_test(
    const Transformation& g,
    const Transform& t,
    const InputDist& in,
    const Noise& ... noise)
  {
    auto x = mean(in);
    const auto p = covariance(in);
    auto y = g(x, mean(noise)...);
    auto [a] = g.jacobian(x);
    auto cross_cov = p*adjoint(a);
    auto jacobians = g.jacobian(x, mean(noise)...);
    auto covariances = std::forward_as_tuple(p, covariance(noise)...);
    auto cov = sumprod(jacobians, covariances, std::make_index_sequence<sizeof...(Noise) + 1>{});
    std::tuple out_true {GaussianDistribution {y, cov}, cross_cov};
    auto res = is_near(t(in, noise...), out_true);
    if (res)
      return ::testing::AssertionSuccess();
    else
      return ::testing::AssertionFailure() << res.message();
  }


  template<std::size_t IN, std::size_t OUT, typename Cov, typename F>
  void run_multiple_linear_tests(Cov cov, const F& f, int N = 20)
  {
    using MatIn = TypedMatrix<Axes<OUT>, Axes<IN>, Eigen::Matrix<double, OUT, IN>>;
    using MatNoise = TypedMatrix<Axes<OUT>, Axes<OUT>, Eigen::Matrix<double, OUT, OUT>>;
    using MIn = Mean<Axes<IN>, Eigen::Matrix<double, IN, 1>>;
    using MNoise = Mean<Axes<OUT>, Eigen::Matrix<double, OUT, 1>>;
    for (int i=1; i<=N; i++)
    {
      auto a = randomize<MatIn, std::uniform_real_distribution>(-i*10., i*10.);
      auto n = randomize<MatNoise, std::uniform_real_distribution>(-double(i), double(i));
      auto g = LinearTransformation(a, n);
      auto t = f(g);
      auto in = GaussianDistribution {MIn::zero(), strict(i * cov)};
      auto b = randomize<MNoise, std::normal_distribution>(0., i*2.);
      auto noise_cov = Covariance {i / 5. * Eigen::Matrix<double, OUT, OUT>::Identity()};
      auto noise = GaussianDistribution {b, noise_cov};
      EXPECT_TRUE(run_linear_test(g, t, in, noise));
    }
  }

};


#endif //LINEAR_TESTS_H
