/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef LINEAR_GTEST_HPP
#define LINEAR_GTEST_HPP

#include "transformations/tests/transformations.gtest.hpp"

#include "transforms/transforms.hpp"


namespace OpenKalman::test
{
  namespace detail
  {
    template<typename Jacobians, typename Covs, std::size_t...ints>
    constexpr auto sumprod(const Jacobians& j, const Covs& c, std::index_sequence<ints...>)
    {
      return make_self_contained(((std::get<ints>(j) * std::get<ints>(c) * adjoint(std::get<ints>(j))) + ...));
    }
  }


  template<typename Transformation, typename Transform, typename InputDist, typename ... Noise>
  ::testing::AssertionResult run_linear_test(
    double err,
    const Transformation& g,
    const Transform& t,
    InputDist&& in,
    Noise&& ... noise)
  {
    auto x = mean_of(in);
    auto p = covariance_of(in);
    auto y = g(x, mean_of(noise)...);
    auto [a] = g.jacobian(x);
    auto cross_cov = p*adjoint(a);
    auto jacobians = g.jacobian(x, mean_of(noise)...);
    auto covariances = std::forward_as_tuple(std::move(p), covariance_of(noise)...);
    auto cov = detail::sumprod(jacobians, covariances, std::make_index_sequence<sizeof...(Noise) + 1>{});
    auto out_true = make_tuple(GaussianDistribution {y, cov}, std::move(cross_cov));

    auto out = t.transform_with_cross_covariance(std::forward<InputDist>(in), g, std::forward<Noise>(noise)...);
    auto res = is_near(out, out_true, err);
    if (res)
      return ::testing::AssertionSuccess();
    else
      return ::testing::AssertionFailure() << res.message();
  }


  template<typename Transformation, typename Transform, typename InputDist, typename ... Noise>
  ::testing::AssertionResult run_identity_test(
    double err,
    const Transformation& g,
    const Transform& t,
    InputDist&& in,
    Noise&& ... noise)
  {
    auto x = mean_of(in);
    auto p = covariance_of(in);
    auto y = g(x, mean_of(noise)...);
    auto [a] = g.jacobian(x);
    auto cross_cov = p*adjoint(a);
    auto jacobians = g.jacobian(x, mean_of(noise)...);
    auto covariances = std::forward_as_tuple(std::move(p), covariance_of(noise)...);
    auto cov = detail::sumprod(jacobians, covariances, std::make_index_sequence<sizeof...(Noise) + 1>{});
    auto out_true = make_tuple(GaussianDistribution {y, cov}, std::move(cross_cov));

    auto out = t.transform_with_cross_covariance(std::forward<InputDist>(in), std::forward<Noise>(noise)...);
    auto res = is_near(out, out_true, err);
    if (res)
      return ::testing::AssertionSuccess();
    else
      return ::testing::AssertionFailure() << res.message();
  }


  template<std::size_t IN, std::size_t OUT, typename Cov, typename T>
  void run_multiple_linear_tests(Cov cov, const T& t, double err = 1e-6, int N = 5)
  {
    using MatIn = Matrix<Axes<OUT>, Axes<IN>, eigen_matrix_t<double, OUT, IN>>;
    using MatNoise = Matrix<Axes<OUT>, Axes<OUT>, eigen_matrix_t<double, OUT, OUT>>;
    using MIn = Mean<Axes<IN>, eigen_matrix_t<double, IN, 1>>;
    using MNoise = Mean<Axes<OUT>, eigen_matrix_t<double, OUT, 1>>;
    for (int i=1; i<=N; i++)
    {
      auto a = randomize<MatIn>(std::uniform_real_distribution<double> {-i, i});
      auto n = randomize<MatNoise>(std::uniform_real_distribution<double> {-0.1 * i, 0.1 * i});
      auto g = LinearTransformation(a, n);
      auto in = GaussianDistribution {MIn::zero(), cov};
      auto b = randomize<MNoise>(std::normal_distribution<double> {0., i*2.});
      auto noise_cov = Covariance {0.5 * eigen_matrix_t<double, OUT, OUT>::Identity()};
      auto noise = GaussianDistribution {b, noise_cov};
      EXPECT_TRUE(run_linear_test(err, g, t, in, noise));
    }
  }


  template<std::size_t DIM, typename Cov>
  void run_multiple_identity_tests(Cov&& cov, double err = 1e-6, int N = 5)
  {
    using MatIn = Matrix<Axes<DIM>, Axes<DIM>, eigen_matrix_t<double, DIM, DIM>>;
    using MatNoise = Matrix<Axes<DIM>, Axes<DIM>, eigen_matrix_t<double, DIM, DIM>>;
    using MIn = Mean<Axes<DIM>, eigen_matrix_t<double, DIM, 1>>;
    using MNoise = Mean<Axes<DIM>, eigen_matrix_t<double, DIM, 1>>;
    for (int i=1; i<=N; i++)
    {
      auto a = MatIn::identity();
      auto n = MatNoise::identity();
      auto g = LinearTransformation(a, n);
      auto t = IdentityTransform();
      auto in = GaussianDistribution {MIn::zero(), make_self_contained(i * std::forward<Cov>(cov))};
      auto b = randomize<MNoise>(std::normal_distribution {0., i*2.});
      auto noise_cov = Covariance {i / 5. * eigen_matrix_t<double, DIM, DIM>::Identity()};
      auto noise = GaussianDistribution {b, noise_cov};
      EXPECT_TRUE(run_identity_test(err, g, t, in, noise));
    }
  }

}


#endif //LINEAR_GTEST_HPP
