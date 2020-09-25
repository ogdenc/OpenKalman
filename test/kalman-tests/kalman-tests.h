/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef KALMAN_TESTS_H
#define KALMAN_TESTS_H

#include "../tests.h"
#include "../transformations.h"


using namespace OpenKalman;

class kalman_tests : public ::testing::Test
{
public:
  std::random_device rd;
  std::mt19937 gen;

  kalman_tests()
  {
    gen = std::mt19937(rd());
  }

  void SetUp() override {}

  void TearDown() override {}

  ~kalman_tests() override {}

  template<
    typename MeasurementTransform,
    typename MeasurementFunc,
    typename StateDist,
    typename MeasDist,
    typename Scalar = double>
  void parameter_test(
    const MeasurementTransform& m_transform,
    const MeasurementFunc& t,
    const StateDist x_0,
    const typename DistributionTraits<StateDist>::Mean& true_state,
    const MeasDist r,
    const Scalar resolution,
    const int iterations = 100)
  {
    auto p_transform = RecursiveLeastSquaresTransform(0.9995);
    auto filter = KalmanFilter {p_transform, m_transform};
    auto meas_dist = MeasDist {t(true_state), covariance(r)};

    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "true_state" << std::endl << true_state << std::endl << std::flush;
    std::cout << "state_0" << std::endl << x_0 << std::endl << std::flush;
    int count = 0;
    Scalar norm = std::numeric_limits<Scalar>::infinity();

    StateDist x = x_0;

    while (norm > resolution && count++ < iterations)
    {
      const auto z = meas_dist();

      std::cout << "----------------" << std::endl;
      x = filter.predict(x);
      x = filter.update(z, x, t, r);

      std::cout << "state_" << count - 1 << " update" << std::endl << x << std::endl << std::flush;

      norm = (mean(x) - true_state).norm() / std::sqrt(DistributionTraits<StateDist>::dimension);
      std::cout << "sqnorm: " << norm << std::endl << std::flush;
    }
    EXPECT_LT(count, iterations);
    EXPECT_LE(norm, resolution);
  }


  auto get_t2()
  {
    auto theta = trace(randomize<Mean<Axis, Eigen::Matrix<double, 1, 1>>, std::uniform_real_distribution>(-M_PI, M_PI));
    using M22 = Eigen::Matrix<double, 2, 2>;
    auto a = TypedMatrix<Axes<2>, Axes<2>, M22> {std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta)};
    return LinearTransformation(a);
  }

  template<typename Cov, typename Trans>
  inline void rotation_2D(const Trans& transform)
  {
    using M2 = Eigen::Matrix<double, 2, 1>;
    using Mean2 = Mean<Axes<2>, M2>;
    for (int i = 0; i < 5; i++)
    {
      auto true_state = randomize<Mean2, std::uniform_real_distribution>(5.0, 10.0);
      auto x = GaussianDistribution<Axes<2>, M2, Cov> {Mean2 {7.5, 7.5}, Cov::identity()};
      auto meas_cov = Cov {0.01, 0, 0, 0.01};
      auto r = GaussianDistribution<Axes<2>, M2, Cov> {Mean2::zero(), meas_cov};
      parameter_test(transform, get_t2(), x, true_state, r, 0.1, 100);
    }
  }

  inline auto get_t3()
  {
    using M3 = Eigen::Matrix<double, 3, 1>;
    using Mean3 = Mean<Axes<3>, M3>;
    using M33 = Eigen::Matrix<double, 3, 3>;
    auto angles = randomize<Mean3, std::uniform_real_distribution>(-M_PI, M_PI);
    auto ax = TypedMatrix<Axes<3>, Axes<3>, M33> {
      1, 0, 0,
      0, std::cos(angles[0]), -std::sin(angles[0]),
      0, std::sin(angles[0]), std::cos(angles[0])};
    auto ay = TypedMatrix<Axes<3>, Axes<3>, M33> {
      std::cos(angles[0]), 0, std::sin(angles[0]),
      0, 1, 0,
      -std::sin(angles[0]), 0, std::cos(angles[0])};
    auto az = TypedMatrix<Axes<3>, Axes<3>, M33> {
      std::cos(angles[0]), -std::sin(angles[0]), 0,
      std::sin(angles[0]), std::cos(angles[0]), 0,
      0, 0, 1};
    return LinearTransformation(ax * ay * az);
  }

  template<typename Cov, typename Trans>
  void rotation_3D(const Trans& transform)
  {
    using M3 = Eigen::Matrix<double, 3, 1>;
    using Mean3 = Mean<Axes<3>, M3>;
    for (int i = 0; i < 5; i++)
    {
      auto true_state = randomize<Mean3, std::uniform_real_distribution>(5.0, 10.0);
      auto x = GaussianDistribution<Axes<3>, M3, Cov> {Mean3 {7.5, 7.5, 7.5}, Cov::identity()};
      auto meas_cov = Cov {0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.1};
      auto r = GaussianDistribution<Axes<3>, M3, Cov> {Mean3::zero(), meas_cov};
      parameter_test(transform, get_t3(), x, true_state, r, 0.1, 100);
    }
  }

  template<typename Cov, typename Trans>
  inline void radar_2D(const Trans& transform)
  {
    using M2 = Eigen::Matrix<double, 2, 1>;
    using Loc2 = Mean<Axes<2>, M2>;
    using Polar2 = Mean<Polar<>, M2>;
    for (int i = 0; i < 5; i++)
    {
      auto true_state = randomize<Loc2, std::uniform_real_distribution>(5.0, 10.0);
      auto x = GaussianDistribution<Axes<2>, M2, Cov> {Loc2 {7.5, 7.5}, Cov::identity()};
      auto meas_cov = Cov {0.01, 0, 0, M_PI/360};
      auto r = GaussianDistribution<Polar<>, M2, Cov> {Polar2::zero(), meas_cov};
      parameter_test(transform, Cartesian2polar, x, true_state, r, 0.1, 100);
    }
  }


};


#endif //KALMAN_TESTS_H
