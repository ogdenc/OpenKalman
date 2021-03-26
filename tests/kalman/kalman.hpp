/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef KALMAN_TESTS_H
#define KALMAN_TESTS_H

#include "../tests.hpp"
#include "../test-transformations.hpp"


using namespace OpenKalman;

class kalman : public ::testing::Test
{
public:
  std::random_device rd;
  std::mt19937 gen;

  kalman()
  {
    gen = std::mt19937(rd());
  }

  void SetUp() override {}

  void TearDown() override {}

  ~kalman() override {}

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
    auto meas_dist = MeasDist {t(true_state), covariance_of(r)};

    int count = 0;
    Scalar norm = std::numeric_limits<Scalar>::infinity();
    StateDist x = x_0;
    while (norm > resolution && count++ < iterations)
    {
      const auto z = meas_dist();
      x = filter.predict(x);
      x = filter.update(z, x, t, r);
      norm = nested_matrix(mean_of(x) - true_state).norm() / std::sqrt(DistributionTraits<StateDist>::dimension);
    }
    if (count >= iterations)
    {
      std::cout << "-----------------------------------------------------------" << std::endl;
      std::cout << "true_state" << std::endl << true_state << std::endl << std::flush;
      std::cout << "state_" << count - 1 << std::endl << x << std::endl << std::flush;
      std::cout << "L2 norm: " << norm << std::endl << std::flush;
    }
    EXPECT_LT(count - 1, iterations);
  }

  template<typename Cov, typename Trans>
  inline void rotation_2D(const Trans& transform);

  template<typename Cov, typename Trans>
  void rotation_3D(const Trans& transform);

  template<typename Cov, typename Trans>
  void artillery_2D(const Trans& transform);

  template<typename Cov, typename Trans>
  void radar_2D(const Trans& transform);


};


#endif //KALMAN_TESTS_H
