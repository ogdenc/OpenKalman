/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef KALMAN_GTEST_HPP
#define KALMAN_GTEST_HPP

#include "transformations/tests/transformations.gtest.hpp"

#include "filters/filters.hpp"


namespace OpenKalman::test
{
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
    using namespace OpenKalman;

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
      norm = nested_object(mean_of(x) - true_state).norm() / std::sqrt(index_dimension_of_v<StateDist, 0>);
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

}

#endif //KALMAN_GTEST_HPP
