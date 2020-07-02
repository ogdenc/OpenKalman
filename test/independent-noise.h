/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_INDEPENDENTNOISE_H
#define OPENKALMAN_TESTS_INDEPENDENTNOISE_H

#include <random>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "variables/interfaces/Eigen3.h"
#include "OpenKalman.h"

using namespace OpenKalman;

template<typename Dist>
struct IndependentNoise : public std::function<const Dist()>
{
  using Scalar = typename DistributionTraits<Dist>::Scalar;
  using Coeffs = typename DistributionTraits<Dist>::Coefficients;
  using M = typename MatrixTraits<Arg1>::template StrictMatrix<>;
  using Cov = typename MatrixTraits<Arg2>::template StrictMatrix<>;

  IndependentNoise(const Scalar factor = 1) :
      std::function<const D()> {[factor]()
        {
          const Mean<Coeffs, M> x {Eigen::Matrix<Scalar, Coeffs::size, 1>::Random().cwiseAbs()};
          const auto L = EigenTriangularMatrix(Eigen::Matrix<Scalar, Coeffs::size, Coeffs::size>::Random().cwiseAbs());
          return Dist {GaussianDistribution(x, L) * std::sqrt(factor)};
        }} {}

};


#endif //OPENKALMAN_TESTS_INDEPENDENTNOISE_H
