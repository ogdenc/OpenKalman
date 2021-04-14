/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MIXTUREOFCONTINUOUSDISTRIBUTIONS_HPP
#define OPENKALMAN_MIXTUREOFCONTINUOUSDISTRIBUTIONS_HPP

#include <vector>
#include <tuple>
#include "distributions/ParticleDistribution.hpp"

namespace OpenKalman {

    /**
     * Weighted mixture of continuous (e.g., Gaussian) distributions
     * \tparam ContinuousDistribution The distribution (e.g., Gaussian, SquareRootGaussian)
     * \tparam continuous_dimensions Number of continuous dimensions of each distribution
     * \tparam Scalar The number type (e.g., double)
     * \tparam OtherProperties Any other properties that each distribution may have
     */
    template<template<int, bool, typename> typename ContinuousDistribution,
            int continuous_dimensions,
            typename Scalar = double,
            typename... OtherProperties>
    struct MixtureOfContinuousDistributions:
            ParticleDistribution<
                    ContinuousDistribution<continuous_dimensions, false, Scalar>,
                    Scalar, // weights
                    OtherProperties...>
    {
    public:
        MixtureOfContinuousDistributions()
        {
            ;
        }

        const Mean mean() const
        {
            ContinuousDistribution<continuous_dimensions, false, Scalar> dist;
            for (auto& n : this) {
                s
            }
            return x;
        }


        const Covariance covariance() const
        {
            return P_xx;
        }


        const Covariance sqrt_covariance() const
        {
            auto S_xx = P_xx.llt();
            if (S_xx.info() != Eigen::Success) {
                throw (std::runtime_error("GaussianDistribution: covariance is not positive definite"));
            }
            return S_xx.matrixL().toDenseMatrix();
        }

    };

}

#endif //OPENKALMAN_MIXTUREOFCONTINUOUSDISTRIBUTIONS_HPP
