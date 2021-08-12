/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef CONTINUOUSPROPERTYPARTICLEDISTRIBUTION_HPP
#define CONTINUOUSPROPERTYPARTICLEDISTRIBUTION_HPP

#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include "distributions/WeightedParticleDistribution.hpp"
#include "distributions/distributions.hpp"

namespace OpenKalman {

    template<template<int, typename> typename ContinuousDistribution,
            int continuous_dimensions,
            typename Scalar = double,
            typename... OtherProperties>
    struct ContinuousPropertyParticleDistribution:
            WeightedParticleDistribution<
                    Scalar, // Weights for each particle
                    eigen_matrix_t<Scalar, continuous_dimensions, 1>,
                    OtherProperties...>
    {
    public:
        using Mean = eigen_matrix_t<Scalar, continuous_dimensions, 1>;
        using Covariance = eigen_matrix_t<Scalar, continuous_dimensions, continuous_dimensions>;

    protected:
        using Parent = WeightedParticleDistribution<eigen_matrix_t<Scalar, continuous_dimensions, 1>, OtherProperties...>;
        using Parent::Properties;

    public:
        const auto getDistribution() const
        {
            using Acc = std::tuple<Mean, Mean>;
            auto acc = std::accumulate(
                    begin(),
                    end(),
                    Acc {Mean::Zero(), Covariance::Zero()},
                    [](const Acc& a, const Properties& p) -> Acc {
                        Mean x = p.template get<0>() * p.template get<1>(); // x == weighted value
                        return {a.template get<0>() + x, a.template get<1>() + x * x.transpose()};
                    }
            );
            // assume weights are normalized
            Mean mean = acc.template get<0>;
            Covariance covariance = acc.template get<1> - mean * mean.transpose();
            return ContinuousDistribution<continuous_dimensions, Scalar>(GaussianDistribution(mean, covariance));
        }


    };
}

#endif //CONTINUOUSPROPERTYPARTICLEDISTRIBUTION_HPP
