/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_NONLINEAR_SQRT_TESTS_ST_H
#define OPENKALMAN_NONLINEAR_SQRT_TESTS_ST_H

#include "nonlinear-sqrt-tests.h"
#include "transforms/classes/SamplePointsTransform.h"
#include "distributions/GaussianDistribution.h"

using namespace OpenKalman;

template<
    template<template<typename, typename> typename, typename, typename, int...> typename Sample,
    typename Scalar,
    typename InCoeffs,
    typename OutCoeffs,
    NoiseType noise_type,
    typename ... SArgs>
::testing::AssertionResult sqrtComparison(
    const VectorTransformation<Scalar, InCoeffs, OutCoeffs, noise_type>& transformation,
    const GaussianDistribution<Scalar, InCoeffs>& d,
    const SArgs& ... sArgs)
{
    const auto id = Covariance<Scalar, OutCoeffs>::Matrix::Identity();
    if constexpr (noise_type == NoiseType::augmented)
    {
        const Sample<GaussianDistribution, Scalar, Concatenate<InCoeffs, OutCoeffs>> s1 {sArgs ...};
        const Sample<SquareRootGaussianDistribution, Scalar, Concatenate<InCoeffs, OutCoeffs>> s2 {sArgs ...};
        const SamplePointsTransform t {s1, transformation};
        const SamplePointsTransform t_sqrt {s2, transformation};
        using Mean = Mean<Scalar, OutCoeffs>;
        const auto res = std::get<0>(t(d, GaussianDistribution<Scalar, OutCoeffs> {id}));
        const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InCoeffs> {d},
                                                 SquareRootGaussianDistribution<Scalar, OutCoeffs> {id}));
        return is_near(res, res_sqrt);
    }
    else
    {
        const Sample<GaussianDistribution, Scalar, InCoeffs> s1 {sArgs ...};
        const Sample<SquareRootGaussianDistribution, Scalar, InCoeffs> s2 {sArgs ...};
        const SamplePointsTransform t {s1, transformation};
        const SamplePointsTransform t_sqrt {s2, transformation};
        if constexpr (noise_type == NoiseType::none)
        {
            const auto res = std::get<0>(t(d));
            const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InCoeffs> {d}));
            return is_near(res, res_sqrt);
        } else
        {
            const auto res = std::get<0>(t(d, GaussianDistribution<Scalar, OutCoeffs> {id}));
            const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InCoeffs> {d},
                                                     SquareRootGaussianDistribution<Scalar, OutCoeffs> {id}));
            return is_near(res, res_sqrt);
        }
    }
}


#endif //OPENKALMAN_NONLINEAR_SQRT_TESTS_ST_H
