/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_RLSTRANSFORM_H
#define OPENKALMAN_RLSTRANSFORM_H

#include <cmath>
#include "transforms/internal/TransformBase.h"
#include "distributions/DistributionTraits.h"

namespace OpenKalman {

    /**
     * @brief Propagates a recursive least squares error distribution of parameters, with a forgetting factor λ.
     * Useful for parameter estimation, where the parameter is expected to possibly drift over time
     * @tparam Dist The distribution type (e.g., Distribution).
     * @tparam Scalar The scalar type.
     * @tparam input_dimensions Number of Coeffs of the input distribution.
     * @tparam output_dimensions Number of Coeffs of the output distribution.
     * @tparam noise_type Type of noise.
     */
    template<
        template<typename, typename> typename Dist,
        typename Scalar,
        typename InputCoeffs,
        typename OutputCoeffs = InputCoeffs,
        NoiseType noise_type = NoiseType::none,
        typename ... Noise>
    struct RecursiveLeastSquaresTransform;

    namespace
    {
        template<
            template<typename, typename> typename Dist,
            typename Scalar,
            typename Coeffs,
            typename ... Noise,
            typename CovarianceForm<Dist<Scalar, Coeffs>, int>::type = 0>
        static constexpr std::tuple<const Dist<Scalar, Coeffs>, const Eigen::Matrix<Scalar, Coeffs::size, Coeffs::size>>
        rls_transform(
            const Scalar inv_lambda,
            const Dist<Scalar, Coeffs>& x,
            const Noise& ... n)
        {
            const auto scaled_cov = covariance(x) * inv_lambda;
            const Dist<Scalar, Coeffs> out
                {
                    (mean(x) + ... + mean(n)),
                    (scaled_cov + ... + covariance(n))
                };
            return {out, scaled_cov};
        }

        template<
            template<typename, typename> typename Dist,
            typename Scalar,
            typename Coeffs,
            typename SquareRootForm<Dist<Scalar, Coeffs>, int>::type = 0,
            typename ... Noise>
        static constexpr std::tuple<const Dist<Scalar, Coeffs>, const Eigen::Matrix<Scalar, Coeffs::size, Coeffs::size>>
        rls_transform(
            const Scalar inv_lambda,
            const Dist<Scalar, Coeffs>& x,
            const Noise& ... n)
        {
            const auto scaled_sqrt_cov = sqrt_covariance(x) * std::sqrt(inv_lambda);
            const Dist<Scalar, Coeffs> out
                {
                    (mean(x) + ... + mean(n)),
                    (scaled_sqrt_cov + ... + sqrt_covariance(n))
                };
            return {out, scaled_sqrt_cov.base_matrix() * scaled_sqrt_cov.adjoint()};
        }

    }

    /**
     * @brief Noiseless transform.
     * @tparam Dist The distribution type (e.g., Distribution).
     * @tparam Scalar The scalar type.
     * @tparam Coeffs Coefficients of the distribution.
     */
    template<
        template<typename, typename> typename Dist,
        typename Scalar,
        typename Coeffs>
    struct RecursiveLeastSquaresTransform<Dist, Scalar, Coeffs, Coeffs, NoiseType::none>
        : TransformBase<Dist, Scalar, Coeffs, Coeffs, NoiseType::none>
    {
        using InOutDist = Dist<Scalar, Coeffs>; // Input/Output distribution type
        using CrossCovariance = Eigen::Matrix<Scalar, Coeffs::size, Coeffs::size>; // Cross-covariance type
        using Output = std::tuple<const InOutDist, const CrossCovariance>; // Output type
        using Base = TransformBase<Dist, Scalar, Coeffs, Coeffs, NoiseType::none>;

        /**
         * @brief Construct RLS transform.
         * @param lambda Forgetting factor, slightly less than 1 (e.g., 0.9995).
         */
        explicit RecursiveLeastSquaresTransform(const Scalar lambda = 0.9995)
            : Base
                  {
                      [lambda](const InOutDist& x) -> Output
                      {
                          return rls_transform(1/lambda, x);
                      }
                  } {}

    };


    /**
     * @brief RLS transform with additive noise.
     * The use case for this is dubious.
     * @tparam Dist The distribution type (e.g., Distribution).
     * @tparam Scalar The scalar type.
     * @tparam Coeffs Coefficients of the distribution.
     */
    template<
        template<typename, typename> typename Dist,
        typename Scalar,
        typename Coeffs,
        typename ... Noise>
    struct RecursiveLeastSquaresTransform<Dist, Scalar, Coeffs, Coeffs, NoiseType::additive, Noise...>
        : TransformBase<Dist, Scalar, Coeffs, Coeffs, NoiseType::additive, Noise...>
    {
        using InOutDist = Dist<Scalar, Coeffs>; // Input/Output distribution type
        using CrossCovariance = Eigen::Matrix<Scalar, Coeffs::size, Coeffs::size>; // Cross-covariance type
        using Output = std::tuple<const InOutDist, const CrossCovariance>; // Output type
        using Base = TransformBase<Dist, Scalar, Coeffs, Coeffs, NoiseType::additive, Noise...>;

        /**
         * @brief Constructs transform that adds noise in addition to forgetting factor.
         * @param lambda Forgetting factor, slightly less than 1 (e.g., 0.9995).
         */
        explicit RecursiveLeastSquaresTransform(
            const Scalar lambda = 0.9995)
            : Base
                  {
                      [lambda](const InOutDist& x, const Noise& ... n) -> Output
                      {
                          return rls_transform(1/lambda, x, n...);
                      }
                  } {}

    };


    /**
     * @brief Augmented transform. Use case is limited.
     * @tparam Dist The distribution type (e.g., Distribution).
     * @tparam Scalar The scalar type.
     * @tparam Coeffs Coefficients of the distribution.
     */
    template<
        template<typename, typename> typename Dist,
        typename Scalar,
        typename Coeffs,
        typename ... Noise>
    struct RecursiveLeastSquaresTransform<Dist, Scalar, Coeffs, Coeffs, NoiseType::augmented, Noise...>
        : TransformBase<Dist, Scalar, Coeffs, Coeffs, NoiseType::augmented, Noise...>
    {
        using AugmentedDist = Dist<Scalar, Concatenate<Coeffs, typename Noise::Coefficients...>>; // Augmented input distribution type
        using OutputDist = Dist<Scalar, Coeffs>; // Output distribution type
        using CrossCovariance = Eigen::Matrix<Scalar, Coeffs::size, Coeffs::size>; // Cross-covariance type
        using Output = std::tuple<const OutputDist, const CrossCovariance>; // Output type
        using Base = TransformBase<Dist, Scalar, Coeffs, Coeffs, NoiseType::augmented, Noise...>;

        /**
         * @brief Construct RLS transform.
         * @param lambda Forgetting factor, slightly less than 1 (e.g., 0.9995).
         */
        explicit RecursiveLeastSquaresTransform(const Scalar lambda = 0.9995)
            : Base
                  {
                      [lambda](const AugmentedDist& x) -> Output
                      {
                          return rls_transform(1/lambda, split<Coeffs>(x);

                      }
                  } {}

    };

}


#endif //OPENKALMAN_RLSTRANSFORM_H
