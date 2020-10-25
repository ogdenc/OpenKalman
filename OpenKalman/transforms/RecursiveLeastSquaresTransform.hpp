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

namespace OpenKalman
{
  /**
   * @brief Propagates a recursive least squares error distribution of parameters, with a forgetting factor λ.
   * Useful for parameter estimation, where the parameter is expected to possibly drift over time
   * @tparam Scalar The scalar type.
   */
  template<typename Scalar = double>
  struct RecursiveLeastSquaresTransform : TransformBase<RecursiveLeastSquaresTransform<Scalar>>
  {
    explicit RecursiveLeastSquaresTransform(const Scalar lambda = 0.9995)
     : inv_lambda(1/lambda) {}

    /**
     * Apply the RLS transform on an input distribution. Any noise distributions are treated as additive.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDists Noise distribution.
     **/
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDists>...>, int> = 0>
    auto operator()(const InputDist& x, const NoiseDists& ...ns) const
    {
      static_assert(std::conjunction_v<is_equivalent<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients>...>,
        "Input and Noise distributions must be the same size and an equivalent type.");
      const auto scaled_x = GaussianDistribution {mean(x), covariance(x) * inv_lambda};
      return strict((scaled_x + ... + ns));
    }

    /**
     * Perform RLS transform, also returning the cross-covariance.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distributions.
     **/
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDists>...>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const NoiseDists& ...ns) const
    {
      static_assert(std::conjunction_v<is_equivalent<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients>...>,
        "Input and Noise distributions must be the same size and an equivalent type.");
      auto scaled_cov = strict(covariance(x) * inv_lambda);
      const auto scaled_x = GaussianDistribution {mean(x), scaled_cov};
      auto y = strict((scaled_x + ... + ns));
      auto cross = Matrix {scaled_cov};
      return std::tuple {std::move(y), std::move(cross)};
    }

  protected:
    const Scalar inv_lambda;

  };

}


#endif //OPENKALMAN_RLSTRANSFORM_H
