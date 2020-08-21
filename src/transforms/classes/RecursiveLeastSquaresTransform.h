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
  struct RecursiveLeastSquaresTransform
  {
    explicit RecursiveLeastSquaresTransform(const Scalar lambda = 0.9995)
     : inv_lambda(1/lambda) {}

    /// Apply the RLS transform on an input distribution. Any noise distributions are treated as additive.
    template<typename InputDist, typename ... NoiseDist>
    auto operator()(InputDist&& in, const NoiseDist& ...n) const
    {
      static_assert(std::conjunction_v<is_equivalent<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDist>::Coefficients>...>,
        "Input and Noise distributions must be the same size and an equivalent type.");

      auto scaled_cov = in * inv_lambda;
      auto out_covariance = strict((scaled_cov + ... + n));
      auto cross_covariance = TypedMatrix {scaled_cov};
      return std::tuple {std::move(out_covariance), std::move(cross_covariance)};
    }

  protected:
    const Scalar inv_lambda;

  };

}


#endif //OPENKALMAN_RLSTRANSFORM_H
