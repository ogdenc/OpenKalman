/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SCALEDSIGMAPOINTSBASE_HPP
#define OPENKALMAN_SCALEDSIGMAPOINTSBASE_HPP

namespace OpenKalman::internal
{
  /*************ScaledSigmaPointsBase*********
   * @brief Base class that embodies a scaled set of sample (e.g., sigma) points.
   *
   * Given random variable X:Ω->ℝⁿ, sample points S⊂ℝⁿ are a finite set of
   * samples within ℝⁿ that are specifically arranged so that the
   * weighted distribution of S encodes approximate statistical information
   * (e.g., mean, standard deviation, etc.) about the distribution of X.
   * As implemented in
   * S. Julier. The scaled unscented transformation. In Proceedings of the American
   * Control Conference, Evanston, IL, pages 1108–1114, 2002.
   */
  template<typename Derived, typename Parameters>
  struct ScaledSigmaPointsBase
  {
    /**
     * Weight for the first sigma point when calculating the posterior mean.
     * See Julier Eq. 15 (not Eq. 24, which appears to be wrong).
     * @tparam dim Number of dimensions of the input variables (including noise).
     * @tparam Scalar Scalar type (e.g., double).
     * @return Weight for the first sigma point.
     */
    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    W_m0()
    {
      constexpr Scalar alpha = Parameters::alpha;
      constexpr Scalar W0 = Derived::template unscaled_W0<dim, Scalar>();
      return (W0 - 1) / (alpha * alpha) + 1;
    };

    /**
     * Weight for the first sigma point when calculating the posterior covariance.
     * See Julier Eq. 27.
     * @tparam dim Number of dimensions of the input variables (including noise).
     * @tparam Scalar Scalar type (e.g., double).
     * @return Weight for the first sigma point.
     */
    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    W_c0()
    {
      constexpr Scalar alpha = Parameters::alpha;
      constexpr Scalar beta = Parameters::beta;
      return W_m0<dim, Scalar>() + 1 - alpha * alpha + beta;
    };

    /**
     * Weight for each sigma point other than the first one, when calculating posterior mean and covariance.
     * See Julier Eq. 15 (not Eq. 24, which appears to be wrong), Eq. 27.
     * @tparam dim Number of dimensions of the input variables (including noise).
     * @tparam Scalar Scalar type (e.g., double).
     * @return Weights for each sigma point.
     */
    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    W()
    {
      constexpr Scalar alpha = Parameters::alpha;
      constexpr Scalar W0 = Derived::template unscaled_W<dim, Scalar>();
      return W0 / (alpha * alpha);
    };

  };

}

#endif //OPENKALMAN_SCALEDSIGMAPOINTSBASE_HPP
