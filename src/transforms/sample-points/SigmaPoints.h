/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SIGMAPOINTS_H
#define OPENKALMAN_SIGMAPOINTS_H

#include <stdexcept>
#include "distributions/DistributionTraits.h"
#include "transforms/sample-points/SigmaPointsTypes/Unscented.h"
#include "transforms/sample-points/SigmaPointsTypes/SphericalSimplex.h"

namespace OpenKalman
{
  /*************GeneralSigmaPoints*********
   * @brief This class embodies a set of sample (e.g., sigma) points.
   *
   * Given random variable X:Ω->ℝⁿ, sample points S⊂ℝⁿ are a finite set of
   * samples within ℝⁿ that are specifically arranged so that the
   * weighted distribution of S encodes approximate statistical information
   * (e.g., mean, standard deviation, etc.) about the distribution of X.
   * As implemented in
   * S. Julier. The scaled unscented transformation. In Proceedings of the American
   * Control Conference, Evanston, IL, pages 1108–1114, 2002.
   * This class has only static members, and is not to be instantiated.
   */
  template<typename SigmaPointsType>
  struct SigmaPoints;


  using UnscentedSigmaPointsStateEstimation = SigmaPoints<Unscented<UnscentedParametersStateEstimation>>;
  using UnscentedSigmaPointsParameterEstimation = SigmaPoints<Unscented<UnscentedParametersParameterEstimation>>;
  using UnscentedSigmaPoints = SigmaPoints<Unscented<>>; ///< Same as UnscentedSigmaPointsStateEstimation.
  using SphericalSimplexSigmaPoints = SigmaPoints<SphericalSimplex<>>;


  template<typename SigmaPointsType>
  struct SigmaPoints
  {
  private:
    template<std::size_t i, typename Scalar>
    static constexpr auto cat_dummy_function(const Scalar w) { return w; };

    template<std::size_t dim, typename Weights, typename Scalar, std::size_t ... ints>
    static auto cat_weights(const Scalar W0, std::index_sequence<ints...>)
    {
      constexpr Scalar W = SigmaPointsType::template W<dim, Scalar>();
      return MatrixTraits<Weights>::make(W0, cat_dummy_function<ints>(W)...);
    };

  protected:
    SigmaPoints() {} // Disallow instantiation.

    template<std::size_t dim, typename Weights>
    static auto mean_weights()
    {
      using Scalar = typename MatrixTraits<Weights>::Scalar;
      constexpr auto count = SigmaPointsType::template sigma_point_count<dim>() - 1;
      constexpr auto W0 = SigmaPointsType::template W_m0<dim, Scalar>();
      return cat_weights<dim, Weights>(W0, std::make_index_sequence<count>());
    };

    template<std::size_t dim, typename Weights>
    static auto covariance_weights()
    {
      using Scalar = typename MatrixTraits<Weights>::Scalar;
      constexpr auto count = SigmaPointsType::template sigma_point_count<dim>() - 1;
      constexpr auto W0 = SigmaPointsType::template W_c0<dim, Scalar>();
      return cat_weights<dim, Weights>(W0, std::make_index_sequence<count>());
    };

  public:
    /**
     * @brief Scale and translate normalized sample points based on mean and (square root) covariance.
     * @return A matrix of sample points (each sample point in a column).
     */
    template<typename...Dist, std::enable_if_t<std::conjunction_v<is_Gaussian_distribution<Dist>...>, int> = 0>
    static auto
    sample_points(const Dist&...ds)
    {
      return SigmaPointsType::template sigma_points(ds...);
    }

    template<size_t dim, typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    static auto
    weighted_means(const Arg& y_means)
    {
      static_assert(is_column_vector_v<Arg>);
      static_assert(not is_Euclidean_transformed_v<Arg>);
      constexpr auto count = MatrixTraits<Arg>::columns;
      static_assert(count == SigmaPointsType::template sigma_point_count<dim>(), "Wrong number of sigma points.");
      using Weights = Mean<Axes<count>, typename MatrixTraits<Arg>::template StrictMatrix<count, 1>>;
      return strict(y_means * mean_weights<dim, Weights>());
    }

    template<std::size_t dim, typename X, typename Y,
      std::enable_if_t<is_typed_matrix_v<X>, int>, std::enable_if_t<is_typed_matrix_v<Y>, int> = 0>
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      using CoefficientsIn = typename MatrixTraits<X>::RowCoefficients;
      using CoefficientsOut = typename MatrixTraits<Y>::RowCoefficients;
      constexpr auto count = MatrixTraits<X>::columns;
      static_assert(count == MatrixTraits<Y>::columns);
      using Weights = Mean<Axes<count>, typename MatrixTraits<X>::template StrictMatrix<count, 1>>;
      const auto w_yT = to_diagonal(covariance_weights<dim, Weights>()) * adjoint(y_deviations);
      const auto cross_covariance = strict(x_deviations * w_yT);
      const auto out_covariance = strict(make_Covariance(y_deviations * w_yT));
      return std::tuple{out_covariance, cross_covariance};
    }

    template<std::size_t dim, typename X, typename Y,
      std::enable_if_t<is_typed_matrix_v<X>, int> = 0, std::enable_if_t<is_typed_matrix_v<Y>, int> = 0>
    static auto
    sqrt_covariance(const X& x_deviations, const Y& y_deviations)
    {
      constexpr auto count = MatrixTraits<X>::columns;
      static_assert(count == MatrixTraits<Y>::columns);
      using Weights = Mean<Axes<count>, typename MatrixTraits<X>::template StrictMatrix<count, 1>>;
      auto weights = covariance_weights<dim, Weights>();
      //
      constexpr auto W_c0 = SigmaPointsType::template W_c0<dim, Scalar>();
      if constexpr (W_c0 < 0)
      {
        // Discard first weight and first y-deviation column for now, since square root of weight would be negative.
        const auto [y_deviations_head, y_deviations_tail] = split_horizontal<1, count - 1>(y_deviations);
        const auto [weights_head, weights_tail] = split_vertical<1, count - 1>(weights);
        const auto sqrt_weights_tail = to_diagonal(apply_coefficientwise(weights_tail, [](const auto x){ return std::sqrt(x); }));
        auto out_covariance = LQ_decomposition(sqrt_weights_tail * y_deviations_tail); // This covariance is in Cholesky form.
        static_assert(is_covariance_v<decltype(out_covariance)>);
        //
        // Factor back in the first weight, using a rank update.
        if (Eigen::internal::llt_inplace<Scalar, Eigen::Lower>::rankUpdate(out_covariance, y_deviations_head, W_c0) >= 0)
        {
          throw (std::runtime_error("Posterior covariance is not positive definite."));
        }
        const auto cross_covariance = strict(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
        return std::tuple{out_covariance, cross_covariance};
      }
      else
      {
        static_assert(is_typed_matrix_v<decltype(weights)>);
        const auto sqrt_weights = apply_coefficientwise(weights, [](const auto x){ return std::sqrt(x); });
        static_assert(is_typed_matrix_v<decltype(sqrt_weights)>);
        static_assert(is_typed_matrix_v<decltype(y_deviations)>);
        static_assert(is_typed_matrix_v<decltype(to_diagonal(sqrt_weights) * y_deviations)>);
        const auto out_covariance = LQ_decomposition(to_diagonal(sqrt_weights) * y_deviations); // This covariance is in Cholesky form.
        static_assert(is_covariance_v<decltype(out_covariance)>);
        const auto cross_covariance = strict(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
        return std::tuple{out_covariance, cross_covariance};
      }
    }

  };

}

#endif //OPENKALMAN_SIGMAPOINTS_H
