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
    static auto cat_weights(const Scalar w0, std::index_sequence<ints...>)
    {
      constexpr Scalar w = SigmaPointsType::template W<dim, Scalar>();
      constexpr auto count = sizeof...(ints) + 1;
      return Weights {w0, cat_dummy_function<ints>(w)...};
    };

  protected:
    SigmaPoints() {} ///< Instantiation is disallowed.

    template<std::size_t dim, typename Weights>
    static auto mean_weights()
    {
      using Scalar = typename MatrixTraits<Weights>::Scalar;
      constexpr auto count = SigmaPointsType::template sigma_point_count<dim>() - 1;
      constexpr auto w0 = SigmaPointsType::template W_m0<dim, Scalar>();
      return cat_weights<dim, Weights>(w0, std::make_index_sequence<count>());
    };

    template<std::size_t dim, typename Weights>
    static auto covariance_weights()
    {
      using Scalar = typename MatrixTraits<Weights>::Scalar;
      constexpr auto count = SigmaPointsType::template sigma_point_count<dim>() - 1;
      constexpr auto w0 = SigmaPointsType::template W_c0<dim, Scalar>();
      return cat_weights<dim, Weights>(w0, std::make_index_sequence<count>());
    };

  public:
    /**
     * @brief Scale and translate normalized sample points based on mean and (square root) covariance.
     * @return A tuple of matrices of sample points (each sample point in a column).
     */
    template<typename...Dist, std::enable_if_t<std::conjunction_v<is_Gaussian_distribution<Dist>...>, int> = 0>
    static auto
    sample_points(const Dist&...ds)
    {
      return SigmaPointsType::template sigma_points(ds...);
    }

    template<size_t dim, typename Arg, std::enable_if_t<is_Euclidean_mean_v<Arg>, int> = 0>
    static auto
    weighted_means(const Arg& y_means)
    {
      static_assert(is_column_vector_v<Arg>);
      constexpr auto count = MatrixTraits<Arg>::columns;
      static_assert(count == SigmaPointsType::template sigma_point_count<dim>(), "Wrong number of sigma points.");
      using Weights = TypedMatrix<Axes<count>, Axis, typename MatrixTraits<Arg>::template StrictMatrix<count, 1>>;
      return strict(from_Euclidean(y_means * mean_weights<dim, Weights>()));
    }

    template<typename InputDist, typename ... NoiseDist, typename X, typename Y>
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      static_assert(is_typed_matrix_v<X> and is_typed_matrix_v<Y>);
      static_assert(is_equivalent_v<typename MatrixTraits<X>::RowCoefficients,
        typename DistributionTraits<InputDist>::Coefficients>);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + ... + DistributionTraits<NoiseDist>::dimension);
      constexpr auto count = MatrixTraits<X>::columns;
      static_assert(count == MatrixTraits<Y>::columns);
      using Weights = TypedMatrix<Axes<count>, Axis, typename MatrixTraits<X>::template StrictMatrix<count, 1>>;
      auto weights = covariance_weights<dim, Weights>();
      if constexpr(is_Cholesky_v<InputDist>)
      {
        using Scalar = typename MatrixTraits<X>::Scalar;
        //
        constexpr auto W_c0 = SigmaPointsType::template W_c0<dim, Scalar>();
        if constexpr (W_c0 < 0)
        {
          // Discard first weight and first y-deviation column for now, since square root of weight would be negative.
          const auto [y_deviations_head, y_deviations_tail] = split_horizontal<1, count - 1>(y_deviations);
          const auto [weights_head, weights_tail] = split_vertical<Axis, Axes<count - 1>>(weights);
          const auto sqrt_weights_tail = apply_coefficientwise(weights_tail, [](const auto x){ return std::sqrt(x); });
          auto out_covariance = Covariance {LQ_decomposition(y_deviations_tail * to_diagonal(sqrt_weights_tail))};
          rank_update(out_covariance, y_deviations_head, W_c0); ///< Factor back in the first weight, using a rank update.
          auto cross_covariance = strict(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
          return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
        }
        else
        {
          const auto sqrt_weights = apply_coefficientwise(weights, [](const auto x){ return std::sqrt(x); });
          auto out_covariance = Covariance {LQ_decomposition(y_deviations * to_diagonal(sqrt_weights))};
          auto cross_covariance = strict(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
          return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
        }
      }
      else
      {
        const auto w_yT = to_diagonal(weights) * adjoint(y_deviations);
        auto out_covariance = strict(make_Covariance(y_deviations * w_yT));
        auto cross_covariance = strict(x_deviations * w_yT);
        return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
      }
    }

  };

}

#endif //OPENKALMAN_SIGMAPOINTS_H
