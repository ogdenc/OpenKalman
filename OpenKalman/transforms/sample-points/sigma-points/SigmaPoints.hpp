/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SIGMAPOINTS_HPP
#define OPENKALMAN_SIGMAPOINTS_HPP

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
    template<std::size_t, typename Scalar>
    static constexpr auto cat_dummy_function(const Scalar w) { return w; };

    template<std::size_t dim, typename Weights, typename Scalar, std::size_t ... ints>
    static auto cat_weights(const Scalar w0, std::index_sequence<ints...>)
    {
      constexpr Scalar w = SigmaPointsType::template W<dim, Scalar>();
      return Weights {w0, cat_dummy_function<ints>(w)...};
    };

  protected:
    SigmaPoints() {} ///< Instantiation is disallowed.

    template<std::size_t dim, typename Weights>
    static auto mean_weights()
    {
      using Scalar = typename MatrixTraits<Weights>::Scalar;
      constexpr auto count = MatrixTraits<Weights>::dimension;
      constexpr auto w0 = SigmaPointsType::template W_m0<dim, Scalar>();
      return cat_weights<dim, Weights>(w0, std::make_index_sequence<count - 1>());
    };

    template<std::size_t dim, typename Weights>
    static auto covariance_weights()
    {
      using Scalar = typename MatrixTraits<Weights>::Scalar;
      constexpr auto count = MatrixTraits<Weights>::dimension;
      constexpr auto w0 = SigmaPointsType::template W_c0<dim, Scalar>();
      return cat_weights<dim, Weights>(w0, std::make_index_sequence<count - 1>());
    };

  public:
    /**
     * @brief Scale and translate normalized sample points based on mean and (square root) covariance.
     * @return A tuple of matrices of sample points (each sample point in a column).
     */
    template<typename...Dist>
    static auto
    sample_points(const Dist&...ds)
    {
      static_assert(std::conjunction_v<is_Gaussian_distribution<Dist>...>);
      return SigmaPointsType::template sigma_points(ds...);
    }

    template<std::size_t dim, typename YMeans, std::enable_if_t<is_Euclidean_mean_v<YMeans>, int> = 0>
    static auto
    weighted_means(const YMeans& y_means)
    {
      static_assert(is_column_vector_v<YMeans>);
      constexpr auto count = MatrixTraits<YMeans>::columns;
      static_assert(count == SigmaPointsType::template sigma_point_count<dim>(), "Wrong number of sigma points.");
      using Weights = Matrix<Axes<count>, Axis, strict_matrix_t<YMeans, count, 1>>;
      return strict(y_means * mean_weights<dim, Weights>());
    }

    template<std::size_t dim, typename InputDist, bool return_cross = false, typename X, typename Y>
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      static_assert(is_typed_matrix_v<X> and is_typed_matrix_v<Y>);
      static_assert(is_equivalent_v<typename MatrixTraits<X>::RowCoefficients,
        typename DistributionTraits<InputDist>::Coefficients>);
      constexpr auto count = MatrixTraits<X>::columns;
      static_assert(count == MatrixTraits<Y>::columns);
      static_assert(count == SigmaPointsType::template sigma_point_count<dim>(), "Wrong number of sigma points.");
      using Weights = Matrix<Axes<count>, Axis, strict_matrix_t<X, count, 1>>;
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
          auto out_covariance = LQ_decomposition(y_deviations_tail * to_diagonal(sqrt_weights_tail));
          rank_update(out_covariance, y_deviations_head, W_c0); ///< Factor back in the first weight, using a rank update.
          if constexpr (return_cross)
          {
            auto cross_covariance = strict(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
            return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
          }
          else
          {
            return out_covariance;
          }
        }
        else
        {
          const auto sqrt_weights = apply_coefficientwise(weights, [](const auto x){ return std::sqrt(x); });
          auto out_covariance = Covariance {LQ_decomposition(y_deviations * to_diagonal(sqrt_weights))};
          if constexpr (return_cross)
          {
            auto cross_covariance = strict(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
            return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
          }
          else
          {
            return out_covariance;
          }
        }
      }
      else
      {
        const auto w_yT = to_diagonal(weights) * adjoint(y_deviations);
        auto out_covariance = strict(make_Covariance(y_deviations * w_yT));
        if constexpr (return_cross)
        {
          auto cross_covariance = strict(x_deviations * w_yT);
          return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
        }
        else
        {
          return out_covariance;
        }
      }
    }

  };

}

#endif //OPENKALMAN_SIGMAPOINTS_HPP
