/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of internal::ScaledSigmaPointsBase.
 */

#ifndef OPENKALMAN_SCALEDSIGMAPOINTSBASE_HPP
#define OPENKALMAN_SCALEDSIGMAPOINTSBASE_HPP

namespace OpenKalman::internal
{

  /**
   * \internal
   * \brief Base class that embodies a scaled set of sample (e.g., sigma) points.
   * \details Given random variable X:Ω->ℝⁿ, sample points S⊂ℝⁿ are a finite set of
   * samples within ℝⁿ that are specifically arranged so that the
   * weighted distribution of S encodes approximate statistical information
   * (e.g., mean, standard deviation, etc.) about the distribution of X.
   * As implemented in
   * S. Julier. The scaled unscented tests. In Proceedings of the American
   * Control Conference, Evanston, IL, pages 1108–1114, 2002.
   * \tparam Derived The derived class
   * constant expressions.
   */
  template<typename Derived>
  struct ScaledSigmaPointsBase
  {

  private:

    /// Instantiation is disallowed.
    ScaledSigmaPointsBase() {}


    /*
     * \brief Weight for each sigma point other than the first one, when calculating posterior mean and covariance.
     * \details See Julier Eq. 15 (not Eq. 24, which appears to be wrong), Eq. 27.
     * \tparam dim Number of dimensions of the input variables (including noise).
     * \tparam Scalar Scalar type (e.g., double).
     */
    template<std::size_t dim>
    static constexpr auto W()
    {
      return Derived::template unscaled_W<dim>() / (Derived::alpha * Derived::alpha);
    }


    /*
     * \brief Weight for the first sigma point when calculating the posterior mean.
     * \details See Julier Eq. 15 (not Eq. 24, which appears to be wrong).
     * \tparam dim Number of dimensions of the input variables (including noise).
     * \tparam Scalar Scalar type (e.g., double).
     */
    template<std::size_t dim>
    static constexpr auto W_m0()
    {
      return (Derived::template unscaled_W0<dim>() - 1) / (Derived::alpha * Derived::alpha) + 1;
    }


    /*
     * \brief Weight for the first sigma point when calculating the posterior covariance.
     * \details See Julier Eq. 27.
     * \tparam dim Number of dimensions of the input variables (including noise).
     * \tparam Scalar Scalar type (e.g., double).
     * \return Weight for the first sigma point.
     */
    template<std::size_t dim>
    static constexpr auto W_c0()
    {
      return W_m0<dim>() + 1 - Derived::alpha * Derived::alpha + Derived::beta;
    }


    template<std::size_t, typename Scalar>
    static constexpr auto cat_dummy_function(const Scalar w) { return w; };


    template<std::size_t dim, typename Weights, typename Scalar, std::size_t ... ints>
    static auto cat_weights(const Scalar w0, std::index_sequence<ints...>)
    {
      return Weights {w0, cat_dummy_function<ints, Scalar>(W<dim>())...};
    };


    template<std::size_t dim, typename Weights>
    static auto mean_weights()
    {
      using Scalar = scalar_type_of_t<Weights>;
      constexpr auto count = index_dimension_of_v<Weights, 0>;
      return cat_weights<dim, Weights, Scalar>(W_m0<dim>(), std::make_index_sequence<count - 1> {});
    };


    template<std::size_t dim, typename Weights>
    static auto covariance_weights()
    {
      using Scalar = scalar_type_of_t<Weights>;
      constexpr auto count = index_dimension_of_v<Weights, 0>;
      return cat_weights<dim, Weights, Scalar>(W_c0<dim>(), std::make_index_sequence<count - 1> {});
    };

  public:

#ifdef __cpp_concepts
    template<std::size_t dim, typed_matrix YMeans> requires has_untyped_index<YMeans, 1> and
      (index_dimension_of_v<YMeans, 0> == coordinates::stat_dimension_of_v<vector_space_descriptor_of_t<YMeans, 0>>)
#else
    template<std::size_t dim, typename YMeans, std::enable_if_t<typed_matrix<YMeans> and has_untyped_index<YMeans, 1> and
      (index_dimension_of<YMeans, 0>::value == coordinates::stat_dimension_of_v<vector_space_descriptor_of_t<YMeans, 0>>), int> = 0>
#endif
    static auto
    weighted_means(YMeans&& y_means)
    {
      static_assert(index_dimension_of_v<YMeans, 1> == Derived::template sigma_point_count<dim>);
      using Weights = dense_writable_matrix_t<YMeans, Layout::none, scalar_type_of_t<YMeans>,
        std::tuple<index_dimension_of<YMeans, 1>, Axis>>;
      return make_self_contained(std::forward<YMeans>(y_means) * mean_weights<dim, Weights>());
    }


    /**
     * \brief Calculate the posterior covariance, given prior and posterior deviations from the sigma points
     * \tparam dim The total number of dimensions of all inputs.
     * \tparam InputDist The prior distribution.
     * \tparam return_cross Whether to return a cross-covariance.
     * \tparam X The scaled sigma points for the prior distribution (the mean is translated to origin).
     * \tparam Y The transformed sigma points for the posterior distribution (the mean is translated to origin).
     * \return The posterior covariance, or (if return_cross, then a tuple comprising the posterior covariance
     * and the cross-covariance.
     */
#ifdef __cpp_concepts
    template<std::size_t dim, typename InputDist, bool return_cross = false, typed_matrix X, typed_matrix Y> requires
      (index_dimension_of_v<X, 1> == index_dimension_of_v<Y, 1>) and
      compares_with<vector_space_descriptor_of_t<X, 0>, typename DistributionTraits<InputDist>::StaticDescriptor>
#else
    template<std::size_t dim, typename InputDist, bool return_cross = false, typename X, typename Y, std::enable_if_t<
      typed_matrix<X> and typed_matrix<Y> and (index_dimension_of<X, 1>::value == index_dimension_of<Y, 1>::value) and
      compares_with<vector_space_descriptor_of_t<X, 0>, typename DistributionTraits<InputDist>::StaticDescriptor>,
        int> = 0>
#endif
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      static_assert(index_dimension_of_v<X, 1> == Derived::template sigma_point_count<dim>);
      constexpr auto count = index_dimension_of_v<X, 1>;
      using Weights = dense_writable_matrix_t<X, Layout::none, scalar_type_of_t<X>, std::tuple<Dimensions<count>, Axis>>;
      auto weights = covariance_weights<dim, Weights>();

      if constexpr(cholesky_form<InputDist>)
      {
        if constexpr (W_c0<dim>() < 0)
        {
          // Discard first weight and first y-deviation column for now, since square root of weight would be negative.
          const auto [y_deviations_head, y_deviations_tail] = split_horizontal<1, count - 1>(y_deviations);
          const auto [weights_head, weights_tail] = split_vertical<Axis, Dimensions<count - 1>>(weights);
          const auto sqrt_weights_tail = apply_coefficientwise([](const auto x){ return values::sqrt(x); }, weights_tail);
          auto out_covariance = make_self_contained(square(LQ_decomposition(y_deviations_tail * to_diagonal(sqrt_weights_tail))));
          static_assert(OpenKalman::covariance<decltype(out_covariance)>);

          // Factor first weight back in, using a rank update:
          rank_update(out_covariance, y_deviations_head, W_c0<dim>());

          if constexpr (return_cross)
          {
            auto cross_covariance = make_self_contained(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
            return std::tuple {std::move(out_covariance), std::move(cross_covariance)};
          }
          else
          {
            return out_covariance;
          }
        }
        else
        {
          const auto sqrt_weights = apply_coefficientwise([](const auto x){ return values::sqrt(x); }, weights);
          auto out_covariance = make_self_contained(square(LQ_decomposition(y_deviations * to_diagonal(sqrt_weights))));
          static_assert(OpenKalman::covariance<decltype(out_covariance)>);

          if constexpr (return_cross)
          {
            auto cross_covariance = make_self_contained(x_deviations * to_diagonal(weights) * adjoint(y_deviations));
            return std::tuple {std::move(out_covariance), std::move(cross_covariance)};
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
        auto out_covariance = make_self_contained(make_covariance(y_deviations * w_yT));

        if constexpr (return_cross)
        {
          auto cross_covariance = make_self_contained(x_deviations * w_yT);
          return std::tuple {std::move(out_covariance), std::move(cross_covariance)};
        }
        else
        {
          return out_covariance;
        }
      }
    }

  };

}

#endif //OPENKALMAN_SCALEDSIGMAPOINTSBASE_HPP
