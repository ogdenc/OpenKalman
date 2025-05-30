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
 * \file
 * \brief Definition of CubaturePoints.
 */

#ifndef OPENKALMAN_CUBATUREPOINTS_HPP
#define OPENKALMAN_CUBATUREPOINTS_HPP


namespace OpenKalman
{

  /**
   * \brief Implementation of a cubature points transform.
   * \details This is as implemented in
   * I. Arasaratnam & S. Haykin, Cubature Kalman Filters, IEEE Transactions on Automatic Control,
   * vol. 54, pp. 1254-1269, 2009.
   * \note This class has only static members, and is not to be instantiated.
   */
  struct CubaturePoints
  {

  private:

    /// Instantiation is disallowed.
    CubaturePoints() {}


    /*
     * \brief Calculate the scaled sample points.
     * \details The function steps recursively through a tuple of input and noise distributions.
     * \tparam dim The total number of dimensions of all inputs.
     * \tparam pos The writing position during this recursive step.
     * \tparam D First input or noise distribution.
     * \tparam Ds Other input or noise distributions.
     * \return A tuple of sample point matrices, one matrix for each input and noise distribution.
     * Each column of these matrices corresponds to a sample point.
     */
    template<std::size_t dim, std::size_t pos = 0, typename D, typename...Ds>
    static auto
    sample_points_impl(const D& d, const Ds&...ds)
    {
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coeffs = typename DistributionTraits<D>::StaticDescriptor;
      using M = typename DistributionTraits<D>::Mean;
      constexpr auto points_count = dim * 2;
      constexpr auto dim_i = index_dimension_of_v<D, 0>;
      constexpr auto frame_size = dim_i * 2;
      constexpr Scalar n = dim;
      const auto delta = make_vector_space_adapter(to_dense_object(square_root(n * covariance_of(d))), Coeffs{}, Dimensions<dim_i>{});

      if constexpr(frame_size == points_count)
      {
        // | delta | -delta |
        static_assert(sizeof...(ds) == 0);
        auto ret {concatenate_horizontal(delta, -delta)};
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple {std::move(ret)};
      }
      else if constexpr (pos == 0)
      {
        // | delta | -delta | 0 ... |
        constexpr auto width = points_count - frame_size;
        using Mright = dense_writable_matrix_t<M, Layout::none, Scalar, std::tuple<Coeffs, Dimensions<width>>>;
        const auto mright = make_zero<Mright>(Dimensions<dim_i>{}, Dimensions<width>{});
        auto ret {concatenate_horizontal(delta, -delta, std::move(mright))};
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)},
          sample_points_impl<dim, frame_size>(ds...));
      }
      else if constexpr (pos + frame_size < points_count)
      {
        // | 0 ... | delta | -delta | 0 ... |
        using Mleft = dense_writable_matrix_t<M, Layout::none, Scalar, std::tuple<Coeffs, Dimensions<pos>>>;
        const auto mleft = make_zero<Mleft>(Dimensions<dim_i>{}, Dimensions<pos>{});
        constexpr auto width = points_count - (pos + frame_size);
        using Mright = dense_writable_matrix_t<M, Layout::none, Scalar, std::tuple<Coeffs, Dimensions<width>>>;
        const auto mright = make_zero<Mright>(Dimensions<dim_i>{}, Dimensions<width>{});
        auto ret {concatenate_horizontal(std::move(mleft), delta, -delta, std::move(mright))};
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)},
          sample_points_impl<dim, pos + frame_size>(ds...));
      }
      else
      {
        // | 0 ... | delta | -delta |
        static_assert(sizeof...(ds) == 0);
        using Mleft = dense_writable_matrix_t<M, Layout::none, Scalar, std::tuple<Coeffs, Dimensions<pos>>>;
        const auto mleft = make_zero<Mleft>(Dimensions<dim_i>{}, Dimensions<pos>{});
        auto ret {concatenate_horizontal(std::move(mleft), delta, -delta)};
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple {std::move(ret)};
      }
    }

  public:

    /**
     * \brief Calculate the scaled sample points, given a prior distribution and noise terms.
     * \details The mean of the sample points is effectively translated the origin.
     * \tparam Dist The prior distribution and any optional noise distributions.
     * \return A tuple of sample point matrices, one matrix for each input and noise distribution.
     * Each column of these matrices corresponds to a sample point.
     */
#ifdef __cpp_concepts
    template<gaussian_distribution ... Dist> requires (sizeof...(Dist) > 0)
#else
    template<typename...Dist, std::enable_if_t<
      (gaussian_distribution<Dist> and ...) and (sizeof...(Dist) > 0), int> = 0>
#endif
    static auto
    sample_points(const Dist&...ds)
    {
      constexpr auto dim = (index_dimension_of_v<Dist, 0> + ...);
      return sample_points_impl<dim>(ds...);
    }


    /**
     * \brief Calculate the weighted average of posterior means for each sample point.
     * \tparam dim The total number of dimensions of all inputs.
     * \tparam Arg A matrix in which each column corresponds to a mean for each sample point.
     * \param y_means
     * \return
     */
#ifdef __cpp_concepts
    template<std::size_t dim, typed_matrix YMeans> requires has_untyped_index<YMeans, 1> and
      (index_dimension_of_v<YMeans, 0> == coordinates::stat_dimension_of_v<vector_space_descriptor_of_t<YMeans, 0>>) and
      (index_dimension_of_v<YMeans, 1> == dim * 2)
#else
    template<std::size_t dim, typename YMeans, std::enable_if_t<typed_matrix<YMeans> and has_untyped_index<YMeans, 1> and
      (index_dimension_of<YMeans, 0>::value == coordinates::stat_dimension_of_v<vector_space_descriptor_of_t<YMeans, 0>>) and
      (index_dimension_of_v<YMeans, 1> == dim * 2), int> = 0>
#endif
    static auto
    weighted_means(YMeans&& y_means)
    {
      return make_self_contained(average_reduce<1>(std::forward<YMeans>(y_means)));
    };


#ifdef __cpp_concepts
    /**
     * \brief Calculate the posterior covariance, given prior and posterior deviations from the sample points
     * \tparam dim The total number of dimensions of all inputs.
     * \tparam InputDist The prior distribution.
     * \tparam return_cross Whether to return a cross-covariance.
     * \tparam X The scaled sample points for the prior distribution (the mean is translated to origin).
     * \tparam Y The transformed sample points for the posterior distribution (the mean is translated to origin).
     * \return The posterior covariance, or (if return_cross, then a tuple comprising the posterior covariance
     * and the cross-covariance.
     */
    template<std::size_t dim, typename InputDist, bool return_cross = false, typed_matrix X, typed_matrix Y> requires
      (index_dimension_of_v<X, 1> == index_dimension_of_v<Y, 1>) and (index_dimension_of_v<X, 1> == dim * 2) and
      compares_with<vector_space_descriptor_of_t<X, 0>, typename DistributionTraits<InputDist>::StaticDescriptor>
#else
    template<std::size_t dim, typename InputDist, bool return_cross = false, typename X, typename Y, std::enable_if_t<
      typed_matrix<X> and typed_matrix<Y> and (index_dimension_of<X, 1>::value == index_dimension_of<Y, 1>::value) and
      (index_dimension_of<X, 1>::value == dim * 2) and
      compares_with<vector_space_descriptor_of_t<X, 0>, typename DistributionTraits<InputDist>::StaticDescriptor>,
        int> = 0>
#endif
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      constexpr auto count = index_dimension_of_v<X, 1>;
      constexpr auto inv_weight = 1 / static_cast<scalar_type_of_t<X>>(count);

      if constexpr(cholesky_form<InputDist>)
      {
        auto out_covariance = make_self_contained(square(LQ_decomposition(y_deviations * values::sqrt(inv_weight))));
        static_assert(OpenKalman::covariance<decltype(out_covariance)>);

        if constexpr (return_cross)
        {
          auto cross_covariance = make_self_contained(x_deviations * inv_weight * adjoint(y_deviations));
          return std::tuple {std::move(out_covariance), std::move(cross_covariance)};
        }
        else
        {
          return out_covariance;
        }
      }
      else
      {
        const auto w_yT = make_self_contained(inv_weight * adjoint(y_deviations));
        auto out_covariance = Covariance {make_self_contained(y_deviations * w_yT)};

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

#endif //OPENKALMAN_CUBATUREPOINTS_HPP
