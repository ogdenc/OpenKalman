/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_CUBATUREPOINTS_HPP
#define OPENKALMAN_CUBATUREPOINTS_HPP


namespace OpenKalman
{

  /*************CubaturePoints************
   * \brief Cubature points, as implemented in
   * I. Arasaratnam & S. Haykin, Cubature Kalman Filters, IEEE Transactions on Automatic Control,
   * vol. 54, pp. 1254-1269, 2009.
   * This class has only static members, and is not to be instantiated.
   */
  struct CubaturePoints
  {

  protected:

    CubaturePoints() {} // Disallow instantiation.

  private:

    /// Scale and translate normalized sample points based on mean and (square root) covariance.
    /// The function steps recursively through a tuple of input and noise distributions.
    template<
      std::size_t dim, ///< The total number of dimensions for which sigma points are assigned.
      std::size_t pos = 0, ///< The writing position during this recursive step.
      typename D, ///< First input or noise distribution.
      typename...Ds> ///< Other input or noise distributions.
    static auto
    sample_points_impl(const D& d, const Ds&...ds)
    {
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coeffs = typename DistributionTraits<D>::Coefficients;
      using M = typename DistributionTraits<D>::Mean;
      constexpr auto points_count = dim * 2;
      constexpr auto dim_i = DistributionTraits<D>::dimension;
      constexpr auto frame_size = dim_i * 2;
      constexpr Scalar n = dim;
      const auto delta = make_matrix<Coeffs, Axes<dim_i>>(make_native_matrix(square_root(n * covariance_of(d))));

      if constexpr(frame_size == points_count)
      {
        // | delta | -delta |
        auto ret = concatenate_horizontal(delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple {std::move(ret)};
      }
      else if constexpr (pos == 0)
      {
        // | delta | -delta | 0 ... |
        constexpr auto width = points_count - frame_size;
        const auto mright = Matrix<Coeffs, Axes<width>, native_matrix_t<M, dim_i, width>>::zero();
        auto ret = concatenate_horizontal(delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sample_points_impl<dim, frame_size>(ds...));
      }
      else if constexpr (pos + frame_size < points_count)
      {
        // | 0 ... | delta | -delta | 0 ... |
        const auto mleft = Matrix<Coeffs, Axes<pos>, native_matrix_t<M, dim_i, pos>>::zero();
        constexpr auto width = points_count - (pos + frame_size);
        const auto mright = Matrix<Coeffs, Axes<width>, native_matrix_t<M, dim_i, width>>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sample_points_impl<dim, pos + frame_size>(ds...));
      }
      else
      {
        // | 0 ... | delta | -delta |
        static_assert(sizeof...(ds) == 0);
        const auto mleft = Matrix<Coeffs, Axes<pos>, native_matrix_t<M, dim_i, pos>>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple {std::move(ret)};
      }
    }

  public:

    /**
     * \brief Scale and translate normalized sample points based on mean and (square root) covariance.
     * \return A matrix of sigma points (each sigma point in a column).
     */
#ifdef __cpp_concepts
    template<gaussian_distribution ... Dist>
#else
    template<typename...Dist, std::enable_if_t<((gaussian_distribution<Dist> and ...)), int> = 0>
#endif
    static auto
    sample_points(const Dist&...ds)
    {
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ...);
      return sample_points_impl<dim>(ds...);
    }


#ifdef __cpp_concepts
    template<std::size_t dim, euclidean_mean Arg>
#else
    template<std::size_t dim, typename Arg, std::enable_if_t<euclidean_mean<Arg>, int> = 0>
#endif
    static auto
    weighted_means(const Arg& y_means)
    {
      static_assert(typed_matrix<Arg> and untyped_columns<Arg>);
      static_assert(MatrixTraits<Arg>::columns == dim * 2, "Wrong number of cubature points.");
      return reduce_columns(y_means);
    };


#ifdef __cpp_concepts
    template<std::size_t dim, typename InputDist, bool return_cross = false, typed_matrix X, typed_matrix Y> requires
      (MatrixTraits<X>::columns == MatrixTraits<Y>::columns) and (MatrixTraits<X>::columns == dim * 2) and
      equivalent_to<typename MatrixTraits<X>::RowCoefficients, typename DistributionTraits<InputDist>::Coefficients>
#else
    template<std::size_t dim, typename InputDist, bool return_cross = false, typename X, typename Y, std::enable_if_t<
      typed_matrix<X> and typed_matrix<Y> and (MatrixTraits<X>::columns == MatrixTraits<Y>::columns) and
      (MatrixTraits<X>::columns == dim * 2) and
      equivalent_to<typename MatrixTraits<X>::RowCoefficients, typename DistributionTraits<InputDist>::Coefficients>,
        int> = 0>
#endif
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      constexpr auto count = MatrixTraits<X>::columns;
      constexpr auto inv_weight = 1 / static_cast<typename MatrixTraits<X>::Scalar>(count);

      if constexpr(cholesky_form<InputDist>)
      {
        auto out_covariance = square(LQ_decomposition(y_deviations * std::sqrt(inv_weight)));
        static_assert(OpenKalman::covariance<decltype(out_covariance)>);

        if constexpr (return_cross)
        {
          auto cross_covariance = make_self_contained(x_deviations * inv_weight * adjoint(y_deviations));
          return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
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

#endif //OPENKALMAN_CUBATUREPOINTS_HPP
