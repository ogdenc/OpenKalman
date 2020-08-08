/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_CUBATUREPOINTS_H
#define OPENKALMAN_CUBATUREPOINTS_H


namespace OpenKalman
{

  /*************CubaturePoints************
   * @brief Cubature points, as implemented in
   * I. Arasaratnam & S. Haykin, Cubature Kalman Filters, IEEE Transactions on Automatic Control,
   * vol. 54, pp. 1254-1269, 2009.
   * This class has only static members, and is not to be instantiated.
   */
  struct CubaturePoints
  {
  protected:
    CubaturePoints() {} // Disallow instantiation.

  private:
    template<std::size_t i, std::size_t pos, std::size_t dim, typename DTuple>
    static auto
    sample_points_impl(const DTuple& dtuple)
    {
      using D = std::tuple_element_t<i, DTuple>;
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coeffs = typename DistributionTraits<D>::Coefficients;
      constexpr auto dim_i = MatrixTraits<typename DistributionTraits<D>::Mean>::dimension;
      constexpr auto count = dim * 2;
      constexpr auto size = DistributionTraits<D>::dimension * 2;
      constexpr auto dsize = std::tuple_size_v<DTuple>;
      const auto d = std::get<i>(dtuple);
      const auto delta = make_Matrix<Coeffs, Axes<dim_i>>(strict_matrix(square_root(OpenKalman::covariance(d) * static_cast<Scalar>(dim))));
      if constexpr(dsize == 1)
      {
        auto ret = concatenate_horizontal(delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple {std::move(ret)};
      }
      else if constexpr (i == 0)
      {
        constexpr auto width = count - (pos + size);
        using MRbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, width>;
        const auto mright = TypedMatrix<Coeffs, Axes<width>, MRbase>::zero();
        auto ret = concatenate_horizontal(delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sample_points_impl<i + 1, pos + size, dim>(dtuple));
      }
      else if constexpr (i < dsize - 1)
      {
        using MLbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, pos>;
        const auto mleft = TypedMatrix<Coeffs, Axes<pos>, MLbase>::zero();
        constexpr auto width = count - (pos + size);
        using MRbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, width>;
        const auto mright = TypedMatrix<Coeffs, Axes<width>, MRbase>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sample_points_impl<i + 1, pos + size, dim>(dtuple));
      }
      else
      {
        using MLbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, pos>;
        const auto mleft = TypedMatrix<Coeffs, Axes<pos>, MLbase>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple {std::move(ret)};
      }
    }

  public:
    /**
     * @brief Scale and translate normalized sample points based on mean and (square root) covariance.
     * @return A matrix of sigma points (each sigma point in a column).
     */
    template<typename...Dist, std::enable_if_t<std::conjunction_v<is_Gaussian_distribution<Dist>...>, int> = 0>
    static auto
    sample_points(const Dist&...ds)
    {
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ...);
      return sample_points_impl<0, 0, dim>(std::tuple {ds...});
    }

    template<size_t dim, typename Arg, std::enable_if_t<is_Euclidean_mean_v<Arg>, int> = 0>
    static auto
    weighted_means(const Arg& y_means)
    {
      static_assert(is_column_vector_v<Arg>);
      static_assert(MatrixTraits<Arg>::columns == dim * 2, "Wrong number of cubature points.");
      return from_Euclidean(reduce_columns(y_means)); ///@TODO: does this need to be strict?
    };

    template<typename InputDist, typename ... NoiseDist, typename X, typename Y>
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      static_assert(is_typed_matrix_v<X> and is_typed_matrix_v<Y>);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + ... + DistributionTraits<NoiseDist>::dimension);
      using Scalar = typename MatrixTraits<X>::Scalar;
      constexpr auto count = MatrixTraits<X>::columns;
      static_assert(count == MatrixTraits<Y>::columns);
      static_assert(count == dim * 2, "Wrong number of cubature points.");
      constexpr auto inv_weight = 1 / Scalar(count);
      if constexpr(is_Cholesky_v<InputDist>)
      {
        auto out_covariance = Covariance {LQ_decomposition(y_deviations * std::sqrt(inv_weight))};
        auto cross_covariance = strict(x_deviations * inv_weight * adjoint(y_deviations));
        return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
      }
      else
      {
        const auto w_yT = strict(inv_weight * adjoint(y_deviations));
        auto out_covariance = Covariance {strict(y_deviations * w_yT)};
        auto cross_covariance = strict(x_deviations * w_yT);
        return std::tuple{std::move(out_covariance), std::move(cross_covariance)};
      }
    }

  };

}

#endif //OPENKALMAN_CUBATUREPOINTS_H
