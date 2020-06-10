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

#include "distributions/DistributionTraits.h"
#include "variables/classes/tests-mean.h"

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
      constexpr auto count = dim * 2;
      constexpr auto size = DistributionTraits<D>::dimension * 2;
      constexpr auto dsize = std::tuple_size_v<DTuple>;
      const auto d = std::get<i>(dtuple);
      const auto delta = make_Mean<Coeffs>(strict_matrix(OpenKalman::square_root(OpenKalman::covariance(d) * static_cast<Scalar>(dim))));
      const auto delta_p = apply_columnwise(delta, [&d](const auto& col) { return col + mean(d); });
      const auto delta_m = apply_columnwise(delta, [&d](const auto& col) { return col - mean(d); });
      if constexpr(dsize == 1)
      {
        return std::tuple {concatenate_horizontal(delta_p, delta_m)};
      }
      else if constexpr (i == 0)
      {
        const auto mright = apply_columnwise<count - (pos + size)>([&d] { return mean(d); });
        const auto ret = concatenate_horizontal(delta_p, delta_m, mright);
        return std::tuple_cat(std::tuple {ret}, sample_points_impl<i + 1, pos + size, dim>(dtuple));
      }
      else if constexpr (i < dsize - 1)
      {
        const auto mleft = apply_columnwise<pos>([&d] { return mean(d); });
        const auto mright = apply_columnwise<count - (pos + size)>([&d] { return mean(d); });
        const auto ret = concatenate_horizontal(mleft, delta_p, delta_m, mright);
        return std::tuple_cat(std::tuple {ret}, sample_points_impl<i + 1, pos + size, dim>(dtuple));
      }
      else
      {
        const auto mleft = apply_columnwise<pos>([&d] { return mean(d); });
        return std::tuple {concatenate_horizontal(mleft, delta_p, delta_m)};
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

    template<size_t dim, typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    static auto
    weighted_means(const Arg& y_means)
    {
      static_assert(is_column_vector_v<Arg>);
      static_assert(not is_Euclidean_transformed_v<Arg>);
      static_assert(MatrixTraits<Arg>::columns == dim * 2, "Wrong number of cubature points.");
      return reduce_columns(y_means); ///@TODO: does this need to be strict?
    };

    template<std::size_t dim, typename X, typename Y,
      std::enable_if_t<is_typed_matrix_v<X>, int> = 0, std::enable_if_t<is_typed_matrix_v<Y>, int> = 0>
    static auto
    covariance(const X& x_deviations, const Y& y_deviations)
    {
      using Scalar = typename MatrixTraits<X>::Scalar;
      constexpr auto count = MatrixTraits<X>::columns;
      static_assert(count == MatrixTraits<Y>::columns);
      static_assert(count == dim * 2, "Wrong number of cubature points.");
      const Scalar inv_weight = count;
      const auto w_yT = adjoint(y_deviations) / inv_weight;
      const auto cross_covariance = strict(x_deviations * w_yT);
      const auto out_covariance = strict(y_deviations * w_yT);
      return std::tuple{out_covariance, cross_covariance};
    }

    template<std::size_t dim, typename X, typename Y,
      std::enable_if_t<is_typed_matrix_v<X>, int>, std::enable_if_t<is_typed_matrix_v<Y>, int> = 0>
    static auto
    sqrt_covariance(const X& x_deviations, const Y& y_deviations)
    {
      using Scalar = typename MatrixTraits<X>::Scalar;
      constexpr auto count = MatrixTraits<X>::count;
      static_assert(count == dim * 2, "Wrong number of cubature points.");
      const auto inv_sqrt_weight = std::sqrt(Scalar(count));
      const auto y_sqw = y_deviations / inv_sqrt_weight;
      const auto out_covariance = LQ_decomposition(y_sqw); // This covariance is in Cholesky form.
      const auto cross_covariance = strict(x_deviations / inv_sqrt_weight * adjoint(y_sqw));
      return std::tuple{out_covariance, cross_covariance};
    }

  };

}

#endif //OPENKALMAN_CUBATUREPOINTS_H
