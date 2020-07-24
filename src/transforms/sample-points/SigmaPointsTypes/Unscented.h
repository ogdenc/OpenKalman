/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_UNSCENTED_H
#define OPENKALMAN_UNSCENTED_H

#include "transforms/support/ScaledSigmaPointsBase.h"

namespace OpenKalman
{
  /*************SymmetricSigmaPoints************
   * @brief Scaled symmetric sigma points, as implemented in, e.g.,
   * E. Wan & R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
   * in Proc. of IEEE Symposium (AS-SPCC), pp. 153-158.
   * S. Julier. The scaled unscented transformation. In Proceedings of the American
   * Control Conference, Evanston, IL, pages 1108–1114, 2002.
   */
  template<typename Parameters>
  struct Unscented;

  /// Unscented parameters for use in state estimation (the default).
  struct UnscentedParametersStateEstimation
  {
    /// Scaling factor (typically 0.001 but may be from 0.0001 to 1).
    static constexpr double alpha = 0.001;
    /// Factor to compensate for the distribution (beta==2 is optimal for Gaussian distributions).
    static constexpr double beta = 2.0;
    /// Secondary scaling parameter. Usually 0 for state estimation and 3-dim for parameter estimation.
    /// This makes it possible to match some of the fourth-order terms when the distribution is Gaussian.
    template<int dim> static constexpr double kappa = 0.0;
  };

  /// Unscented parameters for use in parameter estimation.
  struct UnscentedParametersParameterEstimation
  {
    /// Scaling factor (typically 0.001 but may be from 0.0001 to 1).
    static constexpr double alpha = 0.001;
    /// Factor to compensate for the distribution (beta==2 is optimal for Gaussian distributions).
    static constexpr double beta = 2.0;
    /// Secondary scaling parameter. Usually 0 for state estimation and kurtosis-dim for parameter estimation.
    /// This makes it possible to match some of the fourth-order terms. (For Gaussian, kurtosis = 3).
    template<int dim> static constexpr double kappa = 3 - dim;
  };


  template<typename Parameters = UnscentedParametersStateEstimation>
  struct Unscented : internal::ScaledSigmaPointsBase<Unscented<Parameters>, Parameters>
  {
    /**
     * Number of sigma points.
     * @tparam dim Number of dimensions of the input variable.
     * @return Number of sigma points.
     */
    template<std::size_t dim>
    static constexpr std::size_t
    sigma_point_count() { return dim * 2 + 1; };

  private:
    // Scale and translate normalized sample points based on mean and (square root) covariance.
    // This algorithm decreases the complexity from O(n^3) to O(n^2).
    template<std::size_t i, std::size_t pos, std::size_t dim, typename DTuple>
    static auto
    sigma_points_impl(const DTuple& dtuple)
    {
      using D = std::tuple_element_t<i, DTuple>;
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coeffs = typename DistributionTraits<D>::Coefficients;
      const auto d = std::get<i>(dtuple);
      constexpr auto dim_i = MatrixTraits<typename DistributionTraits<D>::Mean>::dimension;
      constexpr auto count = sigma_point_count<dim>();
      constexpr auto size = sigma_point_count<DistributionTraits<D>::dimension>();
      constexpr auto dsize = std::tuple_size_v<DTuple>;
      constexpr Scalar alpha = Parameters::alpha;
      constexpr Scalar kappa = Parameters::template kappa<dim>;
      constexpr Scalar gamma_L = alpha * alpha * (kappa + dim);
      const auto delta = make_Matrix<Coeffs, Axes<dim_i>>(strict_matrix(square_root(gamma_L * covariance(d))));
      //
      using M0base = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, 1>;
      const auto m0 = TypedMatrix<Coeffs, Axis, M0base>::zero();
      if constexpr(dsize == 1)
      {
        auto ret = concatenate_horizontal(m0, delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple {std::move(ret)};
      }
      else if constexpr (i == 0)
      {
        constexpr auto width = count - (pos + size);
        using MRbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, width>;
        const auto mright = TypedMatrix<Coeffs, Axes<width>, MRbase>::zero();
        auto ret = concatenate_horizontal(m0, delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<i + 1, pos + size, dim>(dtuple));
      }
      else if constexpr (i < dsize - 1)
      {
        using MLbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, pos>;
        const auto mleft = TypedMatrix<Coeffs, Axes<pos>, MLbase>::zero();
        constexpr auto width = count - (pos + 2*dim_i);
        using MRbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<dim_i, width>;
        const auto mright = TypedMatrix<Coeffs, Axes<width>, MRbase>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<i + 1, pos + 2*dim_i, dim>(dtuple));
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
     * This algorithm decreases the complexity from O(n^3) to O(n^2).
     * @param x The input distribution.
     * @return A matrix of sigma points (each sigma point in a column).
     */
    template<typename...Dist, std::enable_if_t<std::conjunction_v<is_Gaussian_distribution<Dist>...>, int> = 0>
    static constexpr auto
    sigma_points(const Dist& ...ds)
    {
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ...);
      return sigma_points_impl<0, 0, dim>(std::tuple {ds...});
    }

  protected:
    Unscented() {}; // Prevent instantiation.

    friend struct internal::ScaledSigmaPointsBase<Unscented<Parameters>, Parameters>;

    template<std::size_t dim, typename Scalar = double>
    static constexpr auto
    unscaled_W0()
    {
      constexpr Scalar kappa = Parameters::template kappa<dim>;
      return kappa / (dim + kappa);
    }

    template<std::size_t dim, typename Scalar = double>
    static constexpr auto
    unscaled_W()
    {
      constexpr Scalar kappa = Parameters::template kappa<dim>;
      return 0.5 / (dim + kappa);
    }

  };


}

#endif //OPENKALMAN_UNSCENTED_H
