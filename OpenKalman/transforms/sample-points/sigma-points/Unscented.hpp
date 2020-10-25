/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_UNSCENTED_HPP
#define OPENKALMAN_UNSCENTED_HPP

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
    /// Scale and translate normalized sample points based on mean and (square root) covariance.
    /// This algorithm decreases the complexity from O(n^3) to O(n^2).
    /// This steps recursively through a tuple of input and noise distributions.
    template<
      std::size_t dim, ///< The total number of dimensions for which sigma points are assigned.
      std::size_t pos = 0, ///< The writing position during this recursive step.
      typename D, ///< First input or noise distribution.
      typename...Ds> ///< Other input or noise distributions.
    static auto
    sigma_points_impl(const D& d, const Ds&...ds)
    {
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coeffs = typename DistributionTraits<D>::Coefficients;
      using M = typename DistributionTraits<D>::Mean;
      constexpr auto points_count = sigma_point_count<dim>();
      constexpr auto dim_i = DistributionTraits<D>::dimension;
      constexpr auto frame_size = dim_i * 2;
      constexpr Scalar alpha = Parameters::alpha;
      constexpr Scalar kappa = Parameters::template kappa<dim>;
      constexpr Scalar gamma_L = alpha * alpha * (kappa + dim);
      const auto delta = make_Matrix<Coeffs, Axes<dim_i>>(strict_matrix(square_root(gamma_L * covariance(d))));
      //
      using M0base = strict_matrix_t<M, dim_i, 1>;
      const auto m0 = Matrix<Coeffs, Axis, M0base>::zero();
      if constexpr(1 + frame_size == points_count)
      {
        auto ret = concatenate_horizontal(m0, delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple {std::move(ret)};
      }
      else if constexpr (pos == 0)
      {
        constexpr auto width = points_count - (1 + frame_size);
        using MRbase = strict_matrix_t<M, dim_i, width>;
        const auto mright = Matrix<Coeffs, Axes<width>, MRbase>::zero();
        auto ret = concatenate_horizontal(m0, delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<dim, 1 + frame_size>(ds...));
      }
      else if constexpr (pos + frame_size < points_count)
      {
        using MLbase = strict_matrix_t<M, dim_i, pos>;
        const auto mleft = Matrix<Coeffs, Axes<pos>, MLbase>::zero();
        constexpr auto width = points_count - (pos + frame_size);
        using MRbase = strict_matrix_t<M, dim_i, width>;
        const auto mright = Matrix<Coeffs, Axes<width>, MRbase>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta, mright);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<dim, pos + frame_size>(ds...));
      }
      else
      {
        static_assert(sizeof...(ds) == 0);
        using MLbase = strict_matrix_t<M, dim_i, pos>;
        const auto mleft = Matrix<Coeffs, Axes<pos>, MLbase>::zero();
        auto ret = concatenate_horizontal(mleft, delta, -delta);
        static_assert(MatrixTraits<decltype(ret)>::columns == points_count);
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
    template<typename...Dist>
    static constexpr auto
    sigma_points(const Dist& ...ds)
    {
      static_assert(std::conjunction_v<is_Gaussian_distribution<Dist>...>);
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ...);
      return sigma_points_impl<dim>(ds...);
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

#endif //OPENKALMAN_UNSCENTED_HPP
