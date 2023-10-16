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
 * \brief Definitions for the unscented transform.
 */

#ifndef OPENKALMAN_UNSCENTED_HPP
#define OPENKALMAN_UNSCENTED_HPP

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


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
    template<std::size_t dim>
    static constexpr double kappa = 0.0;
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
    template<std::size_t dim>
    static constexpr double kappa = 3. - double(dim);
  };


  using UnscentedSigmaPointsStateEstimation = Unscented<UnscentedParametersStateEstimation>;

  using UnscentedSigmaPointsParameterEstimation = Unscented<UnscentedParametersParameterEstimation>;

  /// Same as UnscentedSigmaPointsStateEstimation.
  using UnscentedSigmaPoints = Unscented<UnscentedParametersStateEstimation>;


  /**
   * \brief Scaled symmetric sigma points
   * \details As implemented in, e.g., E. Wan & R. van der Merwe,
   * "The unscented Kalman filter for nonlinear estimation," in Proc. of IEEE Symposium (AS-SPCC), pp. 153-158.
   * S. Julier. The scaled unscented tests. In Proceedings of the American
   * Control Conference, Evanston, IL, pages 1108–1114, 2002.
   */
#if __cpp_nontype_template_args >= 201911L
  template<typename Parameters = UnscentedParametersStateEstimation> // \@todo add as floating-point parameters: alpha, beta, kappa
#else
  template<typename Parameters = UnscentedParametersStateEstimation>
#endif
  struct Unscented : oin::ScaledSigmaPointsBase<Unscented<Parameters>>
  {

    /**
     * \brief Number of sigma points.
     * \tparam dim Number of dimensions of the input variable.
     * \note Used by base class.
     */
    template<std::size_t dim>
    static constexpr std::size_t sigma_point_count = dim * 2 + 1;


    /**
     * \brief The alpha parameter.
     * \note Used by base class.
     */
    static constexpr auto alpha = Parameters::alpha;


    /**
     * \brief The beta parameter.
     * \note Used by base class.
     */
    static constexpr auto beta = Parameters::beta;


    /**
     * \brief The unscaled W0 parameter.
     * \tparam dim The total number of dimensions of all inputs.
     * \note Used by base class.
     */
    template<std::size_t dim>
    static constexpr auto unscaled_W0()
    {
      return Parameters::template kappa<dim> / (dim + Parameters::template kappa<dim>);
    }


    /**
     * \brief The unscaled W parameter.
     * \tparam dim The total number of dimensions of all inputs.
     * \note Used by base class.
     */
    template<std::size_t dim>
    static constexpr auto unscaled_W()
    {
      return 0.5 / (dim + Parameters::template kappa<dim>);
    }

  private:

    /// Prevent instantiation.
    Unscented() {};


    /*
     * Scale and translate normalized sample points based on mean and (square root) covariance.
     * This algorithm decreases the complexity from O(n^3) to O(n^2).
     * This steps recursively through a tuple of input and noise distributions.
     */
    template<
      std::size_t dim, //< The total number of dimensions for which sigma points are assigned.
      std::size_t pos = 0, //< The writing position during this recursive step.
      typename D, //< First input or noise distribution.
      typename...Ds> //< Other input or noise distributions.
    static auto
    sigma_points_impl(const D& d, const Ds&...ds)
    {
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coeffs = typename DistributionTraits<D>::TypedIndex;
      using M = typename DistributionTraits<D>::Mean;
      constexpr auto points_count = sigma_point_count<dim>;
      constexpr auto dim_i = index_dimension_of_v<D, 0>;
      constexpr auto frame_size = dim_i * 2;
      constexpr Scalar gamma_L = alpha * alpha * (Parameters::template kappa<dim> + dim);
      const auto delta = make_matrix<Coeffs, Dimensions<dim_i>>(make_dense_writable_matrix_from(square_root(gamma_L * covariance_of(d))));

      if constexpr(1 + frame_size == points_count)
      {
        // | 0 | delta | -delta |
        static_assert(sizeof...(ds) == 0);
        using M0 = Matrix<Coeffs, Axis, untyped_dense_writable_matrix_t<M, Layout::none, Scalar, dim_i, 1>>;
        const auto m0 = make_zero_matrix_like<M0>(Dimensions<dim_i>{}, Dimensions<1>{});
        auto ret = make_self_contained(concatenate_horizontal(std::move(m0), delta, -delta));
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple {std::move(ret)};
      }
      else if constexpr (pos == 0)
      {
        // | 0 | delta | -delta | 0 ... |
        using M0 = Matrix<Coeffs, Axis, untyped_dense_writable_matrix_t<M, Layout::none, Scalar, dim_i, 1>>;
        const auto m0 = make_zero_matrix_like<M0>(Dimensions<dim_i>{}, Dimensions<1>{});
        constexpr auto width = points_count - (1 + frame_size);
        using Mright = Matrix<Coeffs, Dimensions<width>, untyped_dense_writable_matrix_t<M, Layout::none, Scalar, dim_i, width>>;
        const auto mright = make_zero_matrix_like<Mright>(Dimensions<dim_i>{}, Dimensions<width>{});
        auto ret = make_self_contained(concatenate_horizontal(std::move(m0), delta, -delta, std::move(mright)));
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<dim, 1 + frame_size>(ds...));
      }
      else if constexpr (pos + frame_size < points_count)
      {
        // | 0 | 0 ... | delta | -delta | 0 ... |
        using Mleft = Matrix<Coeffs, Dimensions<pos>, untyped_dense_writable_matrix_t<M, Layout::none, Scalar, dim_i, pos>>;
        const auto mleft = make_zero_matrix_like<Mleft>(Dimensions<dim_i>{}, Dimensions<pos>{});
        constexpr auto width = points_count - (pos + frame_size);
        using Mright = Matrix<Coeffs, Dimensions<width>, untyped_dense_writable_matrix_t<M, Layout::none, Scalar, dim_i, width>>;
        const auto mright = make_zero_matrix_like<Mright>(Dimensions<dim_i>{}, Dimensions<width>{});
        auto ret = make_self_contained(concatenate_horizontal(std::move(mleft), delta, -delta, std::move(mright)));
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<dim, pos + frame_size>(ds...));
      }
      else
      {
        // | 0 | 0 ... | delta | -delta |
        static_assert(sizeof...(ds) == 0);
        using Mleft = Matrix<Coeffs, Dimensions<pos>, untyped_dense_writable_matrix_t<M, Layout::none, Scalar, dim_i, pos>>;
        const auto mleft = make_zero_matrix_like<Mleft>(Dimensions<dim_i>{}, Dimensions<pos>{});
        auto ret = make_self_contained(concatenate_horizontal(std::move(mleft), delta, -delta));
        static_assert(index_dimension_of_v<decltype(ret), 1> == points_count);
        return std::tuple {std::move(ret)};
      }
    }

  public:

    /**
     * \brief Calculate the scaled and translated sigma points, given a prior distribution and noise terms.
     * \details The mean of the sample points is effectively translated the origin.
     * \tparam Dist The prior distribution and any optional noise distributions.
     * \return A tuple of sigma point matrices, one matrix for each input and noise distribution.
     * Each column of these matrices corresponds to a sigma point.
     */
#ifdef __cpp_concepts
    template<gaussian_distribution ... Dist> requires (sizeof...(Dist) > 0)
#else
    template<typename...Dist, std::enable_if_t<
      (gaussian_distribution<Dist> and ...) and (sizeof...(Dist) > 0), int> = 0>
#endif
    static auto
    sample_points(const Dist& ...ds)
    {
      constexpr auto dim = (index_dimension_of_v<Dist, 0> + ...);
      return sigma_points_impl<dim>(ds...);
    }
  };


}

#endif //OPENKALMAN_UNSCENTED_HPP
