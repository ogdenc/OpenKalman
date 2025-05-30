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
 * \brief Definitions for the spherical simplex unscented transform.
 */

#ifndef OPENKALMAN_SPHERICALSIMPLEX_HPP
#define OPENKALMAN_SPHERICALSIMPLEX_HPP

#include <limits>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  template<typename Parameters>
  struct SphericalSimplex;

  struct SphericalSimplexParameters
  {
    /// Scaling factor (typically 0.001 but may be from 0.0001 to 1).
    static constexpr double alpha = 0.001;
    /// Factor to compensate for the distribution (beta==2 is optimal for Gaussian distributions).
    static constexpr double beta = 2.0;
    /// The first weight, 0 <= W0 <= 1. It is a free parameter that affects the fourth and higher moments of the sigma point set.
    /// Here, for simplicity, it is equal to the other weights.
    template<std::size_t dim>
    static constexpr double W0 = 1. / (double(dim) + 2.);
  };


  using SphericalSimplexSigmaPoints = SphericalSimplex<SphericalSimplexParameters>;


  /**
   * \brief Spherical simplex sigma points, as implemented in, e.g.,
   * Simon J. Julier. The spherical simplex unscented tests.
   * In Proceedings of American Control Conference, Denver, Colorado, pages 2430–2434, 2003.
   */
  template<typename Parameters = SphericalSimplexParameters>
  struct SphericalSimplex : oin::ScaledSigmaPointsBase<SphericalSimplex<Parameters>>
  {
    /**
     * \brief Number of sigma points.
     * \tparam dim Number of dimensions of the input variable.
     * \note Used by base class.
     */
    template<std::size_t dim>
    static constexpr size_t sigma_point_count = dim + 2;


    /**
     * \brief The alpha parameter
     * \note Used by base class.
     */
    static constexpr auto alpha = Parameters::alpha;


    /**
     * \brief The beta parameter
     * \note Used by base class.
     */
    static constexpr auto beta = Parameters::beta;


    /*
     * \brief The unscaled W0 parameter.
     * \tparam dim The total number of dimensions of all inputs.
     * \note Used by base class.
     */
    template<std::size_t dim>
    static constexpr auto unscaled_W0()
    {
      return Parameters::template W0<dim>;
    }


    /*
     * \brief The unscaled W parameter.
     * \tparam dim The total number of dimensions of all inputs.
     * \note Used by base class.
     */
    template<std::size_t dim>
    static constexpr auto unscaled_W()
    {
      return (1 - Parameters::template W0<dim>) / (sigma_point_count<dim> - 1);
    }

  private:

    /// Prevent instantiation.
    SphericalSimplex() {};


    template<std::size_t j, std::size_t i, std::size_t dim, typename Scalar>
    static constexpr auto
    sigma_point_coeff()
    {
      constexpr auto denom = 1 / static_cast<std::size_t>(values::sqrt((j + 1)) * (j + 2) * unscaled_W<dim>());
      if constexpr(i == 0)
        return Scalar(0);
      else if constexpr(i < j + 2)
        return -denom;
      else if constexpr (i == j + 2)
        return (j + 1) * denom;
      else
        return Scalar(0);
    }


    template<typename Dist, std::size_t dim, std::size_t pos, std::size_t...ns>
    static constexpr auto
    unscaled_sigma_points(std::index_sequence<ns...>)
    {
      using StaticDescriptor = typename DistributionTraits<Dist>::StaticDescriptor;
      using Scalar = typename DistributionTraits<Dist>::Scalar;
      constexpr auto rows = index_dimension_of_v<Dist, 0>;
      constexpr auto count = sigma_point_count<dim>;
      using M = typename DistributionTraits<Dist>::Mean;
      using Xnative = dense_writable_matrix_t<M, Layout::none, Scalar, std::tuple<Dimensions<rows>, Dimensions<count>>>;
      Matrix<StaticDescriptor, Dimensions<count>, Xnative> X {sigma_point_coeff<ns / count + pos, ns % count, dim, Scalar>()...};
      return X;
    }


    template<std::size_t dim, std::size_t pos = 0, typename D, typename...Ds>
    static auto
    sigma_points_impl(const D& d, const Ds&...ds)
    {
      constexpr auto rows = index_dimension_of_v<D, 0>;
      constexpr auto count = sigma_point_count<dim>;
      auto X = unscaled_sigma_points<D, dim, pos>(std::make_index_sequence<count * rows> {});
      // Scale based on covariance:
      auto ret {make_self_contained(square_root(covariance_of(d)) * alpha * X)};
      //
      if constexpr(sizeof...(ds) > 0)
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<dim, pos + rows>(ds...));
      else
        return std::tuple {std::move(ret)};
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
    static constexpr auto
    sample_points(const Dist&...ds)
    {
      constexpr auto dim = (index_dimension_of_v<Dist, 0> + ...);
      return sigma_points_impl<dim>(ds...);
    }

  };


}

#endif //OPENKALMAN_SPHERICALSIMPLEX_HPP
