/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SPHERICALSIMPLEX_H
#define OPENKALMAN_SPHERICALSIMPLEX_H

#include <limits>

namespace OpenKalman
{
  /*************SphericalSimplexSigmaPoints************
   * @brief Spherical simplex sigma points, as implemented in, e.g.,
   * Simon J. Julier. The spherical simplex unscented transformation.
   * In Proceedings of American Control Conference, Denver, Colorado, pages 2430–2434, 2003.
   */
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
    template<int dim> static constexpr double W0 = 1 / (dim + 2);
  };


  template<typename Parameters = SphericalSimplexParameters>
  struct SphericalSimplex : internal::ScaledSigmaPointsBase<SphericalSimplex<Parameters>, Parameters>
  {
    /**
     * Number of sigma points.
     * @tparam dim Number of dimensions of the input variable.
     * @return Number of sigma points.
     */
    template<std::size_t dim>
    static constexpr size_t
    sigma_point_count() { return dim + 2; };

  private:
    /// Compile time sqrt.
    template<typename Scalar>
    static constexpr Scalar constexpr_sqrt(Scalar x, Scalar guess)
    {
      return (0.25 * (guess + x / guess) * (guess + x / guess)/x - 1 <= std::numeric_limits<Scalar>::epsilon()) ?
        (0.5 * (guess + x / guess)) :
        constexpr_sqrt(x, 0.5 * (guess + x / guess));
    }

    template<typename Scalar>
    static constexpr Scalar constexpr_sqrt(Scalar x) { return constexpr_sqrt(x, 1.); }

    template<std::size_t j, std::size_t i, std::size_t dim, typename Scalar>
    static constexpr auto
    sigma_point_coeff()
    {
      constexpr auto denom = 1 / constexpr_sqrt((j + 1) * (j + 2) * unscaled_W<dim, Scalar>());
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
      using Coefficients = typename DistributionTraits<Dist>::Coefficients;
      using Scalar = typename DistributionTraits<Dist>::Scalar;
      constexpr auto rows = DistributionTraits<Dist>::dimension;
      constexpr auto count = sigma_point_count<dim>();
      using Xbase = strict_matrix_t<typename DistributionTraits<Dist>::Mean, rows, count>;
      TypedMatrix<Coefficients, Axes<count>, Xbase> X {sigma_point_coeff<ns / count + pos, ns % count, dim, Scalar>()...};
      return X;
    }

    template<std::size_t dim, std::size_t pos = 0, typename D, typename...Ds>
    static auto
    sigma_points_impl(const D& d, const Ds&...ds)
    {
      constexpr auto rows = DistributionTraits<D>::dimension;
      constexpr auto count = sigma_point_count<dim>();
      auto X = unscaled_sigma_points<D, dim, pos>(std::make_index_sequence<count * rows>());
      // Scale based on covariance:
      auto ret = strict(square_root(covariance(d)) * Parameters::alpha * X);
      //
      if constexpr(sizeof...(ds) > 0)
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<dim, pos + rows>(ds...));
      else
        return std::tuple {std::move(ret)};
    }

  public:
    template<typename...Dist>
    static constexpr auto
    sigma_points(const Dist&...ds)
    {
      static_assert(std::conjunction_v<is_Gaussian_distribution<Dist>...>);
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ...);
      return sigma_points_impl<dim>(ds...);
    }

  protected:
    SphericalSimplex() {}; // Prevent instantiation.

    friend struct internal::ScaledSigmaPointsBase<SphericalSimplex<Parameters>, Parameters>;

    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    unscaled_W0()
    {
      return Parameters::template W0<dim>;
    }

    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    unscaled_W()
    {
      return (1 - unscaled_W0<dim, Scalar>()) / (sigma_point_count<dim>() - 1);
    }

  };


}

#endif //OPENKALMAN_SPHERICALSIMPLEX_H
