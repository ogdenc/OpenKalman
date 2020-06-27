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

#include "transforms/support/ScaledSigmaPointsBase.h"

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
    /// The first weight, 0 <= W0 <= 1. It is a parameter that affects the fourth and higher moments of the sigma point set
    static constexpr double W0 = 0.5;
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
    template<std::size_t row_start, std::size_t dim, typename D, typename...Ds>
    static auto
    sigma_points_impl(const D& d, const Ds&...ds)
    {
      using Scalar = typename DistributionTraits<D>::Scalar;
      using Coefficients = typename DistributionTraits<D>::Coefficients;
      constexpr auto row_size = DistributionTraits<D>::dimension;
      constexpr auto count = sigma_point_count<dim>();
      // Unscaled sigma points:
      using Xbase = typename MatrixTraits<typename DistributionTraits<D>::Mean>::template StrictMatrix<row_size, count>;
      TypedMatrix<Coefficients, Axes<count>, Xbase> X;
      for (std::size_t j = row_start; j < row_start + row_size; j++)
      {
        const auto row = j - row_start;
        const auto denom = 1 / std::sqrt((j + 1) * (j + 2) * unscaled_W<dim, Scalar>());
        X(row, 0) = 0;
        for (std::size_t i = 1; i < j + 2; i++)
        {
          X(row, i) = -denom;
        }
        X(row, j + 2) = (j + 1) * denom;
        for (std::size_t i = j + 3; i < count; i++)
        {
          X(row, i) = 0;
        }
      }
      // Scaling:
      auto ret = strict(square_root(covariance(d)) * Parameters::alpha * X);
      if constexpr(sizeof...(ds) > 0)
        return std::tuple_cat(std::tuple {std::move(ret)}, sigma_points_impl<row_start + row_size, dim>(ds...));
      else
        return std::tuple {std::move(ret)};
    }

  public:
    template<typename...Dist, std::enable_if_t<std::conjunction_v<is_Gaussian_distribution<Dist>...>, int> = 0>
    static constexpr auto
    sigma_points(const Dist&...ds)
    {
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ...);
      return sigma_points_impl<0, dim>(ds...);
    }

  protected:
    SphericalSimplex() {}; // Prevent instantiation.

    friend struct internal::ScaledSigmaPointsBase<SphericalSimplex<Parameters>, Parameters>;

    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    unscaled_W0()
    {
      return Parameters::W0;
    }

    template<std::size_t dim, typename Scalar = double>
    static constexpr Scalar
    unscaled_W()
    {
      return (1 - unscaled_W0<dim, Scalar>()) / (dim + 1);
    }

  };


}

#endif //OPENKALMAN_SPHERICALSIMPLEX_H
