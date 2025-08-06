/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::Product.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_PRODUCT_HPP
#define OPENKALMAN_EIGEN_TRAITS_PRODUCT_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename LhsType, typename RhsType, int Option>
  struct indexible_object_traits<Eigen::Product<LhsType, RhsType, Option>>
    : Eigen3::indexible_object_traits_base<Eigen::Product<LhsType, RhsType, Option>>
  {
  private:

    using Xpr = Eigen::Product<LhsType, RhsType, Option>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

  public:

    using typename Base::scalar_type;


    // nested_object not defined


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (zero<LhsType>)
      {
        return constant_coefficient{arg.lhs()};
      }
      else if constexpr (zero<RhsType>)
      {
        return constant_coefficient{arg.rhs()};
      }
      else if constexpr (constant_diagonal_matrix<LhsType> and constant_matrix<RhsType>)
      {
        return values::operation(
          std::multiplies<scalar_type>{},
          constant_diagonal_coefficient{arg.lhs()},
          constant_coefficient{arg.rhs()});
      }
      else if constexpr (constant_matrix<LhsType> and constant_diagonal_matrix<RhsType>)
      {
        return values::operation(
          std::multiplies<scalar_type>{},
          constant_coefficient{arg.lhs()},
          constant_diagonal_coefficient{arg.rhs()});
      }
      else
      {
        constexpr auto dim = dynamic_dimension<LhsType, 1> ? index_dimension_of_v<RhsType, 0> : index_dimension_of_v<LhsType, 1>;
        if constexpr (dim == dynamic_size)
        {
          return values::operation(
            std::multiplies<scalar_type>{},
            get_index_dimension_of<1>(arg.lhs()),
            values::operation(std::multiplies<scalar_type>{}, constant_coefficient{arg.lhs()}, constant_coefficient{arg.rhs()}));
        }
        else if constexpr (values::fixed<constant_coefficient<LhsType>>)
        {
          return values::operation(
            std::multiplies<scalar_type>{},
            values::operation(std::multiplies<scalar_type>{}, std::integral_constant<std::size_t, dim>{}, constant_coefficient{arg.lhs()}),
            constant_coefficient{arg.rhs()});
        }
        else
        {
          return values::operation(
            std::multiplies<scalar_type>{},
            values::operation(std::multiplies<scalar_type>{}, std::integral_constant<std::size_t, dim>{}, constant_coefficient{arg.rhs()}),
            constant_coefficient{arg.lhs()});
        }
      }
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return values::operation(std::multiplies<scalar_type>{},
        constant_diagonal_coefficient{arg.lhs()}, constant_diagonal_coefficient{arg.rhs()});
    }


    template<TriangleType t>
    static constexpr bool is_triangular = triangular_matrix<LhsType, t> and triangular_matrix<RhsType, t>;


    static constexpr bool is_triangular_adapter = false;


    /// A constant diagonal matrix times a hermitian matrix (or vice versa) is hermitian.
    static constexpr bool is_hermitian =
      (constant_diagonal_matrix<LhsType> and
        (not values::complex<scalar_type_of_t<LhsType>> or values::not_complex<constant_diagonal_coefficient<LhsType>>) and
        hermitian_matrix<RhsType, Applicability::permitted>) or
      (constant_diagonal_matrix<RhsType> and
        (not values::complex<scalar_type_of_t<RhsType>> or values::not_complex<constant_diagonal_coefficient<RhsType>>) and
        hermitian_matrix<LhsType, Applicability::permitted>);

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_PRODUCT_HPP
