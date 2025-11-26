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
 * \brief Type traits as applied to Eigen::DiagonalWrapper.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_DIAGONALWRAPPER_HPP
#define OPENKALMAN_EIGEN_TRAITS_DIAGONALWRAPPER_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename DiagVectorType>
  struct object_traits<Eigen::DiagonalWrapper<DiagVectorType>>
    : Eigen3::object_traits_base<Eigen::DiagonalWrapper<DiagVectorType>>
  {
  private:

    using Xpr = Eigen::DiagonalWrapper<DiagVectorType>;
    using Base = Eigen3::object_traits_base<Xpr>;

  public:

    template<typename Arg>
    static constexpr auto
    count_indices(const Arg& arg)
    {
      if constexpr (Arg::RowsAtCompileTime == 1 and Arg::ColsAtCompileTime == 1)
        return std::integral_constant<std::size_t, 0_uz>{};
      else
        return std::integral_constant<std::size_t, 2_uz>{};
    }


    template<typename Arg, typename N>
    static constexpr auto get_pattern_collection(const Arg& arg, N)
    {
      if constexpr (has_dynamic_dimensions<DiagVectorType>) return static_cast<std::size_t>(arg.rows());
      else return Dimensions<index_dimension_of_v<DiagVectorType, 0> * index_dimension_of_v<DiagVectorType, 1>>{};
    }


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      auto&& d = std::forward<Arg>(arg).diagonal();
      using D = decltype(d);
      using NCD = std::conditional_t<
        std::is_const_v<std::remove_reference_t<Arg>> or std::is_const_v<DiagVectorType>,
        D, std::conditional_t<std::is_lvalue_reference_v<D>, std::decay_t<D>&, std::decay_t<D>>>;
      return const_cast<NCD>(std::forward<decltype(d)>(d));
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_value {arg.diagonal()};
    }


    template<applicability b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<DiagVectorType, b>;


    template<applicability b>
    static constexpr bool is_square = true;


    static constexpr triangle_type triangle_type_value = triangle_type::diagonal;


    static constexpr bool is_triangular_adapter = false;

    // is_hermitian not defined because matrix is diagonal;

    // make_hermitian_adapter(Arg&& arg) not defined

  };

}

#endif
