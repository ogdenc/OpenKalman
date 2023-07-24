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

#ifndef OPENKALMAN_EIGEN3_TRAITS_DIAGONALWRAPPER_HPP
#define OPENKALMAN_EIGEN3_TRAITS_DIAGONALWRAPPER_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename DiagVectorType>
  struct IndexibleObjectTraits<Eigen::DiagonalWrapper<DiagVectorType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::DiagonalWrapper<DiagVectorType>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      if constexpr (has_dynamic_dimensions<DiagVectorType>) return static_cast<std::size_t>(arg.rows());
      else return Dimensions<index_dimension_of_v<DiagVectorType, 0> * index_dimension_of_v<DiagVectorType, 1>>{};
    }

    template<std::size_t N>
    static constexpr std::size_t dimension = has_dynamic_dimensions<DiagVectorType> ? dynamic_size :
      index_dimension_of_v<DiagVectorType, 0> * index_dimension_of_v<DiagVectorType, 1>;

    template<std::size_t N, typename Arg>
    static constexpr std::size_t dimension_at_runtime(const Arg& arg)
    {
      if constexpr (dimension<N> == dynamic_size) return static_cast<std::size_t>(arg.rows());
      else return dimension<N>;
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<DiagVectorType, b>;

    template<Likelihood b>
    static constexpr bool is_square = true;

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename DiagVectorType::Nested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      decltype(auto) d = std::forward<Arg>(arg).diagonal();
      using D = decltype(d);
      using NCD = std::conditional_t<
        std::is_const_v<std::remove_reference_t<Arg>> or std::is_const_v<DiagVectorType>,
        D, std::conditional_t<std::is_lvalue_reference_v<D>, std::decay_t<D>&, std::decay_t<D>>>;
      return const_cast<NCD>(std::forward<decltype(d)>(d));
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
      return DiagonalMatrix<decltype(d)> {d};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_coefficient {arg.diagonal()};
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = true;

    static constexpr bool is_triangular_adapter = false;

    template<Likelihood b>
    static constexpr bool is_diagonal_adapter = vector<DiagVectorType, 0, b>;

    // is_hermitian not defined because matrix is diagonal;

    // make_hermitian_adapter(Arg&& arg) not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_DIAGONALWRAPPER_HPP
