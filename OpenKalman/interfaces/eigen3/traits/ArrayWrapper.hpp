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
 * \brief Type traits as applied to Eigen::ArrayWrapper.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_ARRAYWRAPPER_HPP
#define OPENKALMAN_EIGEN3_TRAITS_ARRAYWRAPPER_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType>
  struct IndexTraits<Eigen::ArrayWrapper<XprType>>
  {
    static constexpr std::size_t max_indices = max_indices_of_v<XprType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      return OpenKalman::get_index_descriptor<N>(arg.nestedExpression());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<XprType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<XprType, b>;
  };


  template<typename XprType>
  struct Dependencies<Eigen::ArrayWrapper<XprType>>
  {
  private:

    using NestedXpr = typename Eigen::ArrayWrapper<XprType>::NestedExpressionType;

  public:

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<NestedXpr>;

    template<std::size_t i, typename Arg>
    static NestedXpr get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      if constexpr (std::is_lvalue_reference_v<NestedXpr>)
        return const_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
      else
        return static_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::ArrayWrapper<equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
        return N {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }
  };


  template<typename XprType>
  struct SingleConstant<Eigen::ArrayWrapper<XprType>> : SingleConstant<std::decay_t<XprType>>
  {
    SingleConstant(const Eigen::ArrayWrapper<XprType>& xpr) :
      SingleConstant<std::decay_t<XprType>> {xpr.nestedExpression()} {};
  };


  template<typename XprType>
  struct TriangularTraits<Eigen::ArrayWrapper<XprType>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_triangular_adapter = false;
  };


#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> XprType>
  struct HermitianTraits<Eigen::ArrayWrapper<XprType>>
#else
  template<typename XprType>
  struct HermitianTraits<Eigen::ArrayWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType, Likelihood::maybe>>>
#endif
  {
    static constexpr bool is_hermitian = true;
  };


  template<typename XprType>
  struct Conversions<Eigen::ArrayWrapper<XprType>>
  {
    template<typename Arg>
    static constexpr decltype(auto) to_diagonal(Arg&& arg) { return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg))); }

    template<typename Arg>
    static constexpr decltype(auto) diagonal_of(Arg&& arg) { return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg))); }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_ARRAYWRAPPER_HPP
