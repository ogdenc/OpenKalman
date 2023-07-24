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
 * \brief Type traits as applied to Eigen::VectorwiseOp.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_VECTORWISEOP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_VECTORWISEOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ExpressionType, int Direction>
  struct IndexibleObjectTraits<Eigen::VectorwiseOp<ExpressionType, Direction>>
  {
    static constexpr std::size_t max_indices = max_indices_of_v<ExpressionType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      return OpenKalman::get_index_descriptor<N>(arg._expression());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<ExpressionType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<ExpressionType, b>;

    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<typename Eigen::VectorwiseOp<ExpressionType, Direction>::ExpressionTypeNested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg)._expression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::VectorwiseOp<equivalent_self_contained_t<ExpressionType>, Direction>;
      static_assert(self_contained<typename N::ExpressionTypeNested>,
        "This VectorWiseOp expression cannot be made self-contained");
      return N {make_self_contained(arg._expression())};
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg._expression()};
    }

    using scalar_type = scalar_type_of_t<ExpressionType>;

    // No get or set defined.
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_VECTORWISEOP_HPP
