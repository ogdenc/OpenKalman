/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref LQ_decomposition function.
 */

#ifndef OPENKALMAN_LQ_DECOMPOSITION_HPP
#define OPENKALMAN_LQ_DECOMPOSITION_HPP

namespace OpenKalman
{
  /**
   * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed satisfying <code>triangular_matrix<A, triangle_type::lower></code>
   * \returns L as a lower \ref triangular_matrix which is also \ref square_shaped
   */
#ifdef __cpp_concepts
  template<indexible A> requires (not euclidean_transformed<A>)
  constexpr triangular_matrix<triangle_type::lower> decltype(auto)
#else
  template<typename A, std::enable_if_t<indexible<A> and (not euclidean_transformed<A>), int> = 0>
  constexpr decltype(auto)
#endif
  LQ_decomposition(A&& a)
  {
    if constexpr (triangular_matrix<A, triangle_type::lower>)
    {
      return internal::clip_square_shaped(std::forward<A>(a));
    }
    else if constexpr (constant_matrix<A>)
    {
      using Scalar = scalar_type_of_t<A>;

      auto elem = constant_value{a} * values::sqrt(values::cast_to<Scalar>(get_index_dimension_of<1>(a)));

      if constexpr (dynamic_dimension<A, 0>)
      {
        auto dim = patterns::Dimensions {get_index_dimension_of<0>(a)};
        auto col1 = make_constant<A>(elem, dim, patterns::Axis{});

        auto m {make_dense_object<A>(dim, dim)};

        if (get_dimension(dim) == 1) m = std::move(col1);
        else m = concatenate<1>(std::move(col1), make_zero<A>(dim, dim - patterns::Axis{}));

        auto ret {make_triangular_matrix<triangle_type::lower>(std::move(m))};

        // \todo Fix this:
        if constexpr (has_untyped_index<A, 0>) return ret;
        else return SquareRootCovariance {std::move(ret), get_pattern_collection<0>(a)};
      }
      else
      {
        auto ret = make_triangular_matrix<triangle_type::lower>([](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 0>;
          auto col1 = make_constant<A>(elem, patterns::Dimensions<dim>{}, patterns::Axis{});
          if constexpr (dim == 1) return col1;
          else return concatenate<1>(std::move(col1), make_zero<A>(patterns::Dimensions<dim>{}, patterns::Dimensions<dim - 1>{}));
        }(elem));

        // \todo Fix this:
        using C = vector_space_descriptor_of_t<A, 0>;
        if constexpr (patterns::euclidean_pattern<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      decltype(auto) ret = [](A&& a) -> decltype(auto) {
        if constexpr (interface::LQ_decomposition_defined_for<A, A&&>)
        {
          return interface::library_interface<stdex::remove_cvref_t<A>>::LQ_decomposition(std::forward<A>(a));
        }
        else
        {
          static_assert(interface::QR_decomposition_defined_for<A, A&&>,
            "LQ_decomposition requires definition of at least one of interface::LQ_decomposition or interface::QR_decomposition");
          return transpose(interface::library_interface<stdex::remove_cvref_t<A>>::QR_decomposition(transpose(std::forward<A>(a))));
        }
      }(std::forward<A>(a));
      using Ret = decltype(ret);

      static_assert(triangular_matrix<Ret, triangle_type::lower>,
        "Interface implementation error: interface::library_interface<T>::LQ_decomposition must return a lower triangular_matrix.");

      // \todo Fix this:
      if constexpr (has_untyped_index<A, 0>) return ret;
      else return SquareRootCovariance {std::forward<Ret>(ret), get_pattern_collection<0>(a)};
    }
  }


}

#endif
