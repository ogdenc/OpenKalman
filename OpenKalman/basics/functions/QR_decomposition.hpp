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
 * \brief Definition for \ref QR_decomposition function.
 */

#ifndef OPENKALMAN_QR_DECOMPOSITION_HPP
#define OPENKALMAN_QR_DECOMPOSITION_HPP

namespace OpenKalman
{
  /**
   * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns U as an upper \ref triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A>
  constexpr triangular_matrix<TriangleType::upper> decltype(auto)
#else
  template<typename A, std::enable_if_t<indexible<A>, int> = 0>
  constexpr decltype(auto)
#endif
  QR_decomposition(A&& a)
  {
    if constexpr (triangular_matrix<A, TriangleType::upper>)
    {
      return internal::clip_square_shaped(std::forward<A>(a));
    }
    else if constexpr (constant_matrix<A>)
    {
      using Scalar = scalar_type_of_t<A>;

      auto elem = constant_coefficient{a} * internal::constexpr_sqrt(internal::index_to_scalar_constant<Scalar>(get_index_dimension_of<0>(a)));

      if constexpr (dynamic_dimension<A, 1>)
      {
        auto dim = Dimensions {get_index_dimension_of<1>(a)};
        auto row1 = make_constant<A>(elem, Dimensions<1>{}, dim);

        auto m = make_dense_object<A>(dim, dim);

        if (get_dimension_size_of(dim) == 1) m = std::move(row1);
        else m = concatenate<0>(std::move(row1), make_zero<A>(dim - Dimensions<1>{}, dim));

        auto ret {make_triangular_matrix<TriangleType::upper>(std::move(m))};

        // \todo Fix this:
        if constexpr (has_untyped_index<A, 1>) return ret;
        else return SquareRootCovariance {std::move(ret), get_vector_space_descriptor<1>(a)};
      }
      else
      {
        auto ret = make_triangular_matrix<TriangleType::upper>([](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 1>;
          auto row1 = make_constant<A>(elem, Dimensions<1>{}, Dimensions<dim>{});
          if constexpr (dim == 1) return row1;
          else return concatenate<0>(std::move(row1), make_zero<A>(Dimensions<dim - 1>{}, Dimensions<dim>{}));
        }(elem));

        // \todo Fix this:
        using C = vector_space_descriptor_of_t<A, 1>;
        if constexpr (euclidean_vector_space_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      decltype(auto) ret = [](A&& a) -> decltype(auto) {
        if constexpr (interface::QR_decomposition_defined_for<A, A&&>)
        {
          return interface::library_interface<std::decay_t<A>>::QR_decomposition(std::forward<A>(a));
        }
        else
        {
          static_assert(interface::LQ_decomposition_defined_for<A, A&&>,
            "QR_decomposition requires definition of at least one of interface::QR_decomposition or interface::LQ_decomposition");
          return transpose(interface::library_interface<std::decay_t<A>>::LQ_decomposition(transpose(std::forward<A>(a))));
        }
      }(std::forward<A>(a));
      using Ret = decltype(ret);

      static_assert(triangular_matrix<Ret, TriangleType::upper>,
        "Interface implementation error: interface::library_interface<T>::QR_decomposition must return an upper triangular_matrix.");

      // \todo Fix this:
      if constexpr (has_untyped_index<A, 1>) return ret;
      else return SquareRootCovariance {std::forward<Ret>(ret), get_vector_space_descriptor<1>(a)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_QR_DECOMPOSITION_HPP
