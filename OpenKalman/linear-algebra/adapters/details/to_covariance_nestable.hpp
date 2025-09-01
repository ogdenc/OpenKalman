/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for OpenKalman::internal::to_covariance_nestable
 */

#ifndef OPENKALMAN_TOCOVARIANCENESTABLE_HPP
#define OPENKALMAN_TOCOVARIANCENESTABLE_HPP

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<covariance_nestable T, typename Arg>
  requires (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_shaped<Arg> or vector<Arg>))) and
    (index_dimension_of_v<Arg, 0> == index_dimension_of_v<T, 0>) and
    (not zero<T> or zero<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or vector<Arg>)
#else
  template<typename T, typename Arg, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg)
  {
    if constexpr(zero<Arg> or identity_matrix<Arg>)
    {
      // Pass through because both T and Arg are already zero or identity.
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<T>)
    {
      // diagonal -> diagonal
      if constexpr (vector<Arg> and not one_dimensional<Arg>)
      {
        return MatrixTraits<std::decay_t<T>>::make(std::forward<Arg>(arg));
      }
      else
      {
        // Pass through because Arg is already diagonal.
        static_assert(diagonal_matrix<Arg>);
        return std::forward<Arg>(arg);
      }
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return MatrixTraits<std::decay_t<T>>::make(std::forward<Arg>(arg));
    }
    else if constexpr (hermitian_matrix<T> and triangular_matrix<Arg>)
    {
      // non-diagonal triangular --> non-diagonal self-adjoint
      static_assert(hermitian_matrix<decltype(cholesky_square(std::declval<Arg&&>()))>);
      return cholesky_square(std::forward<Arg>(arg));
    }
    else if constexpr (triangular_matrix<T> and hermitian_matrix<Arg>)
    {
      // non-diagonal self-adjoint --> non-diagonal triangular
      static_assert(triangle_type_of_v<T> ==
        triangle_type_of_v<decltype(cholesky_factor<triangle_type_of_v<T>>(std::declval<Arg&&>()))>);
      return cholesky_factor<triangle_type_of_v<T>>(std::forward<Arg>(arg));
    }
    else if constexpr (triangular_matrix<T> and triangular_matrix<Arg> and
      (not triangle_type_of_v<T> == triangle_type_of_v<Arg>))
    {
      // upper triangular <--> lower triangular
      static_assert(triangle_type_of_v<T> == triangle_type_of_v<decltype(transpose(std::declval<Arg&&>()))>);
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (typed_matrix_nestable<Arg> and not covariance_nestable<Arg>)
    {
      // typed_matrix_nestable -> covariance_nestable:
      return MatrixTraits<std::decay_t<T>>::make(std::forward<Arg>(arg));
    }
    else
    {
      // Pass through if no conversion is necessary.
      static_assert(covariance_nestable<Arg>);
      static_assert(hermitian_matrix<T> == hermitian_matrix<Arg>);
      static_assert(triangular_matrix<T> == triangular_matrix<Arg>);
      static_assert(diagonal_matrix<T> == diagonal_matrix<Arg>);
      static_assert(triangular_matrix<Arg, triangle_type::lower> == triangular_matrix<Arg, triangle_type::lower>);
      static_assert(triangular_matrix<Arg, triangle_type::upper> == triangular_matrix<Arg, triangle_type::upper>);
      return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<covariance_nestable T, typename Arg> requires
    (covariance<Arg> or (typed_matrix<Arg> and (square_shaped<Arg> or vector<Arg>))) and
    (index_dimension_of_v<Arg, 0> == index_dimension_of_v<T, 0>) and
    (not zero<T> or zero<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or vector<Arg>)
#else
  template<typename T, typename Arg, typename, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg)
  {
    if constexpr (typed_matrix<Arg> and not covariance<Arg>)
    {
      return to_covariance_nestable<T>(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (diagonal_matrix<T>) // In this case, diagonal_matrix<Arg> or vector<Arg>.
    {
      if constexpr (triangular_covariance<Arg>)
      {
        return to_covariance_nestable<T>(std::forward<Arg>(arg).get_triangular_nested_matrix());
      }
      else
      {
        return to_covariance_nestable<T>(std::forward<Arg>(arg).get_self_adjoint_nested_matrix());
      }
    }
    else if constexpr (hermitian_matrix<T>)
    {
      return to_covariance_nestable<T>(std::forward<Arg>(arg).get_self_adjoint_nested_matrix());
    }
    else
    {
      static_assert(triangular_matrix<T>);
      return to_covariance_nestable<T>(std::forward<Arg>(arg).get_triangular_nested_matrix());
    }
  }


#ifdef __cpp_concepts
  template<typename Arg>
  requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_shaped<Arg> or vector<Arg>))
#else
  template<typename Arg, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg)
  {
    if constexpr (covariance_nestable<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (square_shaped<Arg>)
    {
      return make_hermitian_matrix(std::forward<Arg>(arg));
    }
    else
    {
      static_assert(vector<Arg>);
      return to_diagonal(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires covariance<Arg> or (typed_matrix<Arg> and (square_shaped<Arg> or vector<Arg>))
#else
  template<typename Arg, typename, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg)
  {
    if constexpr (typed_matrix<Arg> and not covariance<Arg>)
    {
      return to_covariance_nestable(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (triangular_covariance<Arg>)
    {
      return std::forward<Arg>(arg).get_triangular_nested_matrix();
    }
    else
    {
      return std::forward<Arg>(arg).get_self_adjoint_nested_matrix();
    }
  }


}

#endif
