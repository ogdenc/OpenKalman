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
  requires (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
    (MatrixTraits<Arg>::rows == MatrixTraits<T>::rows) and
    (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
  template<typename T, typename Arg, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg) noexcept
  {
    if constexpr(zero_matrix<T>)
    {
      // Pass through because Arg is already zero.
      static_assert(zero_matrix<Arg>);
      return std::forward<Arg>(arg);
    }
    else if constexpr(identity_matrix<T>)
    {
      // Pass through because Arg is already identity.
      static_assert(identity_matrix<Arg>);
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      static_assert(not zero_matrix<T>);
      return MatrixTraits<T>::zero();
    }
    else if constexpr (identity_matrix<Arg>)
    {
      static_assert(not identity_matrix<T>);
      return MatrixTraits<T>::identity();
    }
    else if constexpr (diagonal_matrix<T>)
    {
      // diagonal -> diagonal
      if constexpr (column_vector<Arg> and not one_by_one_matrix<Arg>)
      {
        return MatrixTraits<T>::make(std::forward<Arg>(arg));
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
      return MatrixTraits<T>::make(std::forward<Arg>(arg));
    }
    else if constexpr (self_adjoint_matrix<T> and triangular_matrix<Arg>)
    {
      // non-diagonal triangular --> non-diagonal self-adjoint
      static_assert(self_adjoint_matrix<decltype(Cholesky_square(std::declval<Arg&&>()))>);
      return Cholesky_square(std::forward<Arg>(arg));
    }
    else if constexpr (triangular_matrix<T> and self_adjoint_matrix<Arg>)
    {
      // non-diagonal self-adjoint --> non-diagonal triangular
      static_assert(triangle_type_of_v<T> ==
        triangle_type_of_v<decltype(Cholesky_factor<triangle_type_of_v<T>>(std::declval<Arg&&>()))>);
      return Cholesky_factor<triangle_type_of_v<T>>(std::forward<Arg>(arg));
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
      return MatrixTraits<T>::make(std::forward<Arg>(arg));
    }
    else
    {
      // Pass through if no conversion is necessary.
      static_assert(covariance_nestable<Arg>);
      static_assert(self_adjoint_matrix<T> == self_adjoint_matrix<Arg>);
      static_assert(triangular_matrix<T> == triangular_matrix<Arg>);
      static_assert(diagonal_matrix<T> == diagonal_matrix<Arg>);
      static_assert(lower_triangular_matrix<Arg> == lower_triangular_matrix<Arg>);
      static_assert(upper_triangular_matrix<Arg> == upper_triangular_matrix<Arg>);
      return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<covariance_nestable T, typename Arg> requires
    (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
    (MatrixTraits<Arg>::rows == MatrixTraits<T>::rows) and
    (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
    (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
  template<typename T, typename Arg, typename, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg) noexcept
  {
    if constexpr (typed_matrix<Arg> and not covariance<Arg>)
    {
      return to_covariance_nestable<T>(std::forward<Arg>(arg).nested_matrix());
    }
    else if constexpr (diagonal_matrix<T>) // In this case, diagonal_matrix<Arg> or column_vector<Arg>.
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
    else if constexpr (self_adjoint_matrix<T>)
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
  requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
  template<typename Arg, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg) noexcept
  {
    if constexpr (covariance_nestable<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (square_matrix<Arg>)
    {
      using SA = typename MatrixTraits<Arg>::template SelfAdjointMatrixFrom<>;
      static_assert(self_adjoint_matrix<decltype(MatrixTraits<SA>::make(std::forward<Arg>(arg)))>);
      return MatrixTraits<SA>::make(std::forward<Arg>(arg));
    }
    else
    {
      static_assert(column_vector<Arg>);
      using D = typename MatrixTraits<Arg>::template DiagonalMatrixFrom<>;
      static_assert(diagonal_matrix<decltype(MatrixTraits<D>::make(std::forward<Arg>(arg)))>);
      return MatrixTraits<D>::make(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
  template<typename Arg, typename, typename>
#endif
  constexpr decltype(auto)
  to_covariance_nestable(Arg&& arg) noexcept
  {
    if constexpr (typed_matrix<Arg> and not covariance<Arg>)
    {
      return to_covariance_nestable(std::forward<Arg>(arg).nested_matrix());
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


} // namespace OpenKalman::internal

#endif //OPENKALMAN_TOCOVARIANCENESTABLE_HPP
