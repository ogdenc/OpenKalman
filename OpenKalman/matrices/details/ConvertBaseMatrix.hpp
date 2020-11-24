/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_CONVERTBASEMATRIX_H
#define OPENKALMAN_CONVERTBASEMATRIX_H

namespace OpenKalman::internal
{
  /**
   * /internal
   * Convert covariance matrix to a covariance base of type T. If T is void, convert to a triangular matrix if
   * covariance is a square root, or otherwise convert to a self-adjoint matrix.
   * \tparam T Type to which Arg is to be converted (optional).
   * \tparam Arg Type of covariance matrix to be converted
   * \param arg Covariance matrix to be converted.
   * \return A covariance base.
   */
  template<typename T, typename Arg>
#ifdef __cpp_concepts
    requires (std::is_void_v<T> or covariance_base<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
  constexpr decltype(auto)
  convert_base_matrix(Arg&& arg) noexcept
  {
    static_assert(covariance<Arg> or typed_matrix<Arg>);
    using ArgBase = nested_matrix_t<Arg>;

    // Typed matrices:
    if constexpr(typed_matrix<Arg>)
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
      using SA = typename MatrixTraits<ArgBase>::template SelfAdjointBaseType<>;
      if constexpr(std::is_void_v<T>)
        return std::forward<Arg>(arg).base_matrix();
      else
        return MatrixTraits<T>::make(base_matrix(MatrixTraits<SA>::make(std::forward<Arg>(arg).base_matrix())));
    }

    // Natural conversion:
    else if constexpr(std::is_void_v<T>)
    {
      return std::forward<Arg>(arg).get_apparent_base_matrix();
    }

    // strictly triangular or diagonal square root --> self-adjoint
    else if constexpr(((triangular_matrix<ArgBase> and not diagonal_matrix<ArgBase>) or
      (square_root_covariance<Arg> and diagonal_matrix<ArgBase>)) and self_adjoint_matrix<T> and not diagonal_matrix<T>)
    {
      if constexpr((self_adjoint_matrix<ArgBase> and not square_root_covariance<Arg>) or
        (triangular_matrix<ArgBase> and square_root_covariance<Arg>))
        return Cholesky_square(std::forward<Arg>(arg).base_matrix());
      else
        return std::forward<Arg>(arg).get_apparent_base_matrix();
    }

    // strictly self-adjoint or diagonal non-square root --> strictly triangular
    else if constexpr(((not diagonal_matrix<ArgBase> and not triangular_matrix<ArgBase>) or
        (diagonal_matrix<ArgBase> and not square_root_covariance<Arg> )) and triangular_matrix<T> and not diagonal_matrix<T>)
    {
      if constexpr(diagonal_matrix<ArgBase>) // diagonal non-square root
      {
        return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
      }
      else // ArgBase is strictly self-adjoint.
      {
        using B = decltype(Cholesky_factor(std::forward<Arg>(arg).base_matrix()));
        if constexpr(internal::same_triangle_type_as<B, T>) // Converted triangle types match.
        {
          if constexpr((self_adjoint_matrix<ArgBase> and not square_root_covariance<Arg>) or
            (triangular_matrix<ArgBase> and square_root_covariance<Arg>))
            return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
          else
          {
            return std::forward<Arg>(arg).get_apparent_base_matrix();
          }
        }
        else // Converted triangle types don't match.
        {
          return Cholesky_factor(adjoint(std::forward<Arg>(arg).base_matrix()));
        }
      }
    }

    // upper triangular <--> lower triangular
    else if constexpr(cholesky_form<Arg> and triangular_matrix<T> and not (internal::same_triangle_type_as<ArgBase, T>))
    {
      return adjoint(std::forward<Arg>(arg).base_matrix());
    }

    // pass through
    else
    {
      return std::forward<Arg>(arg).base_matrix();
    }
  }


}

#endif //OPENKALMAN_CONVERTBASEMATRIX_H
