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
   * Convert covariance matrix to a covariance base of type T. If T is void, convert to a triangular matrix if
   * covariance is a square root, or otherwise convert to a self-adjoint matrix.
   * @tparam T Type to which Arg is to be converted (optional).
   * @tparam Arg Type of covariance matrix to be converted
   * @param arg Covariance matrix to be converted.
   * @return A covariance base.
   */
  template<typename T, typename Arg>
  constexpr decltype(auto)
  convert_base_matrix(Arg&& arg) noexcept
  {
    static_assert(is_covariance_v<Arg> or is_typed_matrix_v<Arg>);
    using ArgBase = typename MatrixTraits<Arg>::BaseMatrix;

    // Typed matrices:
    if constexpr(is_typed_matrix_v<Arg>)
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
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
    else if constexpr(((is_triangular_v<ArgBase> and not is_diagonal_v<ArgBase>) or
      (is_square_root_v<Arg> and is_diagonal_v<ArgBase>)) and is_self_adjoint_v<T> and not is_diagonal_v<T>)
    {
      if constexpr((is_self_adjoint_v<ArgBase> and not is_square_root_v<Arg>) or
        (is_triangular_v<ArgBase> and is_square_root_v<Arg>))
        return Cholesky_square(std::forward<Arg>(arg).base_matrix());
      else
        return std::forward<Arg>(arg).get_apparent_base_matrix();
    }

    // strictly self-adjoint or diagonal non-square root --> strictly triangular
    else if constexpr(((not is_diagonal_v<ArgBase> and not is_triangular_v<ArgBase>) or
        (is_diagonal_v<ArgBase> and not is_square_root_v<Arg> )) and is_triangular_v<T> and not is_diagonal_v<T>)
    {
      if constexpr(is_diagonal_v<ArgBase>) // diagonal non-square root
      {
        return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
      }
      else // ArgBase is strictly self-adjoint.
      {
        using B = decltype(Cholesky_factor(std::forward<Arg>(arg).base_matrix()));
        if constexpr(is_upper_triangular_v<T> != is_upper_triangular_v<B>) // Converted triangle types don't match.
        {
          return Cholesky_factor(adjoint(std::forward<Arg>(arg).base_matrix()));
        }
        else // Converted triangle types match.
        {
          if constexpr((is_self_adjoint_v<ArgBase> and not is_square_root_v<Arg>) or
            (is_triangular_v<ArgBase> and is_square_root_v<Arg>))
            return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
          else
          {
            return std::forward<Arg>(arg).get_apparent_base_matrix();
          }
        }
      }
    }

    // upper triangular <--> lower triangular
    else if constexpr(is_Cholesky_v<Arg> and is_triangular_v<T>
      and is_upper_triangular_v<ArgBase> != is_upper_triangular_v<T>)
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
