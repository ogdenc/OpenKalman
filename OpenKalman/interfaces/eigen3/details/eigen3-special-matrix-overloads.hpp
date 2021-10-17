/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded functions for Eigen3 extensions
 */

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg)
  {
    return std::forward<Arg>(arg).nested_matrix();
  }


  /// Convert to self-contained version of the matrix.
#ifdef __cpp_concepts
  template<typename...Ts, eigen_zero_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr Arg&&
    make_self_contained(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


  /// Convert to self-contained version of the matrix.
#ifdef __cpp_concepts
  template<typename...Ts, eigen_constant_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr Arg&&
  make_self_contained(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


  /// Convert to self-contained version of the special matrix.
#ifdef __cpp_concepts
  template<typename...Ts, typename Arg> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<Arg> or std::is_lvalue_reference_v<nested_matrix_t<Arg>> or
      ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return arg.rows();
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return arg.rows();
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return row_count(nested_matrix(std::forward<Arg>(arg)));
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return arg.cols();
    else
      return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return arg.cols();
    else
      return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  column_count(Arg&& arg)
  {
    return row_count(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return column_count(nested_matrix(std::forward<Arg>(arg)));
    else
      return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or (square_matrix<Arg> and not one_by_one_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and
    (dynamic_shape<Arg> or (square_matrix<Arg> and not one_by_one_matrix<Arg>)), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (dynamic_rows<Arg>)
    {
      return ZeroMatrix<Scalar, 0, 1> {row_count(arg)};
    }
    else
    {
      return ZeroMatrix<Scalar, MatrixTraits<Arg>::rows, 1> {};
    }
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires dynamic_shape<Arg> or (square_matrix<Arg> and not one_by_one_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and
    (dynamic_shape<Arg> or (square_matrix<Arg> and not one_by_one_matrix<Arg>)), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (dynamic_rows<Arg>)
    {
      return ConstantMatrix<Scalar, constant, 0, 1> {row_count(arg)};
    }
    else
    {
      return ConstantMatrix<Scalar, constant, MatrixTraits<Arg>::rows, 1> {};
    }

  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  diagonal_of(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    return diagonal_of(nested_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr (dynamic_rows<Arg> and dynamic_columns<Arg>)
    {
      return ZeroMatrix<Scalar, 0, 0> {column_count(arg), row_count(arg)};
    }
    else if constexpr (dynamic_rows<Arg> and not dynamic_columns<Arg>)
    {
      constexpr auto cols = MatrixTraits<Arg>::columns;
      return ZeroMatrix<Scalar, cols, 0> {cols, row_count(arg)};
    }
    else if constexpr (not dynamic_rows<Arg> and dynamic_columns<Arg>)
    {
      constexpr auto rows = MatrixTraits<Arg>::rows;
      return ZeroMatrix<Scalar, 0, rows> {column_count(arg), rows};
    }
    else // if constexpr (not dynamic_rows<Arg> and not dynamic_columns<Arg>)
    {
      constexpr auto rows = MatrixTraits<Arg>::rows;
      constexpr auto cols = MatrixTraits<Arg>::columns;

      if constexpr (rows == cols) return std::forward<Arg>(arg);
      else return ZeroMatrix<Scalar, cols, rows> {};
    }
  }


  namespace detail
  {
    template<auto constant, typename Arg>
    constexpr decltype(auto)
    eigen_constant_transpose_impl(Arg&& arg) noexcept
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      if constexpr (dynamic_rows<Arg> and dynamic_columns<Arg>)
      {
        return ConstantMatrix<Scalar, constant, 0, 0> {column_count(arg), row_count(arg)};
      }
      else if constexpr (dynamic_rows<Arg> and not dynamic_columns<Arg>)
      {
        constexpr auto cols = MatrixTraits<Arg>::columns;
        return ConstantMatrix<Scalar, constant, cols, 0> {cols, row_count(arg)};
      }
      else if constexpr (not dynamic_rows<Arg> and dynamic_columns<Arg>)
      {
        constexpr auto rows = MatrixTraits<Arg>::rows;
        return ConstantMatrix<Scalar, constant, 0, rows> {column_count(arg), rows};
      }
      else // if constexpr (not dynamic_rows<Arg> and not dynamic_columns<Arg>)
      {
        constexpr auto rows = MatrixTraits<Arg>::rows;
        constexpr auto cols = MatrixTraits<Arg>::columns;

        if constexpr (rows == cols) return std::forward<Arg>(arg);
        else return ConstantMatrix<Scalar, constant, cols, rows> {};
      }
    }
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    return detail::eigen_constant_transpose_impl<MatrixTraits<Arg>::constant>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg)
  {
    if constexpr (diagonal_matrix<Arg> or not complex_number<typename MatrixTraits<Arg>::Scalar>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (self_adjoint_matrix<nested_matrix_t<Arg>>)
    {
      static_assert(internal::same_storage_triangle_as<Arg, nested_matrix_t<Arg>>);
      return transpose(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      constexpr auto t = (lower_triangular_storage<Arg> ? TriangleType::upper : TriangleType::lower);
      return MatrixTraits<Arg>::template make<t>(
        make_self_contained<Arg>(transpose(nested_matrix(std::forward<Arg>(arg)))));
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg)
  {
    if constexpr (diagonal_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (triangular_matrix<nested_matrix_t<Arg>>)
    {
      static_assert(OpenKalman::internal::same_triangle_type_as<Arg, nested_matrix_t<Arg>>);
      return transpose(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      constexpr auto t = lower_triangular_matrix<Arg> ? TriangleType::upper : TriangleType::lower;
      return MatrixTraits<Arg>::template make<t>(
        make_self_contained<Arg>(transpose(nested_matrix(std::forward<Arg>(arg)))));
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return transpose(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (complex_number<Scalar>)
    {
#ifdef __cpp_lib_constexpr_complex
      constexpr auto adj = std::conj(constant);
#else
      constexpr auto adj = std::complex(constant.real, -constant.imag);
#endif
      return detail::eigen_constant_transpose_impl<adj>(std::forward<Arg>(arg));
    }
    else
    {
      return detail::eigen_constant_transpose_impl<constant>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    if constexpr (complex_number<typename MatrixTraits<Arg>::Scalar>)
    {
      auto n = make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)).conjugate());
      return DiagonalMatrix<std::decay_t<decltype(n)>> {std::move(n)};
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg)
  {
    if constexpr (self_adjoint_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (not complex_number<typename MatrixTraits<Arg>::Scalar>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (self_adjoint_matrix<nested_matrix_t<Arg>>)
    {
      // Arg is eigen_triangular_expr, but its nested matrix is already self-adjoint.
      return nested_matrix(std::forward<Arg>(arg));
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      // Arg is a complex, diagonal eigen_triangular_expr.
      auto col = make_self_contained<Arg>(diagonal_of(nested_matrix(std::forward<Arg>(arg))).conjugate());
      return DiagonalMatrix<std::decay_t<decltype(col)>> {std::move(col)};
    }
    else
    {
      // Arg is a complex, non-diagonal eigen_triangular_expr.
      static_assert(eigen_triangular_expr<Arg>);
      constexpr auto t = lower_triangular_storage<Arg> or lower_triangular_matrix<Arg> ?
        TriangleType::upper : TriangleType::lower;
      return MatrixTraits<Arg>::template make<t>(
        make_self_contained<Arg>(adjoint(nested_matrix(std::forward<Arg>(arg)))));
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    if constexpr (dynamic_shape<Arg>)
      assert(row_count(arg) == column_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    if constexpr (dynamic_shape<Arg>)
      assert(row_count(arg) == column_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr typename MatrixTraits<Arg>::Scalar
  determinant(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(1);
    }
    else if constexpr (eigen_constant_expr<nested_matrix_t<Arg>>)
    {
      if constexpr (dynamic_shape<Arg>)
        return std::pow(MatrixTraits<nested_matrix_t<Arg>>::constant, row_count(arg));
      else
        return OpenKalman::internal::constexpr_pow(MatrixTraits<Arg>::constant, MatrixTraits<Arg>::rows);
    }
    else
    {
      static_assert(eigen_native<nested_matrix_t<Arg>>);
      return nested_matrix(std::forward<Arg>(arg)).prod();
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg> or (eigen_self_adjoint_expr<Arg> and eigen_constant_expr<nested_matrix_t<Arg>>))
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(1);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<Arg>>)
    {
      return determinant(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      static_assert(eigen_native<nested_matrix_t<Arg>>);
      return determinant(make_native_matrix(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
    return static_cast<decltype(MatrixTraits<Arg>::constant)>(MatrixTraits<Arg>::constant * row_count(arg));
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr typename MatrixTraits<Arg>::Scalar
  trace(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<nested_matrix_t<Arg>>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return row_count(arg);
    }
    else if constexpr (eigen_constant_expr<nested_matrix_t<Arg>>)
    {
      return MatrixTraits<nested_matrix_t<Arg>>::constant * row_count(arg);
    }
    else
    {
      static_assert(eigen_native<nested_matrix_t<Arg>>);
      return nested_matrix(std::forward<Arg>(arg)).sum();
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<nested_matrix_t<Arg>>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return row_count(arg);
    }
    else if constexpr (eigen_constant_expr<nested_matrix_t<Arg>>)
    {
      return MatrixTraits<nested_matrix_t<Arg>>::constant * row_count(arg);
    }
    else
    {
      return trace(nested_matrix(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg, typename U> requires one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_native<Arg> and one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>) and
    not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg(0, 0) = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u * adjoint(u)));
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_native Arg, typename U> requires one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_native<Arg> and one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto b = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u * adjoint(u)));
    return Eigen3::eigen_matrix_t<typename MatrixTraits<Arg>::Scalar, 1, 1> {b};
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg, diagonal_matrix U> requires (dynamic_shape<Arg> or square_matrix<Arg>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<eigen_zero_expr<Arg> and diagonal_matrix<U> and
    (dynamic_shape<Arg> or square_matrix<Arg>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(const Arg&, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    return DiagonalMatrix {std::sqrt(alpha) * u.diagonal()};
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg, typename U> requires (not diagonal_matrix<U>) and
    (dynamic_shape<Arg> or square_matrix<Arg>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<eigen_zero_expr<Arg> and not diagonal_matrix<U> and
    (dynamic_shape<Arg> or square_matrix<Arg>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    return std::sqrt(alpha) * u;
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, eigen_diagonal_expr U> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_diagonal_expr<Arg> and eigen_diagonal_expr<U> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg.nested_matrix() =
    (nested_matrix(arg).array().square() + alpha * nested_matrix(u).array().square()).sqrt().matrix();
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, diagonal_matrix U> requires (not eigen_diagonal_expr<U>) and
  (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<eigen_diagonal_expr<Arg> and diagonal_matrix<U> and
    (not eigen_diagonal_expr<U>) and (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg.nested_matrix() =
    (nested_matrix(arg).array().square() + alpha * u.diagonal().array().square()).sqrt().matrix();
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, eigen_diagonal_expr U> requires
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_diagonal_expr<Arg> and eigen_diagonal_expr<U> and
      (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto sa = (nested_matrix(arg).array().square() + alpha * nested_matrix(u).array().square()).sqrt().matrix();
    return DiagonalMatrix {make_self_contained(std::move(sa))};
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, diagonal_matrix U> requires (not eigen_diagonal_expr<U>) and
  (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_diagonal_expr<Arg> and diagonal_matrix<U> and (not eigen_diagonal_expr<U>) and
      (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto d = (nested_matrix(arg).array().square() + alpha * u.diagonal().array().square()).sqrt().matrix();
    return DiagonalMatrix<decltype(d)> {std::move(d)};
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, typename U> requires (not diagonal_matrix<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<eigen_diagonal_expr<Arg> and not diagonal_matrix<U> and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto m = make_native_matrix(std::forward<Arg>(arg));
    TriangularMatrix<std::remove_const_t<decltype(m)>> sa {std::move(m)};
    rank_update(sa, u, alpha);
    return sa;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg, typename U> requires
  (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
    Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg.view().rankUpdate(u, alpha);
    return arg;
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg, typename U> requires
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<Eigen3::eigen_triangular_expr<Arg> and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (not std::is_const_v<std::remove_reference_t<Arg>>), int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto t = lower_triangular_matrix<Arg> ? Eigen::Lower : Eigen::Upper;
    for (Eigen::Index i = 0; i < Eigen::Index(MatrixTraits<U>::columns); ++i)
    {
      if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(arg.nested_matrix(), u.col(i), alpha) >= 0)
      {
        throw (std::runtime_error("TriangularMatrix rank_update: product is not positive definite"));
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Arg, typename U> requires
    (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>) and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<
    (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>) and
    (Eigen3::eigen_matrix<U> or Eigen3::eigen_triangular_expr<U> or Eigen3::eigen_self_adjoint_expr<U> or
      Eigen3::eigen_diagonal_expr<U> or Eigen3::euclidean_expr<U>) and
    (MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    if constexpr (diagonal_matrix<Arg>)
    {
      if constexpr (Eigen3::eigen_triangular_expr<Arg>)
      {
        return rank_update(DiagonalMatrix {diagonal_of(std::forward<Arg>(arg))}, u, alpha);
      }
      else
      {
        static_assert(Eigen3::eigen_self_adjoint_expr<Arg>);
        constexpr TriangleType t =
          MatrixTraits<Arg>::storage_triangle == TriangleType::upper ? TriangleType::upper : TriangleType::lower;
        constexpr unsigned int uplo = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        std::decay_t<native_matrix_t<Arg>> m;
        m.template triangularView<uplo>() = std::forward<Arg>(arg);
        Eigen::SelfAdjointView<decltype(m), uplo> {m}.rankUpdate(u, alpha);
        return SelfAdjointMatrix<decltype(m), t> {std::move(m)};
      }
    }
    else
    {
      // We want sa to be a non-const lvalue reference:
      std::decay_t<Arg> sa {std::forward<Arg>(arg)};
      rank_update(sa, u, alpha);
      return sa;
    }
  }


  /// Solve the equation AX = B for X. A is an invertible square matrix.
#ifdef __cpp_concepts
  template<eigen_constant_expr A, eigen_matrix B> requires square_matrix<A> and
  (MatrixTraits<A>::rows == MatrixTraits<B>::rows)
#else
  template<typename A, typename B, std::enable_if_t<eigen_constant_expr<A> and eigen_matrix<B> and square_matrix<A> and
    (MatrixTraits<A>::rows == MatrixTraits<B>::rows), int> = 0>
#endif
  constexpr auto
  solve(const A& a, const B& b)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    using M = eigen_matrix_t<typename MatrixTraits<B>::Scalar, MatrixTraits<A>::rows, MatrixTraits<B>::columns>;
    if constexpr (eigen_constant_expr<B>)
    {
      constexpr auto c = MatrixTraits<B>::constant / (MatrixTraits<A>::columns * MatrixTraits<A>::constant);
      /// \todo Add options for a dynamically-sized ConstantMatrix.
      return ConstantMatrix<Scalar, c, MatrixTraits<A>::rows, MatrixTraits<B>::columns> {};
    }
    else if constexpr (zero_matrix<B>)
    {
      return make_ZeroMatrix<Scalar, MatrixTraits<A>::rows, MatrixTraits<B>::columns>(row_count(a), column_count(b));
    }
    else if constexpr (one_by_one_matrix<A>)
    {
      static_assert(one_by_one_matrix<B>);
      return b / MatrixTraits<A>::constant;
    }
    else
    {
      return M {a.lu().solve(b)};
    }
  }


  /// Solve the equation AX = B for X. A is a diagonal matrix.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr A, eigen_matrix B> requires (MatrixTraits<A>::rows == MatrixTraits<B>::rows)
#else
  template<typename A, typename B, std::enable_if_t<eigen_diagonal_expr<A> and eigen_matrix<B> and
    (MatrixTraits<A>::rows == MatrixTraits<B>::rows), int> = 0>
#endif
  inline auto
  solve(const A& a, const B& b)
  {
    return (b.array().colwise() / nested_matrix(a).array()).matrix();
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr A, eigen_matrix B>
#else
  template<typename A, typename B, std::enable_if_t<eigen_self_adjoint_expr<A> and eigen_matrix<B>, int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    static_assert(std::is_same_v<Scalar, typename MatrixTraits<B>::Scalar>);

    auto v {std::forward<A>(a).view()};
    using M = Eigen3::eigen_matrix_t<Scalar, MatrixTraits<A>::rows, MatrixTraits<B>::columns>;
    auto llt {v.llt()};

    M ret;
    if (llt.info() == Eigen::Success)
    {
      ret = Eigen::Solve {llt, std::forward<B>(b)};
    }
    else [[unlikely]]
    {
      // A is semidefinite. Use LDLT decomposition instead.
      auto ldlt {v.ldlt()};
      if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
      {
        throw (std::runtime_error("SelfAdjointMatrix solve: A is indefinite"));
      }
      ret = Eigen::Solve {ldlt, std::forward<B>(b)};
    }
    return ret;
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr A, eigen_matrix B>
#else
  template<typename A, typename B, std::enable_if_t<eigen_triangular_expr<A> and eigen_matrix<B>, int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    return make_self_contained<A, B>(Eigen::Solve {std::forward<A>(a).view(), std::forward<B>(b)});
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_rows<Arg>)
    {
      return ZeroMatrix<Scalar, 0, 1> {row_count(arg)};
    }
    else
    {
      constexpr auto rows = MatrixTraits<Arg>::rows;
      return ZeroMatrix<Scalar, rows, 1> {};
    }
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_rows<Arg>)
    {
      return ConstantMatrix<Scalar, constant, 0, 1> {row_count(arg)};
    }
    else
    {
      return ConstantMatrix<Scalar, constant, MatrixTraits<Arg>::rows, 1> {};
    }
  }


  /// Create a column vector from a diagonal matrix. (Same as nested_matrix()).
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg,
    std::enable_if_t<eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).rowwise().sum() / MatrixTraits<Arg>::rows);
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_columns<Arg>)
    {
      return ZeroMatrix<Scalar, 1, 0> {column_count(arg)};
    }
    else
    {
      return ZeroMatrix<Scalar, 1, MatrixTraits<Arg>::columns> {};
    }
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_columns<Arg>)
    {
      return ConstantMatrix<Scalar, constant, 1, 0> {column_count(arg)};
    }
    else
    {
      return ConstantMatrix<Scalar, constant, 1, MatrixTraits<Arg>::columns> {};
    }
  }


  /// Create a row vector from a diagonal matrix. (Same as nested_matrix()).
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg,
      std::enable_if_t<eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_rows(Arg&& arg)
  {
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).colwise().sum() / column_count(arg));
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    if constexpr (dynamic_rows<A>)
    {
      auto dim = row_count(a);
      return ZeroMatrix<Scalar, 0, 0> {dim, dim};
    }
    else
    {
      constexpr auto dim = MatrixTraits<A>::rows;
      return ZeroMatrix<Scalar, dim, dim> {};
    }
  }


/**
 * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
 * Returns L as a lower-triangular matrix.
 */
#ifdef __cpp_concepts
  template<eigen_constant_expr A>
#else
  template<typename A, std::enable_if_t<eigen_constant_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto constant = MatrixTraits<A>::constant;

    const Scalar elem = constant * (
      dynamic_columns<A> ?
      std::sqrt((Scalar) column_count(a)) :
      OpenKalman::internal::constexpr_sqrt((Scalar) MatrixTraits<A>::columns));

    if constexpr (dynamic_rows<A>)
    {
      auto dim = row_count(a);
      auto col1 = Eigen3::eigen_matrix_t<Scalar, 0, 1>::Constant(dim, elem);

      eigen_matrix_t<Scalar, 0, 0> ret {dim, dim};

      if (dim == 1)
        ret = std::move(col1);
      else
        ret = concatenate_horizontal(std::move(col1), ZeroMatrix<Scalar, 0, 0> {dim, dim - 1});
      return ret;
    }
    else
    {
      constexpr auto dim = MatrixTraits<A>::rows;
      auto col1 = Eigen3::eigen_matrix_t<Scalar, dim, 1>::Constant(elem);

      if constexpr (dim > 1)
        return concatenate_horizontal(col1, ZeroMatrix<Scalar, dim, dim - 1> {});
      else
        return col1;
    }
  }


  /**
   * \brief Perform an LQ decomposition of matrix A=[L,0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
   * \return L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<typename A> requires eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>
#else
  template<typename A, std::enable_if_t<
    eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  LQ_decomposition(A&& a)
  {
    if constexpr(lower_triangular_matrix<A>) return std::forward<A>(a);
    else return LQ_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    if constexpr (dynamic_columns<A>)
    {
      auto dim = column_count(a);
      return ZeroMatrix<Scalar, 0, 0> {dim, dim};
    }
    else
    {
      constexpr auto dim = MatrixTraits<A>::columns;
      return ZeroMatrix<Scalar, dim, dim> {};
    }
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr A>
#else
  template<typename A, std::enable_if_t<eigen_constant_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto constant = MatrixTraits<A>::constant;

    const Scalar elem = constant * (
      dynamic_rows<A> ?
      std::sqrt((Scalar) row_count(a)) :
      OpenKalman::internal::constexpr_sqrt((Scalar) MatrixTraits<A>::rows));

    if constexpr (dynamic_columns<A>)
    {
      auto dim = column_count(a);
      auto row1 = Eigen3::eigen_matrix_t<Scalar, 1, 0>::Constant(dim, elem);

      eigen_matrix_t<Scalar, 0, 0> ret {dim, dim};

      if (dim == 1)
      {
        ret = std::move(row1);
      }
      else
      {
        ret = concatenate_vertical(std::move(row1), ZeroMatrix<Scalar, 0, 0> {dim - 1, dim});
      }
      return ret;
    }
    else
    {
      constexpr auto dim = MatrixTraits<A>::columns;
      auto row1 = Eigen3::eigen_matrix_t<Scalar, 1, dim>::Constant(elem);
      if constexpr (dim > 1)
      {
        return concatenate_vertical(row1, ZeroMatrix<Scalar, dim - 1, dim> {});
      }
      else
      {
        return row1;
      }
    }
  }


  /**
   * \brief Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
   * \return U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<typename A> requires eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>
#else
  template<typename A, std::enable_if_t<
    eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  QR_decomposition(A&& a)
  {
    if constexpr(upper_triangular_matrix<A>) return std::forward<A>(a);
    else return QR_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr V, eigen_diagonal_expr ... Vs>
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (eigen_diagonal_expr<V> and ... and eigen_diagonal_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      return MatrixTraits<V>::make(
        concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  namespace detail
  {
#ifdef __cpp_concepts
    template<TriangleType t, eigen_self_adjoint_expr M>
#else
    template<TriangleType t, typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
    decltype(auto)
    maybe_transpose(M&& m)
    {
      if constexpr(t == MatrixTraits<M>::storage_triangle) return nested_matrix(std::forward<M>(m));
      else return transpose(nested_matrix(std::forward<M>(m)));
    }
  }


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr V, eigen_self_adjoint_expr ... Vs>
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      /// \todo Add diagonal case
      constexpr auto t = MatrixTraits<V>::storage_triangle;
      return MatrixTraits<V>::make(
        concatenate_diagonal(nested_matrix(std::forward<V>(v)), detail::maybe_transpose<t>(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


    /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<eigen_triangular_expr V, eigen_triangular_expr ... Vs>
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      /// \todo Add diagonal case
      if constexpr (((upper_triangular_matrix<V> == upper_triangular_matrix<Vs>) and ...))
      {
        return MatrixTraits<V>::make(
          concatenate_diagonal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
      }
      else // There is a mixture of upper and lower triangles.
      {
        return concatenate_diagonal(
          make_native_matrix(std::forward<V>(v)), make_native_matrix(std::forward<Vs>(vs))...);
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  namespace internal
  {
    template<typename G, typename Expr>
    struct SplitSpecF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make(std::forward<Arg>(arg)));
      }
    };
  }


  /// Split a diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, eigen_diagonal_expr Arg> requires (not coefficients<F>)
#else
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<eigen_diagonal_expr<Arg> and not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_vertical<internal::SplitSpecF<F, Arg>, Cs...>(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split a self-adjoint or triangular matrix diagonally.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and (not coefficients<F>)
#else
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<(eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_diagonal<internal::SplitSpecF<F, Arg>, Cs...>(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<(coefficients<Cs> and ...) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_diagonal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg>
  requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    return split_diagonal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix vertically, returning a regular matrix.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, typename Arg> requires (not coefficients<F>) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename F, typename ... Cs, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
      not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_vertical<internal::SplitSpecF<F, native_matrix_t<Arg>>, Cs...>(make_native_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<(coefficients<Cs> and ...) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_vertical<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg> requires
  eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    return split_vertical<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix horizontally, returning a regular matrix.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, typename Arg> requires (not coefficients<F>) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename F, typename ... Cs, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
      not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_horizontal<internal::SplitSpecF<F, native_matrix_t<Arg>>, Cs...>(make_native_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix horizontally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg> requires
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows);
    return split_horizontal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix horizontally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg> requires
  eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Get an element of a ZeroMatrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg& arg, const std::size_t row, const std::size_t col)
  {
    assert(row < row_count(arg));
    assert(col < column_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Get an element of a one-column ZeroMatrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg& arg, const std::size_t row)
  {
    assert(row < row_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Get an element of a ConstantMatrix. Always the constant.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg& arg, const std::size_t row, const std::size_t col)
  {
    assert(row < row_count(arg));
    assert(col < column_count(arg));
    return MatrixTraits<Arg>::constant;
  }


  /// Get an element of a one-column ConstantMatrix. Always the constant.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg& arg, const std::size_t row)
  {
    assert(row < row_count(arg));
    return MatrixTraits<Arg>::constant;
  }


  /// Get element (i) of diagonal matrix arg.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg> requires (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
    (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (element_gettable<nested_matrix_t<Arg>, 1>)
      return get_element(nested_matrix(std::forward<Arg>(arg)), i);
    else
      return get_element(nested_matrix(std::forward<Arg>(arg)), i, 1);
  }


  /// Get element (i, j) of diagonal matrix arg.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg> requires (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
    (element_gettable<nested_matrix_t<Arg>, 1> or
    element_gettable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr (element_gettable<nested_matrix_t<Arg>, 1>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, 1);
    }
    else
    {
      return typename MatrixTraits<Arg>::Scalar(0);
    }
  }


  /// Get element (i, j) of self-adjoint matrix arg.
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg> requires (not diagonal_matrix<Arg>) and
    element_gettable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and not diagonal_matrix<Arg> and
    element_gettable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    decltype(auto) n = nested_matrix(std::forward<Arg>(arg)); using N = decltype(n);

    if constexpr(lower_triangular_storage<Arg>)
    {
      if (i >= j) return get_element(std::forward<N>(n), i, j);
      else return get_element(std::forward<N>(n), j, i);
    }
    else
    {
      if (i <= j) return get_element(std::forward<N>(n), i, j);
      else return get_element(std::forward<N>(n), j, i);
    }
  }


  /// Get element (i, j) of triangular matrix arg.
#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg> requires (not diagonal_matrix<Arg>) and
    element_gettable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and not diagonal_matrix<Arg> and
    element_gettable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_matrix<Arg>)
    {
      if (i >= j) return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
      else return typename MatrixTraits<Arg>::Scalar(0);
    }
    else
    {
      if (i <= j) return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
      else return typename MatrixTraits<Arg>::Scalar(0);
    }
  }


  /// Get element (i, j) of a self-adjoint or triangular matrix that is also diagonal.
#ifdef __cpp_concepts
  template<typename Arg> requires
    diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_gettable<nested_matrix_t<Arg>, 2> or
      element_gettable<nested_matrix_t<Arg>, 1>)
#else
  template<typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and diagonal_matrix<Arg> and
    (element_gettable<nested_matrix_t<Arg>, 2> or
      element_gettable<nested_matrix_t<Arg>, 1>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr(element_gettable<nested_matrix_t<Arg>, 1>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
    }
    else return typename MatrixTraits<Arg>::Scalar(0);
  }


  /// Get element (i) of diagonal self-adjoint or triangular matrix.
#ifdef __cpp_concepts
  template<typename Arg> requires
    diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_gettable<nested_matrix_t<Arg>, 1> or
      element_gettable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, std::enable_if_t<
    diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_gettable<nested_matrix_t<Arg>, 1> or
      element_gettable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    using NestedMatrix = nested_matrix_t<Arg>;
    if constexpr(element_gettable<NestedMatrix, 1>)
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i);
    }
    else
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
    }
  }


  /// Set element (i) of matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (element_settable<nested_matrix_t<Arg>, 1> or element_settable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    eigen_diagonal_expr<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
    (element_settable<nested_matrix_t<Arg>, 1> or element_settable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    if constexpr (element_settable<nested_matrix_t<Arg>, 1>)
      set_element(nested_matrix(arg), s, i);
    else
      set_element(nested_matrix(arg), s, i, 1);
  }


  /// Set element (i, j) of matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (element_settable<nested_matrix_t<Arg>, 1> or element_settable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    eigen_diagonal_expr<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
    (element_settable<nested_matrix_t<Arg>, 1> or element_settable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr (element_settable<nested_matrix_t<Arg>, 1>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, 1);
    }
    else
      throw std::out_of_range("Only diagonal elements of a diagonal matrix may be set.");
  }


  /// Set element (i, j) of self-adjoint matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not diagonal_matrix<Arg>) and
    element_settable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and not diagonal_matrix<Arg> and
    element_settable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_storage<Arg>)
    {
      if (i >= j) set_element(nested_matrix(arg), s, i, j);
      else set_element(nested_matrix(arg), s, j, i);
    }
    else
    {
      if (i <= j) set_element(nested_matrix(arg), s, i, j);
      else set_element(nested_matrix(arg), s, j, i);
    }
  }


  /// Set element (i, j) of triangular matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not diagonal_matrix<Arg>) and
    element_settable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<eigen_triangular_expr<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and not diagonal_matrix<Arg> and
    element_settable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_matrix<Arg>)
    {
      if (i >= j) set_element(nested_matrix(arg), s, i, j);
      else throw std::out_of_range("Only lower-triangle elements of a lower-triangular matrix may be set.");
    }
    else
    {
      if (i <= j) set_element(nested_matrix(arg), s, i, j);
      else throw std::out_of_range("Only upper-triangle elements of an upper-triangular matrix may be set.");
    }
  }


  /// Set element (i, j) of a self-adjoint or triangular matrix that is also diagonal.
#ifdef __cpp_concepts
  template<typename Arg, typename Scalar> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and diagonal_matrix<Arg> and
    (element_settable<nested_matrix_t<Arg>, 2> or
      element_settable<nested_matrix_t<Arg>, 1>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    not std::is_const_v<std::remove_reference_t<Arg>> and diagonal_matrix<Arg> and
    (element_settable<nested_matrix_t<Arg>, 2> or
      element_settable<nested_matrix_t<Arg>, 1>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr(element_settable<nested_matrix_t<Arg>, 1>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, i);
    }
    else throw std::out_of_range("Only diagonal elements of a diagonal matrix may be set.");
  }


  /// Set element (i) of diagonal self-adjoint or triangular matrix.
#ifdef __cpp_concepts
  template<typename Arg, typename Scalar> requires diagonal_matrix<Arg> and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_settable<nested_matrix_t<Arg>, 1> or
      element_settable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<diagonal_matrix<Arg> and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_settable<nested_matrix_t<Arg>, 1> or
      element_settable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    using NestedMatrix = nested_matrix_t<Arg>;
    if constexpr(element_settable<NestedMatrix, 1>)
    {
      set_element(nested_matrix(arg), s, i);
    }
    else
    {
      set_element(nested_matrix(arg), s, i, i);
    }
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    assert(index < column_count(arg));
    return reduce_columns(std::forward<Arg>(arg));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_zero_expr Arg> requires (not dynamic_columns<Arg>) and
  (index < MatrixTraits<Arg>::columns)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_zero_expr<Arg> and (not dynamic_columns<Arg>) and (index < MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    if constexpr(column_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return reduce_columns(std::forward<Arg>(arg));
  }


/// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    assert(index < column_count(arg));
    return reduce_columns(std::forward<Arg>(arg));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_constant_expr Arg> requires (index < MatrixTraits<Arg>::columns)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_constant_expr<Arg> and (index < MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    return reduce_columns(std::forward<Arg>(arg));
  }


/// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    return column(make_native_matrix(std::forward<Arg>(arg)), index);
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, typename Arg> requires (eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < MatrixTraits<Arg>::columns)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<(eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    return column<index>(make_native_matrix(std::forward<Arg>(arg)));
  }


  /// Return row <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t index)
  {
    assert(index < row_count(arg));
    return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_zero_expr Arg> requires (not dynamic_rows<Arg>) and
    (index < MatrixTraits<Arg>::rows)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_zero_expr<Arg> and (not dynamic_rows<Arg>) and (index < MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    if constexpr(row_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t index)
  {
    assert(index < row_count(arg));
    return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_constant_expr Arg> requires (index < MatrixTraits<Arg>::rows)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
      eigen_constant_expr<Arg> and (index < MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
      eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t index)
  {
    return row(make_native_matrix(std::forward<Arg>(arg)), index);
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, typename Arg> requires (eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < MatrixTraits<Arg>::rows)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<(eigen_self_adjoint_expr<Arg> or
      eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    return row<index>(make_native_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<typename Arg, typename Function> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (requires(Arg&& arg, const Function& f) { {f(column(arg, 0))} -> column_vector; } or
      requires(Arg&& arg, const Function& f, std::size_t i) { {f(column(arg, 0), i)} -> column_vector; })
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    return apply_columnwise(make_native_matrix(std::forward<Arg>(arg)), f);
  }


#ifdef __cpp_concepts
  template<typename Arg, typename Function> requires
  (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (requires(Arg&& arg, const Function& f) { {f(row(arg, 0))} -> row_vector; } or
      requires(Arg&& arg, const Function& f, std::size_t i) { {f(row(arg, 0), i)} -> row_vector; })
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
      eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  apply_rowwise(Arg&& arg, const Function& f)
  {
    return apply_rowwise(make_native_matrix(std::forward<Arg>(arg)), f);
  }


#ifdef __cpp_concepts
  template<typename Arg, typename Function> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (requires(Function& f, typename MatrixTraits<Arg>::Scalar& s) {
      {f(s)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar>;
    } or
    requires(Function& f, typename MatrixTraits<Arg>::Scalar& s, std::size_t& i, std::size_t& j) {
      {f(s, i, j)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar>;
    })
#else
  template<typename Arg, typename Function, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    std::is_convertible_v<
      std::invoke_result_t<Function&, typename MatrixTraits<Arg>::Scalar&>,
      const typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    return apply_coefficientwise(make_native_matrix(std::forward<Arg>(arg)), f);
  }


  template<typename Arg, typename Function, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    std::is_convertible_v<
      std::invoke_result_t<Function&, typename MatrixTraits<Arg>::Scalar&, std::size_t&, std::size_t&>,
      const typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    return apply_coefficientwise(make_native_matrix(std::forward<Arg>(arg)), f);
  }


  /**
   * \brief Fill a fixed diagonal matrix with random values selected from one or more random distributions.
   * \details The following example constructs 2-by-2 diagonal matrices in which each diagonal element is
   * a random value selected as indicated:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
   *     D2 m = randomize<D2>(N {1.0, 0.3})); // Both diagonal elements have mean 1.0, s.d. 0.3
   *     D2 n = randomize<D2>(N {1.0, 0.3}, N {2.0, 0.2})); // Second diagonal element has mean 2.0, s.d. 0.2.
   *     D2 p = randomize<D2>(N {1.0, 0.3}, 2.0)); // Second diagonal element is exactly 2.0
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \tparam Dists A set of distributions (e.g., std::normal_distribution<double>) or, alternatively,
   * means (a definite, non-stochastic value).
   **/
#ifdef __cpp_concepts
  template<eigen_diagonal_expr ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename...Dists>
  requires (not dynamic_shape<ReturnType>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<eigen_diagonal_expr<ReturnType> and (not dynamic_shape<ReturnType>), int> = 0>
#endif
  inline auto
  randomize(Dists&&...dists)
  {
    using B = nested_matrix_t<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs matrices (m, and n) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using D0 = DiagonalMatrix<eigen_matrix_t<double, 0, 1>>;
   *     auto m = randomize(D0, 2, 2, std::normal_distribution<double> {1.0, 0.3})); // constructs a 2-by-2 matrix
   *     auto n = randomize(D0, 3, 3, std::normal_distribution<double> {1.0, 0.3}); // constructs a 3-by-2 matrix
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \param rows Number of rows, decided at runtime.
   * \param columns Number of columns, decided at runtime. Columns must equal rows.
   * \tparam Dist A distribution (type distribution_type).
   **/
#ifdef __cpp_concepts
  template<eigen_diagonal_expr ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename Dist>
  requires
    dynamic_shape<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
    eigen_diagonal_expr<ReturnType> and dynamic_shape<ReturnType> and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    assert(rows == columns);
    using B = nested_matrix_t<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(rows, 1, std::forward<Dist>(dist)));
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
