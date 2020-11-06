/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEOVERLOADS_H
#define OPENKALMAN_COVARIANCEOVERLOADS_H

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  constexpr decltype(auto)
  base_matrix(M&& m) noexcept
  {
    return std::forward<M>(m).base_matrix();
  }


#ifdef __cpp_concepts
  template<covariance Arg> requires (not square_root_covariance<Arg>)
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and not square_root_covariance<Arg>, int> = 0>
#endif
  inline auto
  square_root(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    if constexpr(is_diagonal_v<Arg> and not is_zero_v<Arg>)
    {
      return make_SquareRootCovariance<C>(Cholesky_factor(base_matrix(std::forward<Arg>(arg))));
    }
    else
    {
      return make_SquareRootCovariance<C>(base_matrix(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<square_root_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<square_root_covariance<Arg>, int> = 0>
#endif
  inline auto
  square(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    if constexpr(is_diagonal_v<Arg> and not is_zero_v<Arg>)
    {
      return make_Covariance<C>(Cholesky_square(base_matrix(std::forward<Arg>(arg))));
    }
    else
    {
      return make_Covariance<C>(base_matrix(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<covariance Arg> requires (not is_Cholesky_v<Arg>) and (not is_diagonal_v<Arg>)
#else
  template<typename Arg,
    std::enable_if_t<covariance<Arg> and not is_Cholesky_v<Arg> and not is_diagonal_v<Arg>, int> = 0>
#endif
  inline auto
  to_Cholesky(Arg&& arg) noexcept
  {
    return MatrixTraits<Arg>::make(Cholesky_factor(base_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<covariance Arg> requires is_Cholesky_v<Arg> and (not is_diagonal_v<Arg>)
#else
template<typename Arg,
    std::enable_if_t<covariance<Arg> and is_Cholesky_v<Arg> and not is_diagonal_v<Arg>, int> = 0>
#endif
  inline auto
  from_Cholesky(Arg&& arg) noexcept
  {
    return MatrixTraits<Arg>::make(Cholesky_square(base_matrix(std::forward<Arg>(arg))));
  }


  /// Convert to strict regular matrix.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  strict_matrix(Arg&& arg) noexcept
  {
    return strict_matrix(internal::convert_base_matrix(std::forward<Arg>(arg)));
  }


  /// Convert to strict version of the covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  strict(Arg&& arg) noexcept
  {
    if constexpr(is_strict_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(strict(base_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  transpose(Arg&& arg) noexcept
  {
    return MatrixTraits<Arg>::make(transpose(base_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  adjoint(Arg&& arg) noexcept
  {
    return MatrixTraits<Arg>::make(adjoint(base_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    auto d = determinant(base_matrix(std::forward<Arg>(arg)));
    using ArgBase = typename MatrixTraits<Arg>::BaseMatrix;
    if constexpr(is_triangular_v<ArgBase> and not is_self_adjoint_v<ArgBase> and not square_root_covariance<Arg>)
      return d * d;
    else if constexpr(not is_triangular_v<ArgBase> and is_self_adjoint_v<ArgBase> and square_root_covariance<Arg>)
      return std::sqrt(d);
    else
      return d;
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    return trace(convert_base_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typed_matrix U> requires (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<covariance<Arg> and typed_matrix<U> and
    not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, typename MatrixTraits<U>::RowCoefficients>);
    rank_update(base_matrix(arg), base_matrix(u), alpha);
    return arg;
  }


#ifdef __cpp_concepts
  template<covariance Arg, typed_matrix U>
#else
  template<typename Arg, typename U, std::enable_if_t<covariance<Arg> and typed_matrix<U>, int> = 0>
#endif
  inline auto
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, typename MatrixTraits<U>::RowCoefficients>);
    return MatrixTraits<Arg>::make(rank_update(base_matrix(std::forward<Arg>(arg)), base_matrix(u), alpha));
  }


  /// Solves a x = b for x (A is a Covariance or SquareRootCovariance, B is a vector type).
#ifdef __cpp_concepts
  template<covariance A, typed_matrix B>
#else
  template<typename A, typename B, std::enable_if_t<covariance<A> and typed_matrix<B>, int> = 0>
#endif
  inline auto
  solve(A&& a, B&& b) noexcept
  {
    static_assert(is_equivalent_v<typename MatrixTraits<A>::Coefficients, typename MatrixTraits<B>::RowCoefficients>);
    auto x = strict(solve(convert_base_matrix(std::forward<A>(a)), base_matrix(std::forward<B>(b))));
    return MatrixTraits<B>::template make<typename MatrixTraits<A>::Coefficients>(std::move(x));
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    using RC = typename MatrixTraits<Arg>::Coefficients;
    return make_Matrix<RC, Axis>(reduce_columns(convert_base_matrix(std::forward<Arg>(arg))));
  }


  /// Perform an LQ decomposition of matrix A=[L,0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
  /// Returns L as a lower-triangular matrix.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  LQ_decomposition(Arg&& arg)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    auto tm = LQ_decomposition(convert_base_matrix(std::forward<Arg>(arg)));
    return make_SquareRootCovariance<C>(std::move(tm));
  }


  /// Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns L as an upper-triangular matrix.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  QR_decomposition(Arg&& arg)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    auto tm = QR_decomposition(convert_base_matrix(std::forward<Arg>(arg)));
    return make_SquareRootCovariance<C>(std::move(tm));
  }


  /// Concatenate one or more Covariance or SquareRootCovariance objects diagonally.
#ifdef __cpp_concepts
  template<covariance M, covariance ... Ms>
#else
  template<typename M, typename ... Ms, std::enable_if_t<(covariance<M> and ... and covariance<Ms>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(M&& m, Ms&& ... mN) noexcept
  {
    if constexpr(sizeof...(Ms) > 0)
    {
      using Coeffs = Concatenate<typename MatrixTraits<M>::Coefficients, typename MatrixTraits<Ms>::Coefficients...>;
      auto cat = concatenate_diagonal(base_matrix(std::forward<M>(m)), base_matrix(std::forward<Ms>(mN))...);
      return MatrixTraits<M>::template make<Coeffs>(std::move(cat));
    }
    else
    {
      return std::forward<M>(m);
    }
  }


  namespace detail
  {
    template<typename C, typename Expr, typename Arg>
    inline auto
    split_item_impl(Arg&& arg)
    {
      if constexpr(is_1by1_v<Arg> and not square_root_covariance<Expr> and is_Cholesky_v<Expr>)
      {
        return MatrixTraits<Expr>::template make<C>(Cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(is_1by1_v<Arg> and square_root_covariance<Expr> and not is_Cholesky_v<Expr>)
      {
        return MatrixTraits<Expr>::template make<C>(Cholesky_factor(std::forward<Arg>(arg)));
      }
      else
      {
        return MatrixTraits<Expr>::template make<C>(std::forward<Arg>(arg));
      }
    }
  }

  namespace internal
  {
    template<typename Expr, typename F, typename Arg>
    static auto split_cov_diag_impl(const F& f, Arg&& arg)
    {
      if constexpr(is_1by1_v<Arg> and not square_root_covariance<Expr> and is_Cholesky_v<Expr>)
      {
        return f(Cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(is_1by1_v<Arg> and square_root_covariance<Expr> and not is_Cholesky_v<Expr>)
      {
        return f(Cholesky_factor(std::forward<Arg>(arg)));
      }
      else
      {
        return f(std::forward<Arg>(arg));
      }
    }

    template<typename Expr>
    struct SplitCovDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        static_assert(is_equivalent_v<RC, CC>);
        auto f = [](auto&& m) { return MatrixTraits<Expr>::template make<RC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename CC>
    struct SplitCovVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return make_Matrix<RC, CC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename RC>
    struct SplitCovHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return make_Matrix<RC, CC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };
  }

  /// Split Covariance or SquareRootCovariance diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_diagonal(M&& m) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<M>::Coefficients>);
    return split_diagonal<internal::SplitCovDiagF<M>, Cs...>(base_matrix(std::forward<M>(m)));
  }


  /// Split Covariance or SquareRootCovariance vertically. Result is a tuple of typed matrices.
#ifdef __cpp_concepts
  template<coefficients ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_vertical(M&& m) noexcept
  {
    using CC = typename MatrixTraits<M>::Coefficients;
    static_assert(is_prefix_v<Concatenate<Cs...>, CC>);
    return split_vertical<internal::SplitCovVertF<M, CC>, Cs...>(strict_matrix(std::forward<M>(m)));
  }


  /// Split Covariance or SquareRootCovariance vertically. Result is a tuple of typed matrices.
#ifdef __cpp_concepts
  template<coefficients ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_horizontal(M&& m) noexcept
  {
    using RC = typename MatrixTraits<M>::Coefficients;
    static_assert(is_prefix_v<Concatenate<Cs...>, RC>);
    return split_horizontal<internal::SplitCovHorizF<M, RC>, Cs...>(strict_matrix(std::forward<M>(m)));
  }


  /// Get element (i, j) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg> requires is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    return std::forward<Arg>(arg)(i, j);
  }


  /// Get element (i) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg> requires
    ((is_self_adjoint_v<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
      (is_triangular_v<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and
    ((is_self_adjoint_v<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
      (is_triangular_v<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    return std::forward<Arg>(arg)[i];
  }


  /// Set element (i, j) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    covariance<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    arg(i, j) = s;
  }


  /// Set element (i) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
      ((is_self_adjoint_v<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (is_triangular_v<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    covariance<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      ((is_self_adjoint_v<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (is_triangular_v<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    arg[i] = s;
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_Matrix<C, Axis>(column(convert_base_matrix(std::forward<Arg>(arg)), index));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, covariance Arg>
#else
  template<std::size_t index, typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    static_assert(index < MatrixTraits<Arg>::dimension);
    using C = typename MatrixTraits<Arg>::Coefficients;
    using CC = typename C::template Coefficient<index>;
    return make_Matrix<C, CC>(column<index>(convert_base_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typename Function> requires
    typed_matrix<std::invoke_result_t<Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>>>
#else
  template<typename Arg, typename Function, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>>>, int> = 0>
#endif
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    const auto f_base = [&f](const auto& col) { return base_matrix(f(make_Matrix<C, Axis>(col))); };
    return make_Matrix<C, C>(apply_columnwise(convert_base_matrix(std::forward<Arg>(arg)), f_base));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typename Function> requires
    typed_matrix<std::invoke_result_t<Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>, std::size_t>>
#else
  template<typename Arg, typename Function, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    const auto f_base = [&f](const auto& col, std::size_t i) { return base_matrix(f(make_Matrix<C, Axis>(col), i)); };
    return make_Matrix<C, C>(apply_columnwise(convert_base_matrix(std::forward<Arg>(arg)), f_base));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typename Function> requires std::convertible_to<
    std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar>, const typename MatrixTraits<Arg>::Scalar>
#else
  template<typename Arg, typename Function, std::enable_if_t<covariance<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar>,
      const typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_Matrix<C, C>(apply_coefficientwise(convert_base_matrix(std::forward<Arg>(arg)), f));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typename Function> requires std::convertible_to<
    std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar, std::size_t, std::size_t>,
      const typename MatrixTraits<Arg>::Scalar>
#else
  template<typename Arg, typename Function, std::enable_if_t<covariance<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar, std::size_t, std::size_t>,
    const typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_Matrix<C, C>(apply_coefficientwise(convert_base_matrix(std::forward<Arg>(arg)), f));
  }


#ifdef __cpp_concepts
  template<covariance Cov>
#else
  template<typename Cov, std::enable_if_t<covariance<Cov>, int> = 0>
#endif
  inline std::ostream& operator<<(std::ostream& os, const Cov& c)
  {
    os << strict_matrix(c);
    return os;
  }


  /**********************
   * Arithmetic Operators
   **********************/

  /// Add two covariance types or one covariance type and one compatible typed matrix.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (covariance<Arg1> and covariance<Arg2>) or
    (covariance<Arg1> and typed_matrix<Arg2>) or
    (typed_matrix<Arg1> and covariance<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (covariance<Arg1> and covariance<Arg2>) or
    (covariance<Arg1> and typed_matrix<Arg2>) or
    (typed_matrix<Arg1> and covariance<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    using Cov = std::conditional_t<covariance<Arg1>, Arg1, Arg2>;
    using Other = std::conditional_t<covariance<Arg1>, Arg2, Arg1>;
    using C = typename MatrixTraits<Cov>::Coefficients;

    if constexpr(typed_matrix<Other>)
    {
      static_assert(
        is_equivalent_v<C, typename MatrixTraits<Other>::RowCoefficients> and
        is_equivalent_v<C, typename MatrixTraits<Other>::ColumnCoefficients>);
    }
    else
    {
      static_assert(is_equivalent_v<C, typename MatrixTraits<Arg2>::Coefficients>);
    }

    if constexpr(is_zero_v<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr(is_zero_v<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(is_Cholesky_v<Arg1> and is_Cholesky_v<Arg2> and
      not square_root_covariance<Arg1> and not square_root_covariance<Arg2>)
    {
      decltype(auto) E1 = base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) E2 = base_matrix(std::forward<Arg2>(arg2));
      if constexpr(is_upper_triangular_v<decltype(E1)> and is_upper_triangular_v<decltype(E2)>)
      {
        return make_Covariance<C>(strict(QR_decomposition(concatenate_vertical(E1, E2))));
      }
      else if constexpr(is_upper_triangular_v<decltype(E1)> and is_lower_triangular_v<decltype(E2)>)
      {
        return make_Covariance<C>(strict(QR_decomposition(concatenate_vertical(E1, adjoint(E2)))));
      }
      else if constexpr(is_lower_triangular_v<decltype(E1)> and is_upper_triangular_v<decltype(E2)>)
      {
        return make_Covariance<C>(strict(LQ_decomposition(concatenate_horizontal(E1, adjoint(E2)))));
      }
      else
      {
        return make_Covariance<C>(strict(LQ_decomposition(concatenate_horizontal(E1, E2))));
      }
    }
    else
    {
      decltype(auto) b1 = internal::convert_base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::convert_base_matrix(std::forward<Arg2>(arg2));
      constexpr auto conversion = not std::is_lvalue_reference_v<decltype(b1)> or not std::is_lvalue_reference_v<decltype(b2)>;

      const auto sum = [&b1, &b2] { if constexpr(conversion) return strict(b1 + b2); else return b1 + b2; }();
      if constexpr(is_self_adjoint_v<decltype(sum)>)
      {
        return make_Covariance<C>(sum);
      }
      else if constexpr(is_triangular_v<decltype(sum)>)
      {
        return make_SquareRootCovariance<C>(sum);
      }
      else
      {
        return make_Matrix<C, C>(sum);
      }
    }
  }


  /// Subtract two covariance types, or one covariance type and one compatible typed matrix.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
  (covariance<Arg1> and covariance<Arg2>) or
    (covariance<Arg1> and typed_matrix<Arg2>) or
    (typed_matrix<Arg1> and covariance<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (covariance<Arg1> and covariance<Arg2>) or
    (covariance<Arg1> and typed_matrix<Arg2>) or
    (typed_matrix<Arg1> and covariance<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    using Cov = std::conditional_t<covariance<Arg1>, Arg1, Arg2>;
    using Other = std::conditional_t<covariance<Arg1>, Arg2, Arg1>;
    using C = typename MatrixTraits<Cov>::Coefficients;

    if constexpr(typed_matrix<Other>)
    {
      static_assert(
        is_equivalent_v<C, typename MatrixTraits<Other>::RowCoefficients> and
          is_equivalent_v<C, typename MatrixTraits<Other>::ColumnCoefficients>);
    }
    else
    {
      static_assert(is_equivalent_v<C, typename MatrixTraits<Arg2>::Coefficients>);
    }

    if constexpr(is_zero_v<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(is_Cholesky_v<Arg1> and is_Cholesky_v<Arg2> and
      not square_root_covariance<Arg1> and not square_root_covariance<Arg2>)
    {
      using Scalar = typename MatrixTraits<Arg1>::Scalar;
      using B = typename MatrixTraits<Arg2>::BaseMatrix;

      decltype(auto) a = base_matrix(std::forward<Arg1>(arg1));
      const auto b = is_upper_triangular_v<B> ?
        strict_matrix(adjoint(base_matrix(std::forward<Arg2>(arg2)))) :
        strict_matrix(base_matrix(std::forward<Arg2>(arg2)));

      return make_Covariance<C>(strict(rank_update(a, b, Scalar(-1))));
    }
    else
    {
      decltype(auto) b1 = internal::convert_base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::convert_base_matrix(std::forward<Arg2>(arg2));
      constexpr auto conversion = not std::is_lvalue_reference_v<decltype(b1)> or not std::is_lvalue_reference_v<decltype(b2)>;

      const auto diff = [&b1, &b2] { if constexpr(conversion) return strict(b1 - b2); else return b1 - b2; }();
      if constexpr(is_self_adjoint_v<decltype(diff)>)
      {
        return make_Covariance<C>(diff);
      }
      else if constexpr(is_triangular_v<decltype(diff)>)
      {
        return make_SquareRootCovariance<C>(diff);
      }
      else
      {
        return make_Matrix<C, C>(diff);
      }
    }
  }


  /// Multiply two covariance types.
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    covariance<Arg1> and covariance<Arg2>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2) noexcept
  {
    using C = typename MatrixTraits<Arg1>::Coefficients;
    static_assert(is_equivalent_v<C, typename MatrixTraits<Arg2>::Coefficients>);

    if constexpr(is_zero_v<Arg1> or is_zero_v<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(is_identity_v<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr(is_identity_v<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      decltype(auto) b1 = internal::convert_base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::convert_base_matrix(std::forward<Arg2>(arg2));
      constexpr auto conversion = not std::is_lvalue_reference_v<decltype(b1)> or not std::is_lvalue_reference_v<decltype(b2)>;

      const auto prod = [&b1, &b2] { if constexpr(conversion) return strict(b1 * b2); else return b1 * b2; }();
      if constexpr(is_self_adjoint_v<decltype(prod)>)
      {
        return make_Covariance<C>(prod);
      }
      else if constexpr(is_triangular_v<decltype(prod)>)
      {
        return make_SquareRootCovariance<C>(prod);
      }
      else
      {
        return make_Matrix<C, C>(prod);
      }
    }
  }


  /// Multiply a typed matrix by a compatible covariance.
#ifdef __cpp_concepts
  template<typed_matrix M, covariance Cov>
#else
  template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and covariance<Cov>, int> = 0>
#endif
  constexpr decltype(auto) operator*(M&& m, Cov&& cov) noexcept
  {
    using CC = typename MatrixTraits<Cov>::Coefficients;
    static_assert(is_equivalent_v<typename MatrixTraits<M>::ColumnCoefficients, CC>);
    using RC = typename MatrixTraits<M>::RowCoefficients;
    using Mat = strict_matrix_t<M, RC::size, CC::size>;

    if constexpr(is_zero_v<M> or is_zero_v<Cov>)
    {
      return make_Matrix<RC, CC>(MatrixTraits<Mat>::zero());
    }
    else if constexpr(is_identity_v<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr(is_identity_v<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr(is_identity_v<typename MatrixTraits<M>::BaseMatrix>)
    {
      return make_Matrix<RC, CC>(internal::convert_base_matrix(std::forward<Cov>(cov)));
    }
    else
    {
      decltype(auto) mb = base_matrix(std::forward<M>(m));
      decltype(auto) cb = internal::convert_base_matrix(std::forward<Cov>(cov));

      if constexpr(not std::is_lvalue_reference_v<decltype(cb)>)
      {
        return make_Matrix<RC, CC>(strict(mb * cb));
      }
      else
      {
        return make_Matrix<RC, CC>(mb * cb);
      }
    }
  }


  /// Multiply a covariance type by a typed matrix. If the typed matrix is a mean, the result is wrapped.
#ifdef __cpp_concepts
  template<covariance Cov, typed_matrix M>
#else
  template<typename Cov, typename M, std::enable_if_t<covariance<Cov> and typed_matrix<M>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Cov&& cov, M&& m) noexcept
  {
    using RC = typename MatrixTraits<Cov>::Coefficients;
    static_assert(is_equivalent_v<RC, typename MatrixTraits<M>::RowCoefficients>);
    using CC = typename MatrixTraits<M>::ColumnCoefficients;
    using Mat = strict_matrix_t<M, RC::size, CC::size>;

    if constexpr(is_zero_v<Cov> or is_zero_v<M>)
    {
      return make_Matrix<RC, CC>(MatrixTraits<Mat>::zero());
    }
    else if constexpr(is_identity_v<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr(is_identity_v<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr(is_identity_v<typename MatrixTraits<M>::BaseMatrix>)
    {
      return make_Matrix<RC, CC>(internal::convert_base_matrix(std::forward<Cov>(cov)));
    }
    else
    {
      decltype(auto) cb = internal::convert_base_matrix(std::forward<Cov>(cov));
      decltype(auto) mb = base_matrix(std::forward<M>(m));

      if constexpr(not std::is_lvalue_reference_v<decltype(cb)>)
        return make_Matrix<RC, CC>(strict(cb * mb));
      else
        return make_Matrix<RC, CC>(cb * mb);
    }
  }


  /// Multiply a covariance type by a scalar.
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  constexpr auto operator*(M&& m, const S s) noexcept
  {
    using Scalar = const typename MatrixTraits<M>::Scalar;
    if constexpr(is_Cholesky_v<M>)
    {
      if constexpr(square_root_covariance<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * static_cast<Scalar>(s));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
      else
      {
        auto b = strict(base_matrix(std::forward<M>(m)));
        if (s > Scalar(0))
        {
          b *= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < Scalar(0))
        {
          const auto u = strict_matrix(b);
          b = MatrixTraits<decltype(b)>::zero();
          rank_update(b, u, static_cast<Scalar>(s));
        }
        else
        {
          b = MatrixTraits<decltype(b)>::zero();
        }
        return MatrixTraits<M>::make(std::move(b));
      }
    }
    else
    {
      if constexpr(is_zero_v<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else if constexpr(square_root_covariance<M> and not is_diagonal_v<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * (static_cast<Scalar>(s) * static_cast<Scalar>(s)));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
      else
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * static_cast<Scalar>(s));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
    }
  }


  /// Multiply a scalar by a self-adjoint-type covariance type.
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename S, typename M, std::enable_if_t<
    std::is_convertible_v<S, typename MatrixTraits<M>::Scalar> and covariance<M>, int> = 0>
#endif
  constexpr auto operator*(const S s, M&& m) noexcept
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    auto ret = std::forward<M>(m) * static_cast<Scalar>(s);
    if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
  }


  /// Divide a self-adjoint-type covariance type by a scalar.
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  constexpr auto operator/(M&& m, const S s)
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    if constexpr(is_Cholesky_v<M>)
    {
      if constexpr(square_root_covariance<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / static_cast<Scalar>(s));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
      else
      {
        auto b = strict(base_matrix(std::forward<M>(m)));
        if (s > Scalar(0))
        {
          b /= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < Scalar(0))
        {
          const auto u = strict_matrix(b);
          b = MatrixTraits<decltype(b)>::zero();
          rank_update(b, u, 1 / static_cast<Scalar>(s));
        }
        else
        {
          throw (std::runtime_error("operator/(Covariance, Scalar): divide by zero"));
        }
        return MatrixTraits<M>::make(std::move(b));
      }
    }
    else
    {
      if constexpr(is_zero_v<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else if constexpr(square_root_covariance<M> and not is_diagonal_v<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / (static_cast<Scalar>(s) * static_cast<Scalar>(s)));
        if constexpr (not std::is_lvalue_reference_v<M&>) return strict(std::move(ret)); else return ret;
      }
      else
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / static_cast<Scalar>(s));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
    }
  }


  /// Negate a covariance.
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  constexpr auto operator-(M&& m) noexcept
  {
    static_assert(not is_Cholesky_v<M> or square_root_covariance<M>,
      "Cannot negate a Cholesky-based Covariance because the square root would be complex.");
    if constexpr(is_Cholesky_v<M>)
    {
      if constexpr(square_root_covariance<M>)
      {
        auto ret = MatrixTraits<M>::make(-base_matrix(std::forward<M>(m)));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
      else
      {
        auto res = strict(std::forward<M>(m));
        res *= MatrixTraits<M>::Scalar(-1);
        return res;
      }
    }
    else
    {
      if constexpr(is_zero_v<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else
      {
        static_assert(not square_root_covariance<M> or is_diagonal_v<M>,
          "With real numbers, it is impossible to represent the negation of a non-diagonal, non-Cholesky-form "
          "square-root covariance.");
        auto ret = MatrixTraits<M>::make(-base_matrix(std::forward<M>(m)));
        if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
      }
    }
  }


  /// Equality operator.
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2>, int> = 0>
#endif
  constexpr bool operator==(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr(std::is_same_v<strict_matrix_t<Arg1>, strict_matrix_t<Arg2>> and
      is_equivalent_v<typename MatrixTraits<Arg1>::Coefficients, typename MatrixTraits<Arg2>::Coefficients>)
    {
      return strict_matrix(arg1) == strict_matrix(arg2);
    }
    else
    {
      return false;
    }
  }


  /// Inequality operator.
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2>, int> = 0>
#endif
  constexpr bool operator!=(const Arg1& arg1, const Arg2& arg2)
  {
    return not (arg1 == arg2);
  }


  /// Scale a covariance by a factor. Equivalent to multiplication by the square of a scalar.
  /// For a square root covariance, this is equivalent to multiplication by the scalar.
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  inline auto
  scale(M&& m, const S s)
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    if constexpr(is_Cholesky_v<M> or (is_diagonal_v<M> and square_root_covariance<M>))
      return MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * s);
    else
      return MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * (static_cast<Scalar>(s) * s));
  }


  /// Scale a covariance by the inverse of a scalar factor. Equivalent by division by the square of a scalar.
  /// For a square root covariance, this is equivalent to division by the scalar.
#ifdef __cpp_concepts
    template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  inline auto
  inverse_scale(M&& m, const S s)
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    if constexpr(is_Cholesky_v<M> or (is_diagonal_v<M> and square_root_covariance<M>))
    {
      auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / s);
      if constexpr (not std::is_lvalue_reference_v<M&>) return strict(std::move(ret)); else return ret;
    }
    else
    {
      auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / (static_cast<Scalar>(s) * s));
      if constexpr (not std::is_lvalue_reference_v<M&&>) return strict(std::move(ret)); else return ret;
    }
  }


  /// Scale a covariance by a matrix.
  /// A scaled covariance Arg is A * Arg * adjoint(A).
  /// A scaled square root covariance L or U is also scaled accordingly, so that
  /// scale(L * adjoint(L)) = A * L * adjoint(L) * adjoint(A) or
  /// scale(adjoint(U) * U) = A * adjoint(U) * U * adjoint(A).
#ifdef __cpp_concepts
  template<covariance M, typed_matrix A>
#else
  template<typename M, typename A, std::enable_if_t<covariance<M> and typed_matrix<A>, int> = 0>
#endif
  inline auto
  scale(M&& m, A&& a)
  {
    using C = typename MatrixTraits<M>::Coefficients;
    using AC = typename MatrixTraits<A>::RowCoefficients;
    static_assert(is_equivalent_v<typename MatrixTraits<A>::ColumnCoefficients, C>);
    static_assert(not euclidean_transformed<A>);
    using BaseMatrix = typename MatrixTraits<M>::BaseMatrix;

    decltype(auto) mbase = base_matrix(std::forward<M>(m));
    decltype(auto) abase = base_matrix(std::forward<A>(a));

    if constexpr(is_diagonal_v<BaseMatrix>)
    {
      using SABaseType = typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<TriangleType::lower>;
      if constexpr(square_root_covariance<M>)
      {
        auto b = MatrixTraits<SABaseType>::make(strict(abase * (Cholesky_square(mbase) * adjoint(abase))));
        return make_SquareRootCovariance<AC>(std::move(b));
      }
      else
      {
        auto b = MatrixTraits<SABaseType>::make(strict(abase * (mbase * adjoint(abase))));
        return make_Covariance<AC>(std::move(b));
      }
    }
    else if constexpr(is_self_adjoint_v<BaseMatrix>)
    {
      using SABaseType = typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<>;
      auto b = strict(MatrixTraits<SABaseType>::make(abase * (mbase * adjoint(abase))));
      return MatrixTraits<M>::template make<AC>(std::move(b));
    }
    else if constexpr(is_upper_triangular_v<BaseMatrix>)
    {
      const auto b = mbase * adjoint(abase);
      return MatrixTraits<M>::template make<AC>(strict(QR_decomposition(std::move(b))));
    }
    else
    {
      const auto b = abase * base_matrix(std::forward<M>(m));
      return MatrixTraits<M>::template make<AC>(strict(LQ_decomposition(std::move(b))));
    }
  }


}

#endif //OPENKALMAN_COVARIANCEOVERLOADS_H
