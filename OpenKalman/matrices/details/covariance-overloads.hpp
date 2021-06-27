/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEOVERLOADS_HPP
#define OPENKALMAN_COVARIANCEOVERLOADS_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline decltype(auto)
  nested_matrix(M&& m) noexcept
  {
    if constexpr (self_adjoint_matrix<nested_matrix_t<M>>)
    {
      return std::forward<M>(m).get_self_adjoint_nested_matrix();
    }
    else
    {
      return std::forward<M>(m).get_triangular_nested_matrix();
    }
  }


#ifdef __cpp_concepts
  template<covariance Arg> requires (not square_root_covariance<Arg>)
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and not square_root_covariance<Arg>, int> = 0>
#endif
  inline decltype(auto)
  square_root(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).square_root();
  }


#ifdef __cpp_concepts
  template<square_root_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<square_root_covariance<Arg>, int> = 0>
#endif
  inline decltype(auto)
  square(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).square();
  }


  /**
   * \brief Convert to a writable native matrix.
   * \details The resulting matrix is not expected to be self-adjoint, diagonal, or zero.
   */
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr auto
  make_native_matrix(Arg&& arg) noexcept
  {
    return make_native_matrix(OpenKalman::internal::to_covariance_nestable(std::forward<Arg>(arg)));
  }


  /**
   * \brief Convert to self-contained version of the covariance matrix.
   * \details If any types Ts are included, Arg is considered self-contained if every Ts is an lvalue reference.
   * This is to allow a function to return a non-self-contained value so long as it depends on an lvalue reference to
   * a variable that is accessible to the context in which the function is called.
   */
#ifdef __cpp_concepts
  template<typename...Ts, covariance Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
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
      using C = typename MatrixTraits<Arg>::RowCoefficients;
      if constexpr (square_root_covariance<Arg>)
      {
        return SquareRootCovariance<C, self_contained_t<nested_matrix_t<Arg>>> {std::forward<Arg>(arg)};
      }
      else
      {
        return Covariance<C, self_contained_t<nested_matrix_t<Arg>>> {std::forward<Arg>(arg)};
      }
    }
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr std::size_t row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return row_count(nested_matrix(std::forward<Arg>(arg)));
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr std::size_t column_count(Arg&& arg)
  {
    return row_count(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    auto b = make_self_contained<Arg>(
      diagonal_of(OpenKalman::internal::to_covariance_nestable(std::forward<Arg>(arg))));
    return Matrix<C, Axis, decltype(b)>(std::move(b));
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    /// \todo optimize this
    if constexpr (square_root_covariance<Arg>)
    {
      return MatrixTraits<Arg>::make(transpose(std::forward<Arg>(arg).get_triangular_nested_matrix()));
    }
    else
    {
      if constexpr (cholesky_form<Arg>)
      {
        return MatrixTraits<Arg>::make(transpose(std::forward<Arg>(arg).get_triangular_nested_matrix()));
      }
      else return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    /// \todo optimize this
    if constexpr (complex_number<typename MatrixTraits<Arg>::Scalar>)
    {
      return MatrixTraits<Arg>::make(adjoint(std::forward<Arg>(arg).get_triangular_nested_matrix()));
    }
    else
    {
      return transpose(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    auto d = determinant(nested_matrix(std::forward<Arg>(arg)));
    using ArgBase = nested_matrix_t<Arg>;
    if constexpr(triangular_matrix<ArgBase> and not self_adjoint_matrix<ArgBase> and not square_root_covariance<Arg>)
      return d * d;
    else if constexpr(not triangular_matrix<ArgBase> and self_adjoint_matrix<ArgBase> and square_root_covariance<Arg>)
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
    return trace(to_covariance_nestable(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typed_matrix U> requires
    equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>
#else
  template<typename Arg, typename U, std::enable_if_t<covariance<Arg> and typed_matrix<U> and
    equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>, int> = 0>
#endif
  inline decltype(auto)
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    if constexpr (std::is_same_v<Arg&&, std::decay_t<Arg>&>)
    {
      return arg.rank_update(u, alpha);
    }
    else
    {
      auto ret = std::forward<Arg>(arg).rank_update(u, alpha);
      return ret;
    }
  }


  /// Solves a x = b for x (A is a Covariance or SquareRootCovariance, B is a vector type).
#ifdef __cpp_concepts
  template<covariance A, typed_matrix B> requires
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<B>::RowCoefficients>
#else
  template<typename A, typename B, std::enable_if_t<covariance<A> and typed_matrix<B> and
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<B>::RowCoefficients>, int> = 0>
#endif
  inline auto
  solve(A&& a, B&& b) noexcept
  {
    auto x = make_self_contained<A, B>(
      solve(to_covariance_nestable(std::forward<A>(a)), nested_matrix(std::forward<B>(b))));
    return MatrixTraits<B>::template make<typename MatrixTraits<A>::RowCoefficients>(std::move(x));
  }


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    return make_matrix<RC, Axis>(reduce_columns(to_covariance_nestable(std::forward<Arg>(arg))));
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    auto tm = LQ_decomposition(to_covariance_nestable(std::forward<Arg>(arg)));
    return make_square_root_covariance<C>(std::move(tm));
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    auto tm = QR_decomposition(to_covariance_nestable(std::forward<Arg>(arg)));
    return make_square_root_covariance<C>(std::move(tm));
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
      using Coeffs =
        Concatenate<typename MatrixTraits<M>::RowCoefficients, typename MatrixTraits<Ms>::RowCoefficients...>;
      auto cat = concatenate_diagonal(nested_matrix(std::forward<M>(m)), nested_matrix(std::forward<Ms>(mN))...);
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
      if constexpr(one_by_one_matrix<Arg> and not square_root_covariance<Expr> and cholesky_form<Expr>)
      {
        return MatrixTraits<Expr>::template make<C>(Cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(one_by_one_matrix<Arg> and square_root_covariance<Expr> and not cholesky_form<Expr>)
      {
        return MatrixTraits<Expr>::template make<C>(Cholesky_factor<TriangleType::lower>(std::forward<Arg>(arg)));
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
    inline auto split_cov_diag_impl(const F& f, Arg&& arg)
    {
      if constexpr(one_by_one_matrix<Arg> and not square_root_covariance<Expr> and cholesky_form<Expr>)
      {
        return f(Cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(one_by_one_matrix<Arg> and square_root_covariance<Expr> and not cholesky_form<Expr>)
      {
        return f(Cholesky_factor<TriangleType::lower>(std::forward<Arg>(arg)));
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
        static_assert(equivalent_to<RC, CC>);
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
        auto f = [](auto&& m) { return make_matrix<RC, CC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename RC>
    struct SplitCovHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return make_matrix<RC, CC>(std::forward<decltype(m)>(m)); };
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
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<M>::RowCoefficients>);
    return split_diagonal<OpenKalman::internal::SplitCovDiagF<M>, Cs...>(nested_matrix(std::forward<M>(m)));
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
    using CC = typename MatrixTraits<M>::RowCoefficients;
    static_assert(prefix_of<Concatenate<Cs...>, CC>);
    return split_vertical<OpenKalman::internal::SplitCovVertF<M, CC>, Cs...>(make_native_matrix(std::forward<M>(m)));
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
    using RC = typename MatrixTraits<M>::RowCoefficients;
    static_assert(prefix_of<Concatenate<Cs...>, RC>);
    return split_horizontal<OpenKalman::internal::SplitCovHorizF<M, RC>, Cs...>(make_native_matrix(std::forward<M>(m)));
  }


  /// Get element (i, j) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg> requires element_gettable<Arg, 2>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and element_gettable<Arg, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    return std::forward<Arg>(arg)(i, j);
  }


  /// Get element (i) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg> requires element_gettable<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and element_gettable<Arg, 1>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    return std::forward<Arg>(arg)[i];
  }


  /// Set element (i, j) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and element_settable<Arg, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<covariance<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and element_settable<Arg, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    arg.set_element(s, i, j);
  }


  /// Set element (i) of a covariance matrix.
#ifdef __cpp_concepts
  template<covariance Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and element_settable<Arg, 1>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<covariance<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and element_settable<Arg, 1>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    arg.set_element(s, i);
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_matrix<C, Axis>(column(to_covariance_nestable(std::forward<Arg>(arg)), index));
  }


  /// Return column <code>index</code> of Arg, where the index is constexpr.
#ifdef __cpp_concepts
  template<std::size_t index, covariance Arg> requires (index < MatrixTraits<Arg>::rows)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    covariance<Arg> and (index < MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using CC = typename C::template Coefficient<index>;
    return make_matrix<C, CC>(column<index>(to_covariance_nestable(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typename Function> requires
    requires(Arg&& arg, const Function& f) {
      {f(column<0>(arg))} -> typed_matrix;
      MatrixTraits<decltype(f(column<0>(arg)))>::columns == 1;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>>>, int> = 0>
#endif
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f] (auto&& col) -> auto {
      return make_self_contained(nested_matrix(f(make_matrix<C, Axis>(std::forward<decltype(col)>(col)))));
    };
    return make_matrix<C, C>(apply_columnwise(to_covariance_nestable(std::forward<Arg>(arg)), f_nested));
  }


#ifdef __cpp_concepts
  template<covariance Arg, typename Function> requires
    requires(Arg&& arg, const Function& f, std::size_t i) {
      {f(column<0>(arg), i)} -> typed_matrix;
      MatrixTraits<decltype(f(column<0>(arg), i))>::columns == 1;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f] (auto&& col, std::size_t i) -> auto {
      return make_self_contained(nested_matrix(f(make_matrix<C, Axis>(std::forward<decltype(col)>(col)), i)));
    };
    return make_matrix<C, C>(apply_columnwise(to_covariance_nestable(std::forward<Arg>(arg)), f_nested));
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_matrix<C, C>(apply_coefficientwise(to_covariance_nestable(std::forward<Arg>(arg)), f));
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_matrix<C, C>(apply_coefficientwise(to_covariance_nestable(std::forward<Arg>(arg)), f));
  }


#ifdef __cpp_concepts
  template<covariance Cov>
#else
  template<typename Cov, std::enable_if_t<covariance<Cov>, int> = 0>
#endif
  inline std::ostream& operator<<(std::ostream& os, const Cov& c)
  {
    os << make_native_matrix(c);
    return os;
  }


}

#endif //OPENKALMAN_COVARIANCEOVERLOADS_HPP
