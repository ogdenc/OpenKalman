/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPED_MATRIX_OVERLOADS_H
#define OPENKALMAN_TYPED_MATRIX_OVERLOADS_H


namespace OpenKalman
{
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg) noexcept { return std::forward<Arg>(arg).nested_matrix(); }


/**
 * Convert to a self-contained Eigen3 matrix (wrapping any angles, if necessary).
 */
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_native_matrix(Arg&& arg) noexcept
  {
    return make_native_matrix(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Convert vector object to self-contained version (wrapping any angles).
#ifdef __cpp_concepts
  template<typename...Ts, typed_matrix Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<nested_matrix_t<Arg>> or
      (std::is_lvalue_reference_v<Ts> and ... and (sizeof...(Ts) > 0)))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(make_self_contained(nested_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_Euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(not euclidean_transformed<Arg>)
    {
      auto e = OpenKalman::to_Euclidean<Coefficients>(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<Coefficients, std::decay_t<decltype(e)>>(std::move(e));
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_Euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(euclidean_transformed<Arg>)
    {
      auto e = OpenKalman::from_Euclidean<Coefficients>(nested_matrix(std::forward<Arg>(arg)));
      return Mean<Coefficients, std::decay_t<decltype(e)>>(std::move(e));
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
    equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, Axis>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
    equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, Axis>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    auto b = make_self_contained(to_diagonal(nested_matrix(std::forward<Arg>(arg))));
    return Matrix<C, C, decltype(b)>(std::move(b));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (not euclidean_transformed<Arg>)
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>), int> = 0>
#endif
  inline auto
  transpose(Arg&& arg) noexcept
  {
    using CRows = typename MatrixTraits<Arg>::RowCoefficients;
    using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_Matrix<CCols, CRows>(transpose(nested_matrix(from_Euclidean(std::forward<Arg>(arg)))));
    else
      return make_Matrix<CCols, CRows>(transpose(nested_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (not euclidean_transformed<Arg>)
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>), int> = 0>
#endif
  inline auto
  adjoint(Arg&& arg) noexcept
  {
    using CRows = typename MatrixTraits<Arg>::RowCoefficients;
    using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_Matrix<CCols, CRows>(adjoint(nested_matrix(from_Euclidean(std::forward<Arg>(arg)))));
    else
      return make_Matrix<CCols, CRows>(adjoint(nested_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    return determinant(nested_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    return trace(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Solves AX = B for X, where X and B are means of the same type, and A is a square matrix with compatible types.
  /// If wrapping occurs, it will be both before for B and after for the X result.
#ifdef __cpp_concepts
  template<typed_matrix A, typed_matrix B> requires
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<B>::RowCoefficients> and
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<A>::ColumnCoefficients>
#else
  template<typename A, typename B, std::enable_if_t<typed_matrix<A> and typed_matrix<B> and
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<B>::RowCoefficients> and
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<A>::ColumnCoefficients>, int> = 0>
#endif
  inline auto
  solve(A&& a, B&& b) noexcept
  {
    auto x = solve(nested_matrix(std::forward<A>(a)), nested_matrix(std::forward<B>(b)));
    return MatrixTraits<B>::make(std::move(x));
  }


  /// Returns the mean of the column vectors after they are transformed into Euclidean space.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (MatrixTraits<Arg>::columns == 1) or column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    ((MatrixTraits<Arg>::columns == 1) or column_vector<Arg>), int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::columns == 1)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr(euclidean_transformed<Arg>)
      {
        return MatrixTraits<Arg>::make(reduce_columns(nested_matrix(std::forward<Arg>(arg))));
      }
      else if constexpr(mean<Arg>)
      {
        using C = typename MatrixTraits<Arg>::RowCoefficients;
        auto ev = reduce_columns(nested_matrix(to_Euclidean(std::forward<Arg>(arg))));
        return MatrixTraits<Arg>::make(from_Euclidean<C>(std::move(ev)));
      }
      else
      {
        using C = typename MatrixTraits<Arg>::RowCoefficients;
        return make_Matrix<C, Axis>(reduce_columns(nested_matrix(std::forward<Arg>(arg))));
      }
    }
  }


  /// Perform an LQ decomposition of matrix A=[L|0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
  /// Returns L as a Cholesky lower-triangular Covariance. All column coefficients must be axes, and A cannot be
  /// Euclidean-transformed.
#ifdef __cpp_concepts
  template<typed_matrix A> requires (not euclidean_transformed<A>)
#else
  template<typename A, std::enable_if_t<typed_matrix<A> and (not euclidean_transformed<A>), int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using C = typename MatrixTraits<A>::RowCoefficients;
    auto tm = LQ_decomposition(nested_matrix(std::forward<A>(a)));
    return SquareRootCovariance<C, decltype(tm)>(std::move(tm));
  }


  /// Perform a QR decomposition of matrix A=Q[U|0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns U as a Cholesky upper-triangular Covariance. All row coefficients must be axes.
#ifdef __cpp_concepts
  template<typed_matrix A>
#else
  template<typename A, std::enable_if_t<typed_matrix<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using C = typename MatrixTraits<A>::ColumnCoefficients;
    auto tm = QR_decomposition(nested_matrix(std::forward<A>(a)));
    return SquareRootCovariance<C, decltype(tm)>(std::move(tm));
  }


  /// Concatenate one or more typed matrices objects vertically.
#ifdef __cpp_concepts
  template<typed_matrix V, typed_matrix ... Vs> requires (sizeof...(Vs) == 0) or
    (equivalent_to<typename MatrixTraits<V>::ColumnCoefficients, typename MatrixTraits<Vs>::ColumnCoefficients> and ...)
#else
  template<typename V, typename ... Vs, std::enable_if_t<(typed_matrix<V> and ... and typed_matrix<Vs>) and
    ((sizeof...(Vs) == 0) or (equivalent_to<typename MatrixTraits<V>::ColumnCoefficients,
      typename MatrixTraits<Vs>::ColumnCoefficients> and ...)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = Concatenate<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients...>;
      decltype(auto) cat = concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...);
      return MatrixTraits<V>::template make<RC>(std::move(cat));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  /// Concatenate one or more typed matrices vertically. (Synonym for concatenate_vertical.)
#ifdef __cpp_concepts
  template<typed_matrix ... Vs>
#else
  template<typename ... Vs, std::enable_if_t<(typed_matrix<Vs> and ...), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(Vs&& ... vs) noexcept
  {
    return concatenate_vertical(std::forward<Vs>(vs)...);
  };


  /// Concatenate one or more matrix objects vertically.
#ifdef __cpp_concepts
template<typed_matrix V, typed_matrix ... Vs> requires (sizeof...(Vs) == 0) or
    (equivalent_to<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients> and ...)
#else
template<typename V, typename ... Vs, std::enable_if_t<(typed_matrix<V> and ... and typed_matrix<Vs>) and
    ((sizeof...(Vs) == 0) or (equivalent_to<typename MatrixTraits<V>::RowCoefficients,
      typename MatrixTraits<Vs>::RowCoefficients> and ...)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = typename MatrixTraits<V>::RowCoefficients;
      using CC = Concatenate<typename MatrixTraits<V>::ColumnCoefficients, typename MatrixTraits<Vs>::ColumnCoefficients...>;
      decltype(auto) cat = concatenate_horizontal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...);
      if constexpr(CC::axes_only)
      {
        return MatrixTraits<V>::template make<RC, CC>(std::move(cat));
      }
      else
        return make_Matrix<RC, CC>(std::move(cat));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  /// Concatenate one or more typed matrices diagonally.
#ifdef __cpp_concepts
  template<typed_matrix V, typed_matrix ... Vs>
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (typed_matrix<V> and ... and typed_matrix<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = Concatenate<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients...>;
      using CC = Concatenate<typename MatrixTraits<V>::ColumnCoefficients, typename MatrixTraits<Vs>::ColumnCoefficients...>;
      decltype(auto) cat = concatenate_diagonal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...);
      return MatrixTraits<V>::template make<RC, CC>(std::move(cat));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  namespace internal
  {
    template<typename Expr, typename CC>
    struct SplitMatVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<decltype(arg)>(arg));
      }
    };


    template<typename Expr, typename RC>
    struct SplitMatHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<decltype(arg)>(arg));
      }
    };


    template<typename Expr>
    struct SplitMatDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        static_assert(equivalent_to<RC, CC>);
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<decltype(arg)>(arg));
      }
    };
  }


  /// Split typed matrix into one or more typed matrices vertically.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typed_matrix M> requires
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<M>::RowCoefficients>
#else
  template<typename ... Cs, typename M, std::enable_if_t<typed_matrix<M> and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<M>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_vertical(M&& m) noexcept
  {
    using CC = typename MatrixTraits<M>::ColumnCoefficients;
    constexpr auto euclidean = euclidean_transformed<M>;
    return split_vertical<internal::SplitMatVertF<M, CC>, euclidean, Cs...>(nested_matrix(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typed_matrix M> requires
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<M>::ColumnCoefficients>
#else
  template<typename ... Cs, typename M, std::enable_if_t<typed_matrix<M> and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<M>::ColumnCoefficients>, int> = 0>
#endif
  inline auto
  split_horizontal(M&& m) noexcept
  {
    using RC = typename MatrixTraits<M>::RowCoefficients;
    return split_horizontal<internal::SplitMatHorizF<M, RC>, Cs...>(nested_matrix(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally. Column coefficients must all be Axis.
#ifdef __cpp_concepts
  template<std::size_t ... cuts, typed_matrix M> requires column_vector<M> and (sizeof...(cuts) > 0) and
    ((... + cuts) <= MatrixTraits<M>::columns)
#else
  template<std::size_t ... cuts, typename M,
    std::enable_if_t<typed_matrix<M> and column_vector<M> and (sizeof...(cuts) > 0) and
      ((0 + ... + cuts) <= MatrixTraits<M>::columns), int> = 0>
#endif
  inline auto
  split_horizontal(M&& m) noexcept
  {
    return split_horizontal<Axes<cuts>...>(std::forward<M>(m));
  }


  /// Split typed matrix into one or more typed matrices diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typed_matrix M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
  inline auto
  split_diagonal(M&& m) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<M>::RowCoefficients>);
    static_assert(equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, typename MatrixTraits<M>::RowCoefficients>);
    return split_diagonal<internal::SplitMatDiagF<M>, Cs...>(nested_matrix(std::forward<M>(m)));
  }


  /// Get element (i, j) of a typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires element_gettable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    element_gettable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
  }


  /// Get element (i) of a typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires element_gettable<nested_matrix_t<Arg>, 1>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    element_gettable<nested_matrix_t<Arg>, 1>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    return get_element(nested_matrix(std::forward<Arg>(arg)), i);
  }


  /// Set element (i, j) of a typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (not std::is_const_v<std::remove_reference_t<Arg>>) and
    element_settable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, std::enable_if_t<
    typed_matrix<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      element_settable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const typename MatrixTraits<Arg>::Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(wrapped_mean<Arg>)
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg, j] (const std::size_t row) {
        return get_element(nested_matrix(arg), row, j);
      };
      const auto set_coeff = [&arg, j](const typename MatrixTraits<Arg>::Scalar value, const std::size_t row) {
        set_element(nested_matrix(arg), value, row, j);
      };
      internal::wrap_set<Coeffs>(s, i, set_coeff, get_coeff);
    }
    else
    {
      set_element(nested_matrix(arg), s, i, j);
    }
  }


  /// Set element (i) of a typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (not std::is_const_v<std::remove_reference_t<Arg>>) and
    element_settable<nested_matrix_t<Arg>, 1>
#else
  template<typename Arg, std::enable_if_t<
    typed_matrix<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      element_settable<nested_matrix_t<Arg>, 1>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const typename MatrixTraits<Arg>::Scalar s, const std::size_t i)
  {
    if constexpr(wrapped_mean<Arg>)
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg] (const std::size_t row) {
        return get_element(nested_matrix(arg), row);
      };
      const auto set_coeff = [&arg](const typename MatrixTraits<Arg>::Scalar value, const std::size_t row) {
        set_element(nested_matrix(arg), value, row);
      };
      internal::wrap_set<Coeffs>(s, i, set_coeff, get_coeff);
    }
    else
    {
      set_element(nested_matrix(arg), s, i);
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    static_assert(column_vector<Arg>,
      "Runtime-indexed version of column function requires that all columns be identical and of type Axis.");
    // \TODO Make it so this function can accept any typed matrix with identically-typed columns.
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    using CC = Axis;
    return MatrixTraits<Arg>::template make<RC, CC>(column(nested_matrix(std::forward<Arg>(arg)), index));
  }


#ifdef __cpp_concepts
  template<std::size_t index, typed_matrix Arg>
#else
  template<std::size_t index, typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  column(Arg&& arg)
  {
    static_assert(index < MatrixTraits<Arg>::columns, "Column index out of range.");
    if constexpr (MatrixTraits<Arg>::columns == 1)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using RC = typename MatrixTraits<Arg>::RowCoefficients;
      using CC = typename MatrixTraits<Arg>::ColumnCoefficients::template Coefficient<index>;
      return MatrixTraits<Arg>::template make<RC, CC>(column<index>(nested_matrix(std::forward<Arg>(arg))));
    }
  }


  ////

#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<decltype(column(std::declval<Arg&>(), 0))>& >>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<Arg&>(), 0))>& >>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    static_assert(column_vector<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    // \TODO Make it so this function can accept any typed matrix with identically-typed columns.
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(col);
      f(mc);
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(c, f_nested);
    if constexpr(wrapped_mean<Arg>)
      c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<Arg&>(), 0))>&, std::size_t>>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<Function,
    std::decay_t<decltype(column(std::declval<Arg&>(), 0))>&, std::size_t>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    static_assert(column_vector<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(col);
      f(mc, i);
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(c, f_nested);
    if constexpr(wrapped_mean<Arg>)
      c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires (not std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&>>)
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    not std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&& >>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    static_assert(column_vector<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, MatrixTraits<Arg>::columns>;
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](const auto& col) {
      return nested_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(col)));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(nested_matrix(arg), f_nested));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires (not std::is_void_v<std::invoke_result_t<const Function&,
    std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>)
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    not std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    static_assert(column_vector<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0)), std::size_t>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, MatrixTraits<Arg>::columns>;
    const auto f_nested = [&f](const auto& col, std::size_t i) {
      using RC = typename MatrixTraits<Arg>::RowCoefficients;
      return nested_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(col), i));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(nested_matrix(arg), f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires typed_matrix<std::invoke_result_t<const Function&>>
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    typed_matrix<std::invoke_result_t<const Function&>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    const auto f_nested = [&f] { return nested_matrix(f()); };
    using ResultType = std::invoke_result_t<Function>;
    using RC = typename MatrixTraits<ResultType>::RowCoefficients;
    using CC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(CC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = Replicate<CC0, count>;
    return MatrixTraits<ResultType>::template make<RC, CC>(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    typed_matrix<std::invoke_result_t<const Function&, std::size_t>>
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    typed_matrix<std::invoke_result_t<const Function&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    const auto f_nested = [&f](std::size_t i) { return nested_matrix(f(i)); };
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    using RC = typename MatrixTraits<ResultType>::RowCoefficients;
    using CC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(CC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = Replicate<CC0, count>;
    return MatrixTraits<ResultType>::template make<RC, CC>(apply_columnwise<count>(f_nested));
  }


  ////

#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<typename MatrixTraits<Arg>::Scalar>&>>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<typename MatrixTraits<Arg>::Scalar>&>>, int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    apply_coefficientwise(nested_matrix(arg), f);
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(wrapped_mean<Arg>)
      nested_matrix(arg) = wrap_angles<RC>(nested_matrix(arg));
    return arg;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires std::is_void_v<std::invoke_result_t<const Function&,
    std::decay_t<typename MatrixTraits<Arg>::Scalar>&, std::size_t, std::size_t>>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&, std::size_t, std::size_t>>, int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    apply_coefficientwise(nested_matrix(arg), f);
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(wrapped_mean<Arg>)
      nested_matrix(arg) = wrap_angles<RC>(nested_matrix(arg));
    return arg;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires
    std::convertible_to<std::invoke_result_t<const Function&, std::decay_t<typename MatrixTraits<Arg>::Scalar>>,
      typename MatrixTraits<Arg>::Scalar>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<const Function&, std::decay_t<typename MatrixTraits<Arg>::Scalar>>,
      typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return MatrixTraits<Arg>::make(apply_coefficientwise(nested_matrix(arg), f));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires std::convertible_to<std::invoke_result_t<const Function&,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>, std::size_t, std::size_t>, typename MatrixTraits<Arg>::Scalar>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<const Function&,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>, std::size_t, std::size_t>,
      typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return MatrixTraits<Arg>::make(apply_coefficientwise(nested_matrix(arg), f));
  }


#ifdef __cpp_concepts
  template<typed_matrix V, typename Function> requires
    std::convertible_to<std::invoke_result_t<const Function&>, typename MatrixTraits<V>::Scalar>
#else
  template<typename V, typename Function, std::enable_if_t<typed_matrix<V> and
    std::is_convertible_v<std::invoke_result_t<const Function&>, typename MatrixTraits<V>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f)
  {
    constexpr auto rows = MatrixTraits<V>::dimension;
    constexpr auto columns = MatrixTraits<V>::columns;
    using Scalar = typename MatrixTraits<V>::Scalar;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function>>,
      std::decay_t<typename MatrixTraits<V>::Scalar>>)
    {
      return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(f));
    }
    else
    {
      const auto f_conv = [&f] { return static_cast<Scalar>(f()); };
      return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(f_conv));
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix V, typename Function> requires std::convertible_to<
    std::invoke_result_t<const Function&, std::size_t, std::size_t>, typename MatrixTraits<V>::Scalar>
#else
  template<typename V, typename Function, std::enable_if_t<typed_matrix<V> and std::is_convertible_v<
    std::invoke_result_t<const Function&, std::size_t, std::size_t>, typename MatrixTraits<V>::Scalar>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f)
  {
    constexpr auto rows = MatrixTraits<V>::dimension;
    constexpr auto columns = MatrixTraits<V>::columns;
    using Scalar = typename MatrixTraits<V>::Scalar;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function, std::size_t, std::size_t>>,
      std::decay_t<typename MatrixTraits<V>::Scalar>>)
    {
      return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(f));
    }
    else
    {
      const auto f_conv = [&f](size_t i, size_t j){ return static_cast<Scalar>(f(i, j)); };
      return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(f_conv));
    }
  }


  /**
   * Fill a typed  matrix with random values selected from a random distribution.
   * The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
#ifdef __cpp_concepts
  template<
    typed_matrix ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params>
#else
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<typed_matrix<ReturnType>, int> = 0>
#endif
  static auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    constexpr auto rows = MatrixTraits<ReturnType>::dimension;
    constexpr auto cols = MatrixTraits<ReturnType>::columns;
    using Ps = typename distribution_type<Scalar>::param_type;
    static_assert(std::is_constructible_v<Ps, Params...> or sizeof...(Params) == rows or sizeof...(Params) == rows * cols,
      "Params... must be (1) a parameter set or list of parameter sets, "
      "(2) a list of parameter sets, one for each row, or (3) a list of parameter sets, one for each coefficient.");
    using B = nested_matrix_t<ReturnType>;
    return MatrixTraits<ReturnType>::template make(randomize<B, distribution_type, random_number_engine>(params...));
  }


  /// Output the vector to a stream.
#ifdef __cpp_concepts
  template<typed_matrix V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V>, int> = 0>
#endif
  inline std::ostream& operator<<(std::ostream& os, const V& v)
  {
    os << make_native_matrix(v);
    return os;
  }


}

#endif //OPENKALMAN_TYPED_MATRIX_OVERLOADS_H