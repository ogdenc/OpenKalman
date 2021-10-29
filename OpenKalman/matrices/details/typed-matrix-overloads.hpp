/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPED_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_TYPED_MATRIX_OVERLOADS_HPP


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


  #ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).nested_matrix();
  }


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
    if constexpr(self_contained<Arg> or std::is_lvalue_reference_v<nested_matrix_t<Arg>> or
      ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(make_self_contained(nested_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr std::size_t row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return row_count(nested_matrix(std::forward<Arg>(arg)));
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  constexpr std::size_t column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return column_count(nested_matrix(std::forward<Arg>(arg)));
    else
      return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(not euclidean_transformed<Arg>)
    {
      auto n = OpenKalman::to_euclidean<Coefficients>(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<Coefficients, decltype(n)> {std::move(n)};
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(euclidean_transformed<Arg>)
    {
      auto n = OpenKalman::from_euclidean<Coefficients>(nested_matrix(std::forward<Arg>(arg)));
      return Mean<Coefficients, decltype(n)> {std::move(n)};
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
    auto b = to_diagonal(nested_matrix(std::forward<Arg>(arg)));
    return Matrix<C, C, decltype(b)>(std::move(b));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    auto b = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
    return Matrix<C, Axis, decltype(b)>(std::move(b));
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
    {
      auto b = transpose(nested_matrix(from_euclidean(std::forward<Arg>(arg))));
      return Matrix<CCols, CRows, decltype(b)>(std::move(b));
    }
    else
    {
      auto b = transpose(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<CCols, CRows, decltype(b)>(std::move(b));
    }
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
    {
      auto b = adjoint(nested_matrix(from_euclidean(std::forward<Arg>(arg))));
      return Matrix<CCols, CRows, decltype(b)>(std::move(b));
    }
    else
    {
      auto b = adjoint(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<CCols, CRows, decltype(b)>(std::move(b));
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    return determinant(nested_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    return trace(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Solves AX = B for X, where X and B are means of the same type, and A is a square matrix with compatible types.
  /// If wrapping occurs, it will be both before for B and after for the X result.
#ifdef __cpp_concepts
  template<typed_matrix A, typed_matrix B> requires square_matrix<A> and
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<B>::RowCoefficients>
#else
  template<typename A, typename B, std::enable_if_t<typed_matrix<A> and typed_matrix<B> and square_matrix<A> and
    equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<B>::RowCoefficients>, int> = 0>
#endif
  inline auto
  solve(A&& a, B&& b) noexcept
  {
    auto x = solve(nested_matrix(std::forward<A>(a)), nested_matrix(std::forward<B>(b)));
    return MatrixTraits<B>::make(std::move(x));
  }


  /// Returns the mean of the column vectors after they are transformed into Euclidean space.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires (MatrixTraits<Arg>::columns == 1) or untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    ((MatrixTraits<Arg>::columns == 1) or untyped_columns<Arg>), int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    /// \todo add an option where all the column coefficients are the same, but not Axis.
    using C = typename MatrixTraits<Arg>::RowCoefficients;

    if constexpr(MatrixTraits<Arg>::columns == 1)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(euclidean_transformed<Arg>)
    {
      auto b = reduce_columns(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(b)> {std::move(b)};
    }
    else if constexpr(mean<Arg>)
    {
      auto b = from_euclidean<C>(reduce_columns(nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(b)> {std::move(b)};
    }
    else
    {
      auto b = reduce_columns(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<C, Axis, decltype(b)> {std::move(b)};
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
      return MatrixTraits<V>::template make<RC>(
        concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
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
      using CC = Concatenate<typename MatrixTraits<V>::ColumnCoefficients,
      typename MatrixTraits<Vs>::ColumnCoefficients...>;
      auto cat = concatenate_horizontal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...);
      if constexpr(CC::axes_only)
      {
        return MatrixTraits<V>::template make<RC, CC>(std::move(cat));
      }
      else
      {
        return make_matrix<RC, CC>(std::move(cat));
      }
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
      using CC = Concatenate<typename MatrixTraits<V>::ColumnCoefficients,
        typename MatrixTraits<Vs>::ColumnCoefficients...>;
      return MatrixTraits<V>::template make<RC, CC>(
        concatenate_diagonal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
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
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<Arg>(arg));
      }
    };


    template<typename Expr, typename RC>
    struct SplitMatHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<Arg>(arg));
      }
    };


    template<typename Expr>
    struct SplitMatDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        static_assert(equivalent_to<RC, CC>);
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<Arg>(arg));
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
    return split_vertical<oin::SplitMatVertF<M, CC>, euclidean, Cs...>(nested_matrix(std::forward<M>(m)));
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
    return split_horizontal<oin::SplitMatHorizF<M, RC>, Cs...>(nested_matrix(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally. Column coefficients must all be Axis.
#ifdef __cpp_concepts
  template<std::size_t ... cuts, typed_matrix M> requires untyped_columns<M> and (sizeof...(cuts) > 0) and
    ((... + cuts) <= MatrixTraits<M>::columns)
#else
  template<std::size_t ... cuts, typename M,
    std::enable_if_t<typed_matrix<M> and untyped_columns<M> and (sizeof...(cuts) > 0) and
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
    return split_diagonal<oin::SplitMatDiagF<M>, Cs...>(nested_matrix(std::forward<M>(m)));
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
      const auto set_coeff = [&arg, j](const std::size_t row, const typename MatrixTraits<Arg>::Scalar value) {
        set_element(nested_matrix(arg), value, row, j);
      };
      oin::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
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
      const auto set_coeff = [&arg](const std::size_t row, const typename MatrixTraits<Arg>::Scalar value) {
        set_element(nested_matrix(arg), value, row);
      };
      oin::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
    }
    else
    {
      set_element(nested_matrix(arg), s, i);
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
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
    if constexpr (column_vector<Arg>)
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
  template<typed_matrix Arg, typename Function> requires untyped_columns<Arg> and
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col) { f(col); } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    std::is_invocable_v<const Function&,
      std::decay_t<decltype(column(std::declval<Arg&>(), 0))>& > and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(std::move(col));
      f(mc);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(c, f_nested);
    if constexpr(wrapped_mean<Arg>) c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires untyped_columns<Arg> and
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col, std::size_t i) {
      f(col, i);
    } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    std::is_invocable_v<Function,
    std::decay_t<decltype(column(std::declval<Arg&>(), 0))>&, std::size_t> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(std::move(col));
      f(mc, i);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(c, f_nested);
    if constexpr(wrapped_mean<Arg>) c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires untyped_columns<Arg> and
    requires(const Arg& arg, const Function& f) {
      {f(column<0>(arg))} -> typed_matrix;
      {f(column<0>(arg))} -> column_vector;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    typed_matrix<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&& >>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::dimensions == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, MatrixTraits<Arg>::columns>;
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto&& col) -> auto {
      return make_self_contained(
        nested_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(std::forward<decltype(col)>(col)))));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(nested_matrix(arg), f_nested));
  }


#ifdef __cpp_concepts
  template<typed_matrix Arg, typename Function> requires untyped_columns<Arg> and
    requires(const Arg& arg, const Function& f, std::size_t i) {
      {f(column<0>(arg), i)} -> typed_matrix;
      {f(column<0>(arg), i)} -> column_vector;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    typed_matrix<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0)), std::size_t>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::dimensions == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, MatrixTraits<Arg>::columns>;
    const auto f_nested = [&f](auto&& col, std::size_t i) -> auto {
      using RC = typename MatrixTraits<Arg>::RowCoefficients;
      return make_self_contained(
        nested_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(std::forward<decltype(col)>(col)), i)));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(nested_matrix(arg), f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) {
      {f()} -> typed_matrix;
      {nested_matrix(f())} -> column_vector;
    }
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
    static_assert(CC0::dimensions == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = Replicate<CC0, count>;
    return MatrixTraits<ResultType>::template make<RC, CC>(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) {
      {f(i)} -> typed_matrix;
      {nested_matrix(f(i))} -> column_vector;
    }
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
    static_assert(CC0::dimensions == 1, "Function argument of apply_columnwise must return a column vector.");
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
    constexpr auto rows = MatrixTraits<V>::rows;
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
  apply_coefficientwise(Function&& f)
  {
    constexpr auto rows = MatrixTraits<V>::rows;
    constexpr auto columns = MatrixTraits<V>::columns;
    using Scalar = typename MatrixTraits<V>::Scalar;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function, std::size_t, std::size_t>>,
      std::decay_t<typename MatrixTraits<V>::Scalar>>)
    {
      return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(std::forward<Function>(f)));
    }
    else
    {
      if constexpr (std::is_lvalue_reference_v<Function>)
      {
        auto f_conv = [&f](size_t i, size_t j){ return static_cast<Scalar>(f(i, j)); };
        return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(std::move(f_conv)));
      }
      else
      {
        auto f_conv = [f = std::move(f)](size_t i, size_t j){ return static_cast<Scalar>(f(i, j)); };
        return MatrixTraits<V>::make(apply_coefficientwise<rows, columns>(std::move(f_conv)));
      }


    }
  }


  /**
   * \brief Fill a fixed-shape typed matrix with random values selected from a random distribution.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *
   *  - One distribution for the entire matrix. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Matrix<Axes<2>, Axes<2>, Eigen::Matrix<double, 2, 2>>>(N {1.0, 0.3}));
   *   \endcode
   *
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix n containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     auto n = randomize<Matrix<Axes<2>, Axes<2>, Eigen::Matrix<double, 2, 2>>>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *
   *  - One distribution for each row. The following code constructs a 3-by-2 (o) or 2-by-2 (p) matrices
   *  in which elements in each row are selected according to the three (o) or two (p) listed distribution
   *  parameters:
   *   \code
   *     auto o = randomize<Matrix<Axes<3>, Coefficients<angle::Radians, angle::Radians>, Eigen::Matrix<double, 3, 2>>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto p = randomize<Matrix<Axes<2>, Coefficients<angle::Radians, angle::Radians>, Eigen::Matrix<double, 2, 2>>>(N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of p, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *
   *  - One distribution for each column. The following code constructs 2-by-3 matrix m
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto m = randomize<Matrix<Coefficients<angle::Radians, angle::Radians>, Axes<3>, Eigen::Matrix<double, 2, 3>>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   *
   * \tparam ReturnType The return type reflecting the size of the matrix to be filled. The actual result will be
   * a fixed typed matrix.
   * \tparam random_number_engine The random number engine.
   * \tparam Dists A set of distributions (e.g., std::normal_distribution<double>) or, alternatively,
   * means (a definite, non-stochastic value).
  **/
#ifdef __cpp_concepts
  template<typed_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename...Dists>
  requires (not dynamic_shape<ReturnType>) and (sizeof...(Dists) > 0) and
    (((requires { typename std::decay_t<Dists>::result_type;  typename std::decay_t<Dists>::param_type; } or
      std::is_arithmetic_v<std::decay_t<Dists>>) and ... )) and
    ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
    (sizeof...(Dists) == 1 or MatrixTraits<ReturnType>::rows * MatrixTraits<ReturnType>::columns == sizeof...(Dists) or
      MatrixTraits<ReturnType>::rows == sizeof...(Dists) or MatrixTraits<ReturnType>::columns == sizeof...(Dists))
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<typed_matrix<ReturnType> and (not dynamic_shape<ReturnType>) and (sizeof...(Dists) > 0) and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
      (sizeof...(Dists) == 1 or
        MatrixTraits<ReturnType>::rows * MatrixTraits<ReturnType>::columns == sizeof...(Dists) or
        MatrixTraits<ReturnType>::rows == sizeof...(Dists) or
        MatrixTraits<ReturnType>::columns == sizeof...(Dists)), int> = 0>
#endif
  inline auto
  randomize(Dists&& ... dists)
  {
    using B = nested_matrix_t<ReturnType>;
    return MatrixTraits<ReturnType>::template make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape typed_matrix with random values selected from a single random distribution.
   * \details The following example constructs two 2-by-2 matrices (m, n, and p) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     auto m = randomize<Matrix<Axes<2>, Axes<2>, Eigen::Matrix<float, 2, Eigen::Dynamic>>>(2, 2, std::normal_distribution<float> {1.0, 0.3}));
   *     auto n = randomize<Matrix<Axes<2>, Axes<2>, Eigen::Matrix<double, Eigen::Dynamic, 2>>>(2, 2, std::normal_distribution<double> {1.0, 0.3}));
   *     auto p = randomize<Matrix<Axes<2>, Axes<2>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>>(2, 2, std::normal_distribution<double> {1.0, 0.3});
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \param rows Number of rows, decided at runtime. Must match rows of ReturnType if they are fixed.
   * \param columns Number of columns, decided at runtime. Must match columns of ReturnType if they are fixed.
   * \tparam Dist A distribution (type distribution_type).
   **/
#ifdef __cpp_concepts
  template<typed_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename Dist>
  requires dynamic_shape<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
      typed_matrix<ReturnType> and dynamic_shape<ReturnType> and
      (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    if constexpr (not dynamic_rows<ReturnType>) assert(rows == MatrixTraits<ReturnType>::rows);
    if constexpr (not dynamic_columns<ReturnType>) assert(columns == MatrixTraits<ReturnType>::columns);
    using B = nested_matrix_t<ReturnType>;
    return MatrixTraits<ReturnType>::template make(randomize<B, random_number_engine>(
      rows, columns, std::forward<Dist>(dist)));
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