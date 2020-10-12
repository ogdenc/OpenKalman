/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPEDMATRIXOVERLOADS_H
#define OPENKALMAN_TYPEDMATRIXOVERLOADS_H


namespace OpenKalman
{
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  base_matrix(Arg&& arg) noexcept { return std::forward<Arg>(arg).base_matrix(); }


  /// Convert to strict regular matrix (wrapping any angles, if necessary).
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict_matrix(Arg&& arg) noexcept
  {
    return strict_matrix(base_matrix(std::forward<Arg>(arg)));
  }


  /// Convert vector object to strict version (wrapping any angles).
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict(Arg&& arg) noexcept
  {
    if constexpr(is_strict_v<typename MatrixTraits<Arg>::BaseMatrix>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(strict(base_matrix(std::forward<Arg>(arg))));
    }
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  to_Euclidean(Arg&& arg) noexcept
  {
    static_assert(is_column_vector_v<Arg>);
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(not is_Euclidean_transformed_v<Arg>)
    {
      auto e = OpenKalman::to_Euclidean<Coefficients>(base_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<Coefficients, std::decay_t<decltype(e)>>(std::move(e));
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  from_Euclidean(Arg&& arg) noexcept
  {
    static_assert(is_column_vector_v<Arg>);
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(is_Euclidean_transformed_v<Arg>)
    {
      auto e = OpenKalman::from_Euclidean<Coefficients>(base_matrix(std::forward<Arg>(arg)));
      return Mean<Coefficients, std::decay_t<decltype(e)>>(std::move(e));
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, Axis>);
    static_assert(not is_Euclidean_transformed_v<Arg>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    auto b = to_diagonal(base_matrix(std::forward<Arg>(arg)));
    return TypedMatrix<C, C, decltype(b)>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  transpose(Arg&& arg) noexcept
  {
    static_assert(not is_Euclidean_transformed_v<Arg>);
    using CRows = typename MatrixTraits<Arg>::RowCoefficients;
    using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
    if constexpr(is_Euclidean_transformed_v<Arg>)
      return make_Matrix<CCols, CRows>(transpose(base_matrix(from_Euclidean(std::forward<Arg>(arg)))));
    else
      return make_Matrix<CCols, CRows>(transpose(base_matrix(std::forward<Arg>(arg))));
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  adjoint(Arg&& arg) noexcept
  {
    static_assert(not is_Euclidean_transformed_v<Arg>);
    using CRows = typename MatrixTraits<Arg>::RowCoefficients;
    using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
    if constexpr(is_Euclidean_transformed_v<Arg>)
      return make_Matrix<CCols, CRows>(adjoint(base_matrix(from_Euclidean(std::forward<Arg>(arg)))));
    else
      return make_Matrix<CCols, CRows>(adjoint(base_matrix(std::forward<Arg>(arg))));
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return determinant(base_matrix(std::forward<Arg>(arg)));
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return trace(base_matrix(std::forward<Arg>(arg)));
  }


  /// Solves AX = B for X, where X and B are means of the same type, and A is a square matrix with compatible types.
  /// If wrapping occurs, it will be both before for B and after for the X result.
  template<
    typename A, typename B,
    std::enable_if_t<is_typed_matrix_v<A>, int> = 0,
    std::enable_if_t<is_typed_matrix_v<B>, int> = 0>
  inline auto
  solve(A&& a, B&& b) noexcept
  {
    using C = typename MatrixTraits<A>::RowCoefficients;
    static_assert(OpenKalman::is_equivalent_v<C, typename MatrixTraits<B>::RowCoefficients>);
    static_assert(OpenKalman::is_equivalent_v<C, typename MatrixTraits<A>::ColumnCoefficients>);
    auto x = solve(base_matrix(std::forward<A>(a)), base_matrix(std::forward<B>(b)));
    return MatrixTraits<B>::make(std::move(x));
  }


  /// Returns the mean of the column vectors after they are transformed into Euclidean space.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::columns == 1)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      static_assert(is_column_vector_v<Arg>);
      if constexpr(is_Euclidean_transformed_v<Arg>)
      {
        return MatrixTraits<Arg>::make(reduce_columns(base_matrix(std::forward<Arg>(arg))));
      }
      else if constexpr(is_mean_v<Arg>)
      {
        using C = typename MatrixTraits<Arg>::RowCoefficients;
        auto ev = reduce_columns(base_matrix(to_Euclidean(std::forward<Arg>(arg))));
        return MatrixTraits<Arg>::make(from_Euclidean<C>(std::move(ev)));
      }
      else
      {
        using C = typename MatrixTraits<Arg>::RowCoefficients;
        return make_Matrix<C, Axis>(reduce_columns(base_matrix(std::forward<Arg>(arg))));
      }
    }
  }


  /// Perform an LQ decomposition of matrix A=[L|0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
  /// Returns L as a Cholesky lower-triangular Covariance. All column coefficients must be axes, and A cannot be
  /// Euclidean-transformed.
  template<typename A, std::enable_if_t<is_typed_matrix_v<A>, int> = 0>
  inline auto
  LQ_decomposition(A&& a)
  {
    static_assert(not is_Euclidean_transformed_v<A>);
    using C = typename MatrixTraits<A>::RowCoefficients;
    auto tm = LQ_decomposition(base_matrix(std::forward<A>(a)));
    return SquareRootCovariance<C, decltype(tm)>(std::move(tm));
  }


  /// Perform a QR decomposition of matrix A=Q[U|0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns U as a Cholesky upper-triangular Covariance. All row coefficients must be axes.
  template<typename A, std::enable_if_t<is_typed_matrix_v<A>, int> = 0>
  inline auto
  QR_decomposition(A&& a)
  {
    using C = typename MatrixTraits<A>::ColumnCoefficients;
    auto tm = QR_decomposition(base_matrix(std::forward<A>(a)));
    return SquareRootCovariance<C, decltype(tm)>(std::move(tm));
  }


  /// Concatenate one or more typed matrices objects vertically.
  template<
    typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<V>, is_typed_matrix<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      static_assert((is_equivalent_v<
        typename MatrixTraits<V>::ColumnCoefficients,
        typename MatrixTraits<Vs>::ColumnCoefficients> and ...));
      using RC = Concatenate<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients...>;
      decltype(auto) cat = concatenate_vertical(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...);
      return MatrixTraits<V>::template make<RC>(std::move(cat));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  /// Concatenate one or more typed matrices vertically. (Synonym for concatenate_vertical.)
  template<typename ... Vs, std::enable_if_t<std::conjunction_v<is_typed_matrix<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate(Vs&& ... vs) noexcept
  {
    return concatenate_vertical(std::forward<Vs>(vs)...);
  };


  /// Concatenate one or more matrix objects vertically.
  template<typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<V>, is_typed_matrix<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      static_assert((is_equivalent_v<
        typename MatrixTraits<V>::RowCoefficients,
        typename MatrixTraits<Vs>::RowCoefficients> and ...));
      using RC = typename MatrixTraits<V>::RowCoefficients;
      using CC = Concatenate<typename MatrixTraits<V>::ColumnCoefficients, typename MatrixTraits<Vs>::ColumnCoefficients...>;
      decltype(auto) cat = concatenate_horizontal(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...);
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
  template<
    typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<V>, is_typed_matrix<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = Concatenate<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients...>;
      using CC = Concatenate<typename MatrixTraits<V>::ColumnCoefficients, typename MatrixTraits<Vs>::ColumnCoefficients...>;
      decltype(auto) cat = concatenate_diagonal(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...);
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
        static_assert(is_equivalent_v<RC, CC>);
        return MatrixTraits<Expr>::template make<RC, CC>(std::forward<decltype(arg)>(arg));
      }
    };
  }

  /// Split typed matrix into one or more typed matrices vertically.
  template<typename ... Cs, typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
  inline auto
  split_vertical(M&& m) noexcept
  {
    using CC = typename MatrixTraits<M>::ColumnCoefficients;
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<M>::RowCoefficients>);
    constexpr auto euclidean = is_Euclidean_transformed_v<M>;
    return split_vertical<internal::SplitMatVertF<M, CC>, euclidean, Cs...>(base_matrix(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally.
  template<typename ... Cs, typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
  inline auto
  split_horizontal(M&& m) noexcept
  {
    using RC = typename MatrixTraits<M>::RowCoefficients;
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<M>::ColumnCoefficients>);
    return split_horizontal<internal::SplitMatHorizF<M, RC>, Cs...>(base_matrix(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally. Column coefficients must all be Axis.
  template<std::size_t ... cuts, typename M,
    std::enable_if_t<is_typed_matrix_v<M> and is_column_vector_v<M> and (sizeof...(cuts) > 0), int> = 0>
  inline auto
  split_horizontal(M&& m) noexcept
  {
    static_assert((... + cuts) <= MatrixTraits<M>::columns);
    return split_horizontal<Axes<cuts>...>(std::forward<M>(m));
  }


  /// Split typed matrix into one or more typed matrices diagonally.
  template<typename ... Cs, typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
  inline auto
  split_diagonal(M&& m) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<M>::RowCoefficients>);
    static_assert(is_equivalent_v<typename MatrixTraits<M>::ColumnCoefficients, typename MatrixTraits<M>::RowCoefficients>);
    return split_diagonal<internal::SplitMatDiagF<M>, Cs...>(base_matrix(std::forward<M>(m)));
  }


  /// Get element (i, j) of a typed matrix.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    return get_element(base_matrix(std::forward<Arg>(arg)), i, j);
  }


  /// Get element (i) of a typed matrix.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    return get_element(base_matrix(std::forward<Arg>(arg)), i);
  }


  /// Set element (i, j) of a typed matrix.
  template<typename Arg, typename Scalar,
    std::enable_if_t<is_typed_matrix_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(is_wrapped_v<Arg>)
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg, j] (const std::size_t row) {
        return get_element(base_matrix(arg), row, j);
      };
      const auto set_coeff = [&arg, j](const Scalar value, const std::size_t row) {
        set_element(base_matrix(arg), value, row, j);
      };
      wrap_set<Coeffs>(s, i, set_coeff, get_coeff);
    }
    else
    {
      set_element(base_matrix(arg), s, i, j);
    }
  }


  /// Set element (i) of a typed matrix.
  template<typename Arg, typename Scalar,
    std::enable_if_t<is_typed_matrix_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    if constexpr(is_wrapped_v<Arg>)
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg] (const std::size_t row) {
        return get_element(base_matrix(arg), row);
      };
      const auto set_coeff = [&arg](const Scalar value, const std::size_t row) {
        set_element(base_matrix(arg), value, row);
      };
      wrap_set<Coeffs>(s, i, set_coeff, get_coeff);
    }
    else
    {
      set_element(base_matrix(arg), s, i);
    }
  }


  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    static_assert(is_column_vector_v<Arg>,
      "Runtime-indexed version of column function requires that all columns be identical and of type Axis.");
    /// @TODO Make it so this function can accept any typed matrix with identically-typed columns.
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    using CC = Axis;
    return MatrixTraits<Arg>::template make<RC, CC>(column(base_matrix(std::forward<Arg>(arg)), index));
  }


  template<size_t index, typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
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
      return MatrixTraits<Arg>::template make<RC, CC>(column<index>(base_matrix(std::forward<Arg>(arg))));
    }
  }


  ////

  template<
    typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_void_v<std::invoke_result_t<Function,
    std::decay_t<decltype(column(std::declval<Arg>(), 0))>& >>, int> = 0>
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    static_assert(is_column_vector_v<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    /// @TODO Make it so this function can accept any typed matrix with identically-typed columns.
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_base = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(col);
      f(mc);
    };
    auto& c = base_matrix(arg);
    apply_columnwise(c, f_base);
    if constexpr(is_wrapped_v<Arg>)
      c = wrap_angles<RC>(c);
    return arg;
  }


  template<
    typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_void_v<std::invoke_result_t<Function,
    std::decay_t<decltype(column(std::declval<Arg>(), 0))>&, std::size_t>>, int> = 0>
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    static_assert(is_column_vector_v<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_base = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(col);
      f(mc, i);
    };
    auto& c = base_matrix(arg);
    apply_columnwise(c, f_base);
    if constexpr(is_wrapped_v<Arg>)
      c = wrap_angles<RC>(c);
    return arg;
  }


  template<
    typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<Arg>(), 0))>&& >>, int> = 0>
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    static_assert(is_column_vector_v<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, MatrixTraits<Arg>::columns>;
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_base = [&f](const auto& col) {
      return base_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(col)));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(base_matrix(arg), f_base));
  }


  template<
    typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<Arg>(), 0))>&&, std::size_t>>, int> = 0>
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    static_assert(is_column_vector_v<Arg>,
      "Columnwise application requires that all columns be identical column vectors of type Axis.");
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0)), std::size_t>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, MatrixTraits<Arg>::columns>;
    const auto f_base = [&f](const auto& col, std::size_t i) {
      using RC = typename MatrixTraits<Arg>::RowCoefficients;
      return base_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(col), i));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(base_matrix(arg), f_base));
  }


  template<
    std::size_t count, typename Function,
    std::enable_if_t<is_typed_matrix_v<std::invoke_result_t<Function>>, int> = 0>
  inline auto
  apply_columnwise(const Function& f)
  {
    const auto f_base = [&f] { return base_matrix(f()); };
    using ResultType = std::invoke_result_t<Function>;
    using RC = typename MatrixTraits<ResultType>::RowCoefficients;
    using CC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(CC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = Replicate<CC0, count>;
    return MatrixTraits<ResultType>::template make<RC, CC>(apply_columnwise<count>(f_base));
  }


  template<
    std::size_t count, typename Function,
    std::enable_if_t<is_typed_matrix_v<std::invoke_result_t<Function, std::size_t>>, int> = 0>
  inline auto
  apply_columnwise(const Function& f)
  {
    const auto f_base = [&f](std::size_t i) { return base_matrix(f(i)); };
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    using RC = typename MatrixTraits<ResultType>::RowCoefficients;
    using CC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(CC0::size == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = Replicate<CC0, count>;
    return MatrixTraits<ResultType>::template make<RC, CC>(apply_columnwise<count>(f_base));
  }


  ////

  template<typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&>>, int> = 0>
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    apply_coefficientwise(base_matrix(arg), f);
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(is_wrapped_v<Arg>)
      base_matrix(arg) = wrap_angles<RC>(base_matrix(arg));
    return arg;
  }


  template<typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&, std::size_t, std::size_t>>, int> = 0>
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    apply_coefficientwise(base_matrix(arg), f);
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(is_wrapped_v<Arg>)
      base_matrix(arg) = wrap_angles<RC>(base_matrix(arg));
    return arg;
  }


  template<typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_convertible_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>>,
      typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return MatrixTraits<Arg>::make(apply_coefficientwise(base_matrix(arg), f));
  }


  template<typename Arg, typename Function, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0,
    std::enable_if_t<std::is_convertible_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>, std::size_t, std::size_t>,
      typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return MatrixTraits<Arg>::make(apply_coefficientwise(base_matrix(arg), f));
  }


  template<typename V, typename Function, std::enable_if_t<is_typed_matrix_v<V>, int> = 0,
    std::enable_if_t<std::is_convertible_v<
      std::invoke_result_t<Function>, typename MatrixTraits<V>::Scalar>, int> = 0>
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


  template<typename V, typename Function, std::enable_if_t<is_typed_matrix_v<V>, int> = 0,
    std::enable_if_t<std::is_convertible_v<
      std::invoke_result_t<Function, std::size_t, std::size_t>, typename MatrixTraits<V>::Scalar>, int> = 0>
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
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<is_typed_matrix_v<ReturnType>, int> = 0>
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
    using B = typename MatrixTraits<ReturnType>::BaseMatrix;
    return MatrixTraits<ReturnType>::template make(randomize<B, distribution_type, random_number_engine>(params...));
  }


  /// Output the vector to a stream.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V>, int> = 0>
  inline std::ostream& operator<<(std::ostream& os, const V& v)
  {
    os << strict_matrix(v);
    return os;
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

  /// Add two typed matrices. If the operands are of different types, the result will be a regular typed matrix.
  template<typename V1, typename V2, std::enable_if_t<is_typed_matrix_v<V1> and is_typed_matrix_v<V2>, int> = 0>
  inline auto operator+(V1&& v1, V2&& v2)
  {
    using RC1 = typename MatrixTraits<V1>::RowCoefficients;
    using CC1 = typename MatrixTraits<V1>::ColumnCoefficients;
    static_assert(is_equivalent_v<typename MatrixTraits<V2>::RowCoefficients, RC1>);
    static_assert(is_equivalent_v<typename MatrixTraits<V2>::ColumnCoefficients, CC1>);
    static_assert(is_Euclidean_transformed_v<V1> == is_Euclidean_transformed_v<V2>);
    using CommonV = std::decay_t<std::conditional_t<
      (is_Euclidean_mean_v<V1> and is_Euclidean_mean_v<V2>) or (is_mean_v<V1> and is_mean_v<V2>),
      V1, TypedMatrix<RC1, CC1, typename MatrixTraits<V1>::BaseMatrix>>>;
    auto ret = MatrixTraits<CommonV>::make(base_matrix(std::forward<V1>(v1)) + base_matrix(std::forward<V2>(v2)));
    if constexpr (not std::is_lvalue_reference_v<V1&&> or not std::is_lvalue_reference_v<V2&&>) return strict(std::move(ret)); else return ret;
  }


  /// Subtract two typed matrices. The result is a regular typed matrix unless both operands are EuclideanMean.
  template<typename V1, typename V2, std::enable_if_t<is_typed_matrix_v<V1> and is_typed_matrix_v<V2>, int> = 0>
  inline auto operator-(V1&& v1, V2&& v2)
  {
    using RC1 = typename MatrixTraits<V1>::RowCoefficients;
    using CC1 = typename MatrixTraits<V1>::ColumnCoefficients;
    static_assert(is_equivalent_v<typename MatrixTraits<V2>::RowCoefficients, RC1>);
    static_assert(is_equivalent_v<typename MatrixTraits<V2>::ColumnCoefficients, CC1>);
    static_assert(is_Euclidean_transformed_v<V1> == is_Euclidean_transformed_v<V2>);
    using CommonV = std::decay_t<std::conditional_t<
      (is_Euclidean_mean_v<V1> and is_Euclidean_mean_v<V2>),
      V1, TypedMatrix<RC1, CC1, typename MatrixTraits<V1>::BaseMatrix>>>;
    auto b = base_matrix(std::forward<V1>(v1)) - base_matrix(std::forward<V2>(v2));
    if constexpr (is_mean_v<V1> and is_mean_v<V2>)
    {
      // WC is the difference type for the coefficients. However, the result should retain coefficient types RC1.
      using WC = typename RC1::difference_type;
      auto ret = MatrixTraits<CommonV>::make(wrap_angles<WC>(std::move(b)));
      if constexpr (not std::is_lvalue_reference_v<V1&&> or not std::is_lvalue_reference_v<V2&&>)
        return strict(std::move(ret));
      else return ret;
    }
    else
    {
      auto ret = MatrixTraits<CommonV>::make(std::move(b));
      if constexpr (not std::is_lvalue_reference_v<V1&&> or not std::is_lvalue_reference_v<V2&&>)
        return strict(std::move(ret));
      else return ret;
    }
  }


  /// Multiply a typed matrix by a scalar. The result type is the same as the operand type, so angles in the result may be wrapped.
  template<
    typename V,
    typename S,
    std::enable_if_t<is_typed_matrix_v<V>, int> = 0,
    std::enable_if_t<std::is_convertible_v<S, const typename MatrixTraits<V>::Scalar>, int> = 0>
  inline auto operator*(V&& v, S scale)
  {
    using Sc = typename MatrixTraits<V>::Scalar;
    auto ret = MatrixTraits<V>::make(base_matrix(std::forward<V>(v)) * static_cast<Sc>(scale));
    if constexpr (not std::is_lvalue_reference_v<V&&>) return strict(std::move(ret)); else return ret;
  }


  /// Multiply a scalar by a typed matrix. The result type is the same as the operand type, so angles in the result may be wrapped.
  template<
    typename V,
    typename S,
    std::enable_if_t<is_typed_matrix_v<V>, int> = 0,
    std::enable_if_t<std::is_convertible_v<S, const typename MatrixTraits<V>::Scalar>, int> = 0>
  inline auto operator*(S scale, V&& v)
  {
    using Sc = const typename MatrixTraits<V>::Scalar;
    auto ret = MatrixTraits<V>::make(static_cast<Sc>(scale) * base_matrix(std::forward<V>(v)));
    if constexpr (not std::is_lvalue_reference_v<V&&>) return strict(std::move(ret)); else return ret;
  }


  /// Divide a typed matrix by a scalar. The result type is the same as the operand type, so angles in the result may be wrapped.
  template<
    typename V,
    typename S,
    std::enable_if_t<is_typed_matrix_v<V>, int> = 0,
  std::enable_if_t<std::is_convertible_v<S, const typename MatrixTraits<V>::Scalar>, int> = 0>
  inline auto operator/(V&& v, S scale)
  {
    using Sc = typename MatrixTraits<V>::Scalar;
    auto ret = MatrixTraits<V>::make(base_matrix(std::forward<V>(v)) / static_cast<Sc>(scale));
    if constexpr (not std::is_lvalue_reference_v<V&&>) return strict(std::move(ret)); else return ret;
  }


  /// Multiply a typed matrix by another typed matrix. The result is a regular typed matrix unless the first operand is EuclideanMean.
  template<
    typename V1,
    typename V2,
    std::enable_if_t<is_typed_matrix_v<V1> and is_typed_matrix_v<V2>, int> = 0>
  inline auto operator*(V1&& v1, V2&& v2)
  {
    static_assert(is_equivalent_v<typename MatrixTraits<V1>::ColumnCoefficients, typename MatrixTraits<V2>::RowCoefficients>);
    static_assert(MatrixTraits<V1>::columns == MatrixTraits<V2>::dimension);
    using RC = typename MatrixTraits<V1>::RowCoefficients;
    using CC = typename MatrixTraits<V2>::ColumnCoefficients;
    auto b = base_matrix(std::forward<V1>(v1)) * base_matrix(std::forward<V2>(v2));
    using CommonV = std::decay_t<std::conditional_t<is_Euclidean_mean_v<V1>,
      V1, TypedMatrix<RC, CC, decltype(b)>>>;
    auto ret = MatrixTraits<CommonV>::template make<RC, CC>(std::move(b));
    if constexpr (not std::is_lvalue_reference_v<V1&&> or not std::is_lvalue_reference_v<V2&&>) return strict(std::move(ret)); else return ret;
  }


  /// Negate a vector object. The result is a regular typed matrix unless the operand is EuclideanMean.
  template<
    typename V,
    std::enable_if_t<is_typed_matrix_v<V>, int> = 0>
  inline auto operator-(V&& v)
  {
    using RC = typename MatrixTraits<V>::RowCoefficients;
    using CC = typename MatrixTraits<V>::ColumnCoefficients;
    using Res = std::decay_t<std::conditional_t<is_Euclidean_mean_v<V>,
      V, TypedMatrix<RC, CC, typename MatrixTraits<V>::BaseMatrix>>>;
    auto ret = MatrixTraits<Res>::make(-base_matrix(std::forward<V>(v)));
    if constexpr (not std::is_lvalue_reference_v<V&&>) return strict(std::move(ret)); else return ret;
  }


  /// Equality operator.
  template<typename V1, typename V2, std::enable_if_t<is_typed_matrix_v<V1> and is_typed_matrix_v<V2>, int> = 0>
  constexpr auto operator==(V1&& v1, V2&& v2)
  {
    if constexpr(
      is_equivalent_v<typename MatrixTraits<V1>::RowCoefficients, typename MatrixTraits<V2>::RowCoefficients> and
      is_equivalent_v<typename MatrixTraits<V1>::ColumnCoefficients, typename MatrixTraits<V2>::ColumnCoefficients>)
    {
      return strict_matrix(std::forward<V1>(v1)) == strict_matrix(std::forward<V2>(v2));
    }
    else
    {
      return false;
    }
  }


  /// Inequality operator.
  template<typename V1, typename V2, std::enable_if_t<is_typed_matrix_v<V1> and is_typed_matrix_v<V2>, int> = 0>
  constexpr auto operator!=(V1&& v1, V2&& v2)
  {
    return not (std::forward<V1>(v1) == std::forward<V2>(v2));
  }


}

#endif //OPENKALMAN_TYPEDMATRIXOVERLOADS_