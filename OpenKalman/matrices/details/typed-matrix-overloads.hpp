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

#include <iostream>

namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<typed_matrix T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    element_gettable<nested_matrix_of_t<Arg>, I...>
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, I..., std::enable_if_t<typed_matrix<T> and element_gettable<nested_matrix_of_t<Arg>, I...> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>)>>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I...i)
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i...);
    }
  };


#ifdef __cpp_concepts
  template<typed_matrix T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and element_settable<nested_matrix_of_t<T>, I, Is...>
  struct SetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...I>
  struct SetElement<T, I, Is..., std::enable_if_t<typed_matrix<T> and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_settable<nested_matrix_of_t<T>, I, Is...>>>
#endif
  {
    template<typename Arg, typename Scalar>
    static constexpr auto set(Arg&& arg, const Scalar s, I i, Is...is)
    {
      if constexpr(wrapped_mean<Arg>)
      {
        using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
          return get_element(nested_matrix(arg), row, is...);
        };
        const auto set_coeff = [&arg, is...](const std::size_t row, const scalar_type_of_t<Arg> value) {
          set_element(nested_matrix(arg), value, row, is...);
        };
        oin::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
      }
      else
      {
        set_element(nested_matrix(arg), s, i, is...);
      }
    }
  };

} // namespace OpenKalman::interface


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires uniform_coefficients<typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    uniform_coefficients<typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t index)
  {
    using RC = typename has_uniform_coefficients<typename MatrixTraits<Arg>::RowCoefficients>::common_coefficient;
    using CC = typename MatrixTraits<Arg>::ColumnCoefficients;
    return MatrixTraits<Arg>::template make<RC, CC>(column(nested_matrix(std::forward<Arg>(arg)), index));
  }


#ifdef __cpp_concepts
  template<std::size_t index, typed_matrix Arg> requires (index < row_dimension_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (index < row_dimension_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  row(Arg&& arg)
  {
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


#ifdef __cpp_concepts
  template<typed_matrix Arg> requires uniform_coefficients<typename MatrixTraits<Arg>::ColumnCoefficients>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    uniform_coefficients<typename MatrixTraits<Arg>::ColumnCoefficients>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    using CC = typename has_uniform_coefficients<typename MatrixTraits<Arg>::ColumnCoefficients>::common_coefficient;
    return MatrixTraits<Arg>::template make<RC, CC>(column(nested_matrix(std::forward<Arg>(arg)), index));
  }


#ifdef __cpp_concepts
  template<std::size_t index, typed_matrix Arg> requires (index < column_dimension_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (index < column_dimension_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  column(Arg&& arg)
  {
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


  namespace interface
  {

  #ifdef __cpp_concepts
    template<typed_matrix T>
    struct ElementAccess<T>
  #else
    template<typename T>
    struct ElementAccess<T, std::enable_if_t<typed_matrix<T>>>
  #endif
    {
    };


#ifdef __cpp_concepts
    template<typed_matrix T>
    struct ElementWiseOperations<T>
#else
    template<typename T>
    struct ElementWiseOperations<T, std::enable_if_t<typed_matrix<T>>>
#endif
    {

      template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
      static constexpr auto fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
      {
        return OpenKalman::fold<order>(b, std::forward<Accum>(accum), nested_matrix(std::forward<Arg>(arg)));
      }

    };


#ifdef __cpp_concepts
  template<typed_matrix T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<typed_matrix<T>>>
#endif
  {

    template<typename Arg>
    static auto
    to_diagonal(Arg&& arg) noexcept
    {
      using C = typename MatrixTraits<Arg>::RowCoefficients;
      auto b = to_diagonal(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<C, C, decltype(b)>(std::move(b));
    }


    template<typename Arg>
    static auto
    diagonal_of(Arg&& arg) noexcept
    {
      using C = typename MatrixTraits<Arg>::RowCoefficients;
      auto b = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<C, Axis, decltype(b)>(std::move(b));
    }

  };


#ifdef __cpp_concepts
    template<typed_matrix T>
    struct LinearAlgebra<T>
#else
    template<typename T>
    struct linearAlgebra<T, std::enable_if_t<typed_matrix<T>>>
#endif
    {

      template<typename Arg>
      static constexpr auto conjugate(Arg&& arg) noexcept
      {
        using CRows = typename MatrixTraits<Arg>::RowCoefficients;
        using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
        if constexpr(euclidean_transformed<Arg>)
        {
          auto b = OpenKalman::conjugate(nested_matrix(from_euclidean(std::forward<Arg>(arg))));
          return Matrix<CRows, CCols, decltype(b)>(std::move(b));
        }
        else
        {
          auto b = OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg)));
          return Matrix<CRows, CCols, decltype(b)>(std::move(b));
        }
      }


      template<typename Arg>
      static constexpr auto transpose(Arg&& arg) noexcept
      {
        using CRows = typename MatrixTraits<Arg>::RowCoefficients;
        using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
        if constexpr(euclidean_transformed<Arg>)
        {
          auto b = OpenKalman::transpose(nested_matrix(from_euclidean(std::forward<Arg>(arg))));
          return Matrix<CCols, CRows, decltype(b)>(std::move(b));
        }
        else
        {
          auto b = OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
          return Matrix<CCols, CRows, decltype(b)>(std::move(b));
        }
      }


      template<typename Arg>
      static constexpr auto adjoint(Arg&& arg) noexcept
      {
        using CRows = typename MatrixTraits<Arg>::RowCoefficients;
        using CCols = typename MatrixTraits<Arg>::ColumnCoefficients;
        if constexpr(euclidean_transformed<Arg>)
        {
          auto b = OpenKalman::adjoint(nested_matrix(from_euclidean(std::forward<Arg>(arg))));
          return Matrix<CCols, CRows, decltype(b)>(std::move(b));
        }
        else
        {
          auto b = OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg)));
          return Matrix<CCols, CRows, decltype(b)>(std::move(b));
        }
      }


      template<typename Arg>
      static constexpr auto determinant(Arg&& arg) noexcept
      {
        return OpenKalman::determinant(nested_matrix(std::forward<Arg>(arg)));
      }


      template<typename Arg>
      static constexpr auto trace(Arg&& arg) noexcept
      {
        return OpenKalman::trace(nested_matrix(std::forward<Arg>(arg)));
      }


    #ifdef __cpp_concepts
      template<TriangleType t, typed_matrix Arg, typed_matrix U, typename Alpha> requires
        equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>
    #else
      template<typename Arg, typename U, typename Alpha, std::enable_if_t<typed_matrix<Arg> and typed_matrix<U> and
        equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>, int> = 0>
    #endif
      static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_self_adjoint<t>(
          nested_matrix(std::forward<A>(a)), nested_matrix(std::forward<U>(u)), alpha);
      }


  #ifdef __cpp_concepts
      template<TriangleType t, typed_matrix Arg, typed_matrix U, typename Alpha> requires
        equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>
  #else
      template<typename Arg, typename U, typename Alpha, std::enable_if_t<typed_matrix<Arg> and typed_matrix<U> and
        equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>, int> = 0>
  #endif
      static decltype(auto) rank_update_triangular(Arg&& arg, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_triangular<t>(
          nested_matrix(std::forward<A>(a)), nested_matrix(std::forward<U>(u)), alpha);
      }

    };

  } // namespace interface


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
  template<typed_matrix Arg> requires (column_dimension_of_v<Arg> == 1) or untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    ((column_dimension_of<Arg>::value == 1) or untyped_columns<Arg>), int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    /// \todo add an option where all the column coefficients are the same, but not Axis.
    using C = typename MatrixTraits<Arg>::RowCoefficients;

    if constexpr(column_dimension_of_v<Arg> == 1)
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
    ((... + cuts) <= column_dimension_of_v<M>)
#else
  template<std::size_t ... cuts, typename M,
    std::enable_if_t<typed_matrix<M> and untyped_columns<M> and (sizeof...(cuts) > 0) and
      ((0 + ... + cuts) <= column_dimension_of<M>::value), int> = 0>
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


  ////

#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires untyped_columns<Arg> and
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col) { f(col); } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    std::is_invocable_v<const Function&,
      std::decay_t<decltype(column(std::declval<Arg&>(), 0))>& > and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
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
    apply_columnwise(f_nested, c);
    if constexpr(wrapped_mean<Arg>) c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires untyped_columns<Arg> and
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col, std::size_t i) {
      f(col, i);
    } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    std::is_invocable_v<Function,
    std::decay_t<decltype(column(std::declval<Arg&>(), 0))>&, std::size_t> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<RC, Axis>(std::move(col));
      f(mc, i);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(f_nested, c);
    if constexpr(wrapped_mean<Arg>) c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires untyped_columns<Arg> and
    requires(const Arg& arg, const Function& f) {
      {f(column<0>(arg))} -> typed_matrix;
      {f(column<0>(arg))} -> column_vector;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    typed_matrix<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&& >>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::dimension == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, column_dimension_of_v<Arg>>;
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto&& col) -> auto {
      return make_self_contained(
        nested_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(std::forward<decltype(col)>(col)))));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(f_nested, nested_matrix(arg)));
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires untyped_columns<Arg> and
    requires(const Arg& arg, const Function& f, std::size_t i) {
      {f(column<0>(arg), i)} -> typed_matrix;
      {f(column<0>(arg), i)} -> column_vector;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg> and
    typed_matrix<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0)), std::size_t>;
    using ResRC = typename MatrixTraits<ResultType>::RowCoefficients;
    using ResCC0 = typename MatrixTraits<ResultType>::ColumnCoefficients;
    static_assert(ResCC0::dimension == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = Replicate<ResCC0, column_dimension_of_v<Arg>>;
    const auto f_nested = [&f](auto&& col, std::size_t i) -> auto {
      using RC = typename MatrixTraits<Arg>::RowCoefficients;
      return make_self_contained(
        nested_matrix(f(MatrixTraits<Arg>::template make<RC, Axis>(std::forward<decltype(col)>(col)), i)));
    };
    return MatrixTraits<ResultType>::template make<ResRC, ResCC>(apply_columnwise(f_nested, nested_matrix(arg)));
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
    static_assert(CC0::dimension == 1, "Function argument of apply_columnwise must return a column vector.");
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
    static_assert(CC0::dimension == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = Replicate<CC0, count>;
    return MatrixTraits<ResultType>::template make<RC, CC>(apply_columnwise<count>(f_nested));
  }


  ////

#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<scalar_type_of_t<Arg>>&>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<scalar_type_of_t<Arg>>&>>, int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(const Function& f, Arg& arg)
  {
    apply_coefficientwise(f, nested_matrix(arg));
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(wrapped_mean<Arg>)
      nested_matrix(arg) = wrap_angles<RC>(nested_matrix(arg));
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires std::is_void_v<std::invoke_result_t<const Function&,
    std::decay_t<scalar_type_of_t<Arg>>&, std::size_t, std::size_t>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<scalar_type_of_t<Arg>>&, std::size_t, std::size_t>>, int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(const Function& f, Arg& arg)
  {
    apply_coefficientwise(f, nested_matrix(arg));
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(wrapped_mean<Arg>)
      nested_matrix(arg) = wrap_angles<RC>(nested_matrix(arg));
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires
    std::convertible_to<std::invoke_result_t<const Function&, std::decay_t<scalar_type_of_t<Arg>>>,
      scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<const Function&, std::decay_t<typename scalar_type_of<Arg>::type>>,
      typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, const Arg& arg)
  {
    return MatrixTraits<Arg>::make(apply_coefficientwise(f, nested_matrix(arg)));
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires std::convertible_to<std::invoke_result_t<const Function&,
      std::decay_t<scalar_type_of_t<Arg>>, std::size_t, std::size_t>, scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<const Function&,
      std::decay_t<typename scalar_type_of<Arg>::type>, std::size_t, std::size_t>,
      typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, const Arg& arg)
  {
    return MatrixTraits<Arg>::make(apply_coefficientwise(f, nested_matrix(arg)));
  }


#ifdef __cpp_concepts
  template<typed_matrix V, typename Function> requires
    std::convertible_to<std::invoke_result_t<const Function&>, scalar_type_of_t<V>>
#else
  template<typename V, typename Function, std::enable_if_t<typed_matrix<V> and
    std::is_convertible_v<std::invoke_result_t<const Function&>, typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f)
  {
    constexpr auto rows = row_dimension_of_v<V>;
    constexpr auto columns = column_dimension_of_v<V>;
    using Scalar = scalar_type_of_t<V>;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function>>,
      std::decay_t<scalar_type_of_t<V>>>)
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
    std::invoke_result_t<const Function&, std::size_t, std::size_t>, scalar_type_of_t<V>>
#else
  template<typename V, typename Function, std::enable_if_t<typed_matrix<V> and std::is_convertible_v<
    std::invoke_result_t<const Function&, std::size_t, std::size_t>, typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Function&& f)
  {
    constexpr auto rows = row_dimension_of_v<V>;
    constexpr auto columns = column_dimension_of_v<V>;
    using Scalar = scalar_type_of_t<V>;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function, std::size_t, std::size_t>>,
      std::decay_t<scalar_type_of_t<V>>>)
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
  requires (not any_dynamic_dimension<ReturnType>) and (sizeof...(Dists) > 0) and
    (((requires { typename std::decay_t<Dists>::result_type;  typename std::decay_t<Dists>::param_type; } or
      std::is_arithmetic_v<std::decay_t<Dists>>) and ... )) and
    ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
    (sizeof...(Dists) == 1 or row_dimension_of_v<ReturnType> * column_dimension_of_v<ReturnType> == sizeof...(Dists) or
      row_dimension_of_v<ReturnType> == sizeof...(Dists) or column_dimension_of_v<ReturnType> == sizeof...(Dists))
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<typed_matrix<ReturnType> and (not any_dynamic_dimension<ReturnType>) and (sizeof...(Dists) > 0) and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
      (sizeof...(Dists) == 1 or
        row_dimension_of<ReturnType>::value * column_dimension_of<ReturnType>::value == sizeof...(Dists) or
        row_dimension_of<ReturnType>::value == sizeof...(Dists) or
        column_dimension_of<ReturnType>::value == sizeof...(Dists)), int> = 0>
#endif
  inline auto
  randomize(Dists&& ... dists)
  {
    using B = nested_matrix_of_t<ReturnType>;
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
  requires any_dynamic_dimension<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
      typed_matrix<ReturnType> and any_dynamic_dimension<ReturnType> and
      (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    if constexpr (not dynamic_rows<ReturnType>) assert(rows == row_dimension_of_v<ReturnType>);
    if constexpr (not dynamic_columns<ReturnType>) assert(columns == column_dimension_of_v<ReturnType>);
    using B = nested_matrix_of_t<ReturnType>;
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
    os << make_dense_writable_matrix_from(v);
    return os;
  }


}

#endif //OPENKALMAN_TYPED_MATRIX_OVERLOADS_H