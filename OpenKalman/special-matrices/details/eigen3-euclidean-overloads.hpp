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
 * \file
 * \brief Overloaded functions relating to Eigen3::euclidean_expr types
 */

#ifndef OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP

namespace OpenKalman::Eigen3
{
  // ----------- //
  //  Overloads  //
  // ----------- //

  /**
   * \tparam Arg A Euclidean expression
   * \param index The index of the row
   * \return The row at the specified index.
   */
#ifdef __cpp_concepts
  template<std::size_t...index, euclidean_expr Arg, std::convertible_to<const std::size_t>...runtime_index_t>
  requires (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((index + ... + 0) < row_extent_of_v<Arg>))
#else
  template<std::size_t...index, typename Arg, typename...runtime_index_t, std::enable_if_t<euclidean_expr<Arg> and
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((index + ... + 0) < row_extent_of<Arg>::value)), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg, runtime_index_t...i)
  {
    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (uniform_coefficients<typename MatrixTraits<Arg>::RowCoefficients> and
      MatrixTraits<Arg>::RowCoefficients::dimensions == MatrixTraits<Arg>::RowCoefficients::euclidean_dimensions)
    {
      using RC = typename has_uniform_coefficients<typename MatrixTraits<Arg>::RowCoefficients>::common_coefficient;

      if constexpr (from_euclidean_expr<Arg>)
        return from_euclidean<RC>(row<index...>(nested_matrix(std::forward<Arg>(arg)), i...));
      else
        return to_euclidean<RC>(row<index...>(nested_matrix(std::forward<Arg>(arg)), i...));
    }
    else
    {
      return row<index...>(make_native_matrix(std::forward<Arg>(arg)), i...);
    }
  }


  /**
   * \tparam index The index of the column
   * \tparam Arg A Euclidean expression
   * \return The column at the specified index.
   */
#ifdef __cpp_concepts
  template<std::size_t...index, euclidean_expr Arg, std::convertible_to<const std::size_t>...runtime_index_t>
    requires (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((index + ... + 0) < column_extent_of_v<Arg>))
#else
  template<std::size_t...index, typename Arg, typename...runtime_index_t, std::enable_if_t<euclidean_expr<Arg> and
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((index + ... + 0) < column_extent_of<Arg>::value)), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg, runtime_index_t...i)
  {
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using RC = typename MatrixTraits<Arg>::RowCoefficients;

      if constexpr (from_euclidean_expr<Arg>)
        return from_euclidean<RC>(column<index...>(nested_matrix(std::forward<Arg>(arg)), i...));
      else
        return to_euclidean<RC>(column<index...>(nested_matrix(std::forward<Arg>(arg)), i...));
    }
  }


  /// Special case for converting a matrix to Euclidean form. This is a shortcut.
  /// Returns the nested matrix of the argument, because ToEuclideanExpr<FromEuclideanExpr<M>> reduces to M.
#ifdef __cpp_concepts
  template<from_euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, from_euclidean_expr Arg> requires
    equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    from_euclidean_expr<Arg> and equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    if constexpr (Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return nested_matrix(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<to_euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(Coefficients::axes_only)
    {
      return nested_matrix(std::forward<Arg>(arg));
    }
    else
    {
      return FromEuclideanExpr<Coefficients, Arg> {std::forward<Arg>(arg)};
    }
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, to_euclidean_expr Arg> requires
    equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    to_euclidean_expr<Arg> and equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    return from_euclidean(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, from_euclidean_expr Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    from_euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr( MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return to_diagonal(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      return to_diagonal(make_native_matrix(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return diagonal_of(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      return std::forward<Arg>(arg).diagonal();
    }
  }


} // namespace OpenKalman::Eigen3


namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
    element_gettable<nested_matrix_of_t<T>, I, Is...>
  struct GetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct GetElement<T, std::enable_if_t<euclidean_expr<T> and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_gettable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i, Is...is)
    {
      if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, is...);
      }
      else
      {
        using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
            return get_element(nested_matrix(std::forward<Arg>(arg)), row, is...);
        };
        if constexpr(to_euclidean_expr<Arg>)
        {
          return OpenKalman::internal::to_euclidean_coeff<Coeffs>(i, get_coeff);
        }
        else
        {
          return OpenKalman::internal::from_euclidean_coeff<Coeffs>(i, get_coeff);
        }
      }
    }
  };


#ifdef __cpp_concepts
  template<from_euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and to_euclidean_expr<nested_matrix_of_t<T>> and
    element_gettable<nested_matrix_of_t<T>, I, Is...>
  struct GetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct GetElement<T, std::enable_if_t<from_euclidean_expr<T> and to_euclidean_expr<nested_matrix_of_t<T>> and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_gettable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i, Is...is)
    {
      if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), i, is...);
      }
      else
      {
        using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
          return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), row, is...);
        };
        return OpenKalman::internal::wrap_get<Coeffs>(i, get_coeff);
      }
    }
  };


#ifdef __cpp_concepts
  template<euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
      MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_of_t<T>, I, Is...>
  struct SetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct SetElement<T, std::enable_if_t<euclidean_expr<T> and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    MatrixTraits<T>::RowCoefficients::axes_only and
    (sizeof...(Is) <= 1) and element_settable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    template<typename Arg, typename Scalar>
    static constexpr auto set(Arg&& arg, const Scalar s, I i, Is...is)
    {
      set_element(nested_matrix(arg), s, i, is...);
    }
  };


#ifdef __cpp_concepts
  template<from_euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and to_euclidean_expr<nested_matrix_of_t<T>> and
      element_settable<nested_matrix_of_t<T>, I, Is...>
  struct SetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct SetElement<T, std::enable_if_t<from_euclidean_expr<T> and to_euclidean_expr<nested_matrix_of_t<T>> and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_settable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    /**
     * \brief Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
     * \details This function sets the nested matrix, not the wrapped resulting matrix.
     * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
     * single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
     * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
     * \tparam Arg The matrix to set.
     * \tparam Scalar The value to set the coefficient to.
     * \param i The row of the coefficient.
     * \param j The column of the coefficient.
     */
    template<typename Arg, typename Scalar>
    static constexpr auto set(Arg&& arg, const Scalar s, I i, Is...is)
    {
      if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        set_element(nested_matrix(nested_matrix(arg)), s, i, is...);
      }
      else
      {
        using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
          return get_element(nested_matrix(nested_matrix(arg)), row, is...);
        };
        const auto set_coeff = [&arg, is...] (const std::size_t row, const Scalar value) {
          set_element(nested_matrix(nested_matrix(arg)), value, row, is...);
        };
        OpenKalman::internal::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
      }
    }
  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct ElementWiseOperations<T>
#else
  template<typename T>
  struct ElementWiseOperations<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

    template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
    static constexpr auto fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
    {
      return OpenKalman::fold<order>(b, std::forward<Accum>(accum), make_native_matrix(std::forward<Arg>(arg)));
    }

  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {
  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct LinearAlgebra<T>
#else
  template<typename T>
  struct LinearAlgebra<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

    template<typename Arg>
    static constexpr decltype(auto) conjugate(Arg&& arg) noexcept
    {
      if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).conjugate(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr decltype(auto) transpose(Arg&& arg) noexcept
    {
      if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).transpose(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr decltype(auto) adjoint(Arg&& arg) noexcept
    {
      if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).adjoint(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) noexcept
    {
      if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return OpenKalman::determinant(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return arg.determinant(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr auto trace(Arg&& arg) noexcept
    {
      if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
      {
        return OpenKalman::trace(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return arg.trace(); //< \todo Generalize this.
      }
    }


#ifdef __cpp_concepts
    template<TriangleType t, typed_matrix A, typename U, typename Alpha> requires
      equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>
#else
    template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
      typed_matrix<A> and typed_matrix<U> and
      equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>, int> = 0>
#endif
    static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_self_adjoint<t>(make_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
    }


#ifdef __cpp_concepts
    template<TriangleType t, typed_matrix A, typename U, typename Alpha> requires
      equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>
#else
    template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
      typed_matrix<A> and typed_matrix<U> and
      equivalent_to<typename MatrixTraits<A>::RowCoefficients, typename MatrixTraits<U>::RowCoefficients>, int> = 0>
#endif
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_triangular<t>(make_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
    }

  };

} // namespace OpenKalman::interface


namespace OpenKalman::Eigen3
{

  /// Solves AX = B for X (A is a regular matrix type, and B is a Euclidean expression).
  /// A must be invertible. (Does not check.)
#ifdef __cpp_concepts
  template<euclidean_expr A, eigen_matrix B> requires square_matrix<A>
#else
  template<typename A, typename B, std::enable_if_t<euclidean_expr<A> and square_matrix<A> and
    eigen_matrix<B>, int> = 0>
#endif
  inline auto solve(A&& a, B&& b) noexcept
  {
    static_assert(row_extent_of_v<A> == row_extent_of_v<B>);
    return solve(make_native_matrix(std::forward<A>(a)), std::forward<B>(b));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg) noexcept
  {
    return make_native_matrix(reduce_columns(make_native_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_rows(Arg&& arg) noexcept
  {
    return make_native_matrix(reduce_rows(make_native_matrix(std::forward<Arg>(arg))));
  }


/**
 * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
 * Returns L as a triangular matrix.
 */
#ifdef __cpp_concepts
  template<euclidean_expr A>
#else
  template<typename A, std::enable_if_t<euclidean_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    return LQ_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline auto
  QR_decomposition(Arg&& arg)
  {
    return QR_decomposition(make_native_matrix(std::forward<Arg>(arg)));
  }


  /// Concatenate one or more EuclideanExpr objects vertically.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      constexpr auto cols = column_extent_of_v<V>;
      static_assert(((cols == column_extent_of_v<Vs>) and ...));
      using C = Concatenate<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients...>;
      return MatrixTraits<V>::template make<C>(
        concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /// Concatenate one or more EuclideanExpr objects horizontally.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using C = typename MatrixTraits<V>::RowCoefficients;
      static_assert(std::conjunction_v<std::is_same<C, typename MatrixTraits<Vs>::RowCoefficients>...>);
      return MatrixTraits<V>::template make<C>(
        concatenate_horizontal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace internal
  {
    template<typename G, typename Expr, typename CC>
    struct SplitEuclideanVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };


    template<typename G, typename Expr, typename RC>
    struct SplitEuclideanHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };


    template<typename G, typename Expr>
    struct SplitEuclideanDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };
  } // internal


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, coefficients...Cs, euclidean_expr Arg> requires (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    using CC = Axes<column_extent_of_v<Arg>>;
    return split_vertical<internal::SplitEuclideanVertF<F, Arg, CC>, from_euclidean_expr<Arg>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, bool, coefficients...Cs, euclidean_expr Arg> requires (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg> requires
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions vertically.
   * \details The expression is evaluated to a self_contained matrix first.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Number of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires
    ((cut + ... + cuts) <= row_extent_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and ((cut + ... + cuts) <= row_extent_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    if constexpr(cut == row_extent_of_v<Arg> and sizeof...(cuts) == 0)
    {
      return std::tuple<Arg> {std::forward<Arg>(arg)};
    }
    else
    {
      return split_vertical<cut, cuts...>(make_native_matrix(std::forward<Arg>(arg)));
    }
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<typename F, coefficients...Cs, euclidean_expr Arg> requires
    (not coefficients<F>) and ((0 + ... + Cs::dimensions) <= column_extent_of_v<Arg>)
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not coefficients<F>) and ((0 + ... + Cs::dimensions) <= column_extent_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    return split_horizontal<internal::SplitEuclideanHorizF<F, Arg, RC>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions horizontally.
   * \tparam cut Number of columns in the first cut.
   * \tparam cuts Number of columns in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires
    ((cut + ... + cuts) <= column_extent_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    ((cut + ... + cuts) <= column_extent_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
   */
#ifdef __cpp_concepts
  template<typename F, coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg> and (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<internal::SplitEuclideanDiagF<F, Arg>, from_euclidean_expr<Arg>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<typename F, bool, coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    square_matrix<Arg> and (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg> and
    prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg> and
    (coefficients<Cs> and ...) and prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<OpenKalman::internal::default_split_function, false, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions diagonally.
   * |details The expression (which must be square) is evaluated to a self_contained matrix first.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Number of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires square_matrix<Arg> and
    ((cut + ... + cuts) <= row_extent_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and square_matrix<Arg> and ((cut + ... + cuts) <= row_extent_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    if constexpr(cut == row_extent_of_v<Arg> and sizeof...(cuts) == 0)
    {
      return std::tuple<Arg> {std::forward<Arg>(arg)};
    }
    else
    {
      return split_diagonal<cut, cuts...>(make_native_matrix(std::forward<Arg>(arg)));
    }
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg An lvalue reference to the matrix.
   * \tparam Function The function, which takes a column and returns a column.
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col) { f(col); } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    std::is_invocable_v<Function, decltype(column(std::declval<Arg&>(), 0))&> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(std::move(col));
      f(mc);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(f_nested, c);
    return arg;
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg An lvalue reference to the matrix.
   * \tparam Function The function, which takes a column and an index and returns a column.
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col, std::size_t i) {
      f(col, i);
    } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    std::is_invocable_v<Function, decltype(column(std::declval<Arg&>(), 0))&, std::size_t> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(std::move(col));
      f(mc, i);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(f_nested, c);
    return arg;
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg A constant lvalue reference or rvalue reference to the matrix.
   * \tparam Function The function, which takes a column and returns a column.
   * \todo add situation where Arg is a native matrix but the function result is a euclidean_expr
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Arg& arg, const Function& f) { {f(column(arg, 0))} -> column_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    column_vector<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto&& col) -> auto {
      return make_self_contained(f(MatrixTraits<Arg>::template make<Coefficients>(std::forward<decltype(col)>(col))));
    };
    return apply_columnwise(f_nested, nested_matrix(arg));
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg A constant lvalue reference or rvalue reference to the matrix.
   * \tparam Function The function, which takes a column and an index and returns a column.
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Arg& arg, const Function& f, std::size_t i) { {f(column(arg, 0), i)} -> column_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    column_vector<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto&& col, std::size_t i) -> auto {
      return make_self_contained(
        f(MatrixTraits<Arg>::template make<Coefficients>(std::forward<decltype(col)>(col)), i));
    };
    return apply_columnwise(f_nested, nested_matrix(arg));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) { {f()} -> euclidean_expr; {f()} -> column_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function>> and column_vector<std::invoke_result_t<Function>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function>;
    const auto f_nested = [&f] () -> auto { return make_self_contained(nested_matrix(f())); };
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) { {f(i)} -> euclidean_expr; {f(i)} -> column_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function, std::size_t>> and
    column_vector<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    const auto f_nested = [&f](std::size_t i) -> auto { return make_self_contained(nested_matrix(f(i))); };
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_nested));
  }


  /**
   * \brief Apply a function to each row of a matrix.
   * \tparam Arg A constant lvalue reference or rvalue reference to the matrix.
   * \tparam Function The function, which takes a row (and optionally an index) and returns a row.
   */
  #ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Function& f, const Arg& arg) { {f(row(arg, 0))} -> row_vector; } or
    requires(const Function& f, const Arg& arg, std::size_t i) { {f(row(arg, 0), i)} -> row_vector; }
  #else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    row_vector<std::invoke_result_t<Function, std::decay_t<decltype(row(std::declval<const Arg&>(), 0))>&&>>, int> = 0>
  inline auto
  apply_rowwise(const Function& f, Arg&& arg)
  {
    return apply_rowwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }

  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    row_vector<std::invoke_result_t<Function,
      std::decay_t<decltype(row(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
  #endif
  inline auto
  apply_rowwise(const Function& f, Arg&& arg)
  {
    return apply_rowwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


  #ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) { {f()} -> euclidean_expr; {f()} -> row_vector; }
  #else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function>> and row_vector<std::invoke_result_t<Function>>, int> = 0>
  #endif
  inline auto
  apply_rowwise(const Function& f)
  {
    const auto f_nested = [&f] () -> auto { return make_native_matrix(f()); };
    return apply_rowwise<count>(f_nested);
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) { {f(i)} -> euclidean_expr; {f(i)} -> row_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function, std::size_t>> and
      row_vector<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f)
  {
    const auto f_nested = [&f](std::size_t i) -> auto { return make_native_matrix(f(i)); };
    return apply_rowwise<count>(f_nested);
  }


#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(Function& f, scalar_type_of_t<Arg>& s) {
      {f(s)} -> std::convertible_to<const scalar_type_of_t<Arg>>;
    } or
    requires(Function& f, scalar_type_of_t<Arg>& s, std::size_t& i, std::size_t& j) {
      {f(s, i, j)} -> std::convertible_to<const scalar_type_of_t<Arg>>;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and std::is_convertible_v<
    std::invoke_result_t<Function&, typename scalar_type_of<Arg>::type&>,
    const typename scalar_type_of<Arg>::type>, int> = 0>
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    return apply_coefficientwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and std::is_convertible_v<
    std::invoke_result_t<Function&, typename scalar_type_of<Arg>::type&, std::size_t&, std::size_t&>,
    const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    return apply_coefficientwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \brief Fill a matrix of to-Euclidean- or from_euclidean-transformed values selected from a random distribution.
   * \details The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
#ifdef __cpp_concepts
  template<euclidean_expr ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename...Dists>
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<euclidean_expr<ReturnType>, int> = 0>
#endif
  inline auto
  randomize(Dists...dists)
  {
    using B = nested_matrix_of_t<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }

} // namespace OpenKalman::eigen3

#endif //OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP
