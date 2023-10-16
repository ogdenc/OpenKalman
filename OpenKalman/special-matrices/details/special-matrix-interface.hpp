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
 * \brief Interface for DiagonalMatrix, TriangularMatrix, and SelfAdjointMatrix.
 */

#ifndef OPENKALMAN_SPECIAL_MATRIX_INTERFACE_HPP
#define OPENKALMAN_SPECIAL_MATRIX_INTERFACE_HPP

#include <complex>


namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<typename T> requires eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct library_interface<T>
#else
  template<typename T>
  struct library_interface<T, std::enable_if_t<eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
    : library_interface<std::decay_t<nested_matrix_of_t<T>>>
  {
  private:

    using Nested = std::decay_t<nested_matrix_of_t<T>>;
    using Base = library_interface<Nested>;

  public:

    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, Nested>;


    // to_native_matrix inherited

    // make_default inherited

    // fill_with_elements inherited

    // make_constant_matrix inherited

    // make_identity_matrix inherited


    template<typename Arg, typename...Begin, typename...Size>
    static auto
    get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      auto dense = make_dense_writable_matrix_from<scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
      static_assert(not eigen_diagonal_expr<decltype(dense)> and not eigen_triangular_expr<decltype(dense)> and
        not eigen_self_adjoint_expr<decltype(dense)>);
      return OpenKalman::get_block(std::move(dense), begin, size);
    }


    template<typename Arg, typename Block, typename...Begin>
    static constexpr Arg&
    set_block(Arg& arg, Block&& block, Begin...begin) = delete;


    template<TriangleType t, typename A, typename B>
    static A& set_triangle(A& a, B&& b)
    {
      if constexpr (eigen_triangular_expr<A>)
      {
        return OpenKalman::internal::set_triangle<t>(nested_matrix(a), std::forward<B>(b));
      }
      else if constexpr (eigen_self_adjoint_expr<A>)
      {
        if constexpr ((t == TriangleType::upper and not hermitian_adapter<A, HermitianAdapterType::upper>) or
          (t == TriangleType::lower and not hermitian_adapter<A, HermitianAdapterType::lower>))
          OpenKalman::internal::set_triangle<t>(nested_matrix(a), adjoint(std::forward<B>(b)));
        else
          OpenKalman::internal::set_triangle<t>(nested_matrix(a), std::forward<B>(b));
      }
      else
      {
        static_assert(eigen_diagonal_expr<A>);
        static_assert(diagonal_matrix<B>);
        nested_matrix(a) = diagonal_of(std::forward<B>(b));
      }
      return a;
    }


    template<typename Arg>
    static decltype(auto)
    to_diagonal(Arg&& arg) noexcept
    {
      // Note: the interface only needs to handle constant and dynamic-sized zero matrices.
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }


    template<typename Arg>
    static decltype(auto)
    diagonal_of(Arg&& arg) noexcept
    {
      // Note: the global diagonal_of function already handles all zero and constant cases.
      if constexpr (eigen_diagonal_expr<Arg>) return nested_matrix(std::forward<Arg>(arg));
      else return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg)));
    }


    // replicate inherited


    template<typename...Ds, typename Operation, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
    {
      using Traits = library_interface<std::decay_t<nested_matrix_of_t<T>>>;
      return Traits::n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto)
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      using Traits = library_interface<std::decay_t<nested_matrix_of_t<T>>>;
      return Traits::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }

    // to_euclidean not defined
    // from_euclidean not defined
    // wrap_angles not defined

    template<typename Arg>
    static constexpr decltype(auto) conjugate(Arg&& arg) noexcept
    {
      if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        constexpr auto t = hermitian_adapter_type_of_v<Arg>;
        return make_hermitian_matrix<t>(OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg))));
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);
        constexpr auto t = triangle_type_of_v<Arg>;
        return make_triangular_matrix<t>(OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg))));
      }
      // Global conjugate function already handles DiagonalMatrix
    }


    template<typename Arg>
    static constexpr decltype(auto) transpose(Arg&& arg) noexcept
    {
      if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (hermitian_matrix<nested_matrix_of_t<Arg>>)
        {
          return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = (hermitian_adapter<Arg, HermitianAdapterType::lower> ? TriangleType::upper : TriangleType::lower);
          return make_hermitian_matrix<t>(OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg))));
        }
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);
        if constexpr (triangular_matrix<nested_matrix_of_t<Arg>>)
        {
          return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = triangular_matrix<Arg, TriangleType::lower> ? TriangleType::upper : TriangleType::lower;
          return make_triangular_matrix<t>(OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg))));
        }
      }
      // Global transpose function already handles DiagonalMatrix
    }


    template<typename Arg>
    static constexpr decltype(auto) adjoint(Arg&& arg) noexcept
    {
      // Global conjugate function already handles SelfAdjointMatrix and DiagonalMatrix
      static_assert(eigen_triangular_expr<Arg>);

      constexpr auto t = triangular_matrix<Arg, TriangleType::lower> ? TriangleType::upper : TriangleType::lower;
      return make_triangular_matrix<t>(OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg))));
    }


    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) noexcept
    {
      // The general determinant function already handles TriangularMatrix and DiagonalMatrix.
      static_assert(eigen_self_adjoint_expr<Arg>);
      return OpenKalman::determinant(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
    }


    template<typename A, typename B>
    static constexpr auto sum(A&& a, B&& b)
    {
      return library_interface<std::decay_t<nested_matrix_of_t<T>>>::sum(std::forward<A>(a), std::forward<B>(b));
    }


    template<typename A, typename B>
    static constexpr auto contract(A&& a, B&& b)
    {
      return library_interface<std::decay_t<nested_matrix_of_t<T>>>::contract(std::forward<A>(a), std::forward<B>(b));
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      decltype(auto) n = nested_matrix(std::forward<A>(a));
      using Trait = interface::library_interface<std::decay_t<decltype(n)>>;
      if constexpr (eigen_self_adjoint_expr<A>)
      {
        decltype(auto) m = Trait::template rank_update_self_adjoint<significant_triangle>(std::forward<decltype(n)>(n), std::forward<U>(u), alpha);
        return make_hermitian_matrix<significant_triangle>(std::forward<decltype(m)>(m));
      }
      else
      {
        static_assert(eigen_diagonal_expr<A>);
        return Trait::template rank_update_self_adjoint<significant_triangle>(to_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      static_assert(eigen_diagonal_expr<A>);
      using N = std::decay_t<nested_matrix_of_t<T>>;
      return library_interface<N>::template rank_update_triangular<triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b)
    {
      using N = std::decay_t<nested_matrix_of_t<T>>;
      return library_interface<N>::template solve<must_be_unique, must_be_exact>(to_native_matrix(std::forward<A>(a)), std::forward<B>(b));
    }


    template<typename A>
    static constexpr decltype(auto)
    LQ_decomposition(A&& a)
    {
      using N = std::decay_t<nested_matrix_of_t<T>>;
      return library_interface<N>::LQ_decomposition(to_native_matrix(std::forward<A>(a)));
    }


    template<typename A>
    static constexpr decltype(auto)
    QR_decomposition(A&& a)
    {
      using N = std::decay_t<nested_matrix_of_t<T>>;
      return library_interface<N>::QR_decomposition(to_native_matrix(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_SPECIAL_MATRIX_INTERFACE_HPP
