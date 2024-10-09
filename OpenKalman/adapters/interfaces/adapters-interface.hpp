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

#ifndef OPENKALMAN_ADAPTERS_INTERFACE_HPP
#define OPENKALMAN_ADAPTERS_INTERFACE_HPP

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
    : library_interface<std::decay_t<nested_object_of_t<T>>>
  {
  private:

    using Nested = std::decay_t<nested_object_of_t<T>>;
    using Base = library_interface<Nested>;

  public:

    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, Nested>;


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires
      std::convertible_to<std::ranges::range_value_t<Indices>, const typename std::decay_t<Arg>::Index>
#else
    template<typename Arg, typename Indices>
#endif
    static constexpr scalar_type_of_t<Arg>
    get_component(Arg&& arg, const Indices& indices)
    {
      constexpr std::size_t N = static_range_size_v<Indices>;
      static_assert(N == dynamic_size or N <= 2);

      using Scalar = scalar_type_of_t<Arg>;
      auto it = indices.begin();
      std::size_t i {*it};
      std::size_t j {N == 1 ? 1 : *++it};

      if constexpr (eigen_diagonal_expr<Arg>)
      {
        if (i == j)
          return OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), indices);
        else
          return static_cast<Scalar>(0);
      }
      else if constexpr (eigen_triangular_expr<Arg>)
      {
        if (triangular_matrix<Arg, TriangleType::lower> ? i >= j : j >= i)
          return OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), indices);
        else
          return static_cast<Scalar>(0);
      }
      else
      {
        static_assert(eigen_self_adjoint_expr<Arg>);

        if constexpr (complex_number<Scalar>)
        {
          if (i == j)
            return internal::constexpr_real(OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), indices));
          else if (hermitian_adapter<Arg, HermitianAdapterType::lower> ? i > j : j > i)
            return OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), indices);
          else
            return internal::constexpr_conj(OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), j, i));
        }
        else
        {
          if (hermitian_adapter<Arg, HermitianAdapterType::lower> ? i >= j : j >= i)
            return OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), indices);
          else
            return OpenKalman::get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), j, i);
        }
      }
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires (not std::is_const_v<Arg>) and
    std::convertible_to<std::ranges::range_value_t<Indices>, const typename Arg::Index>
#else
    template<typename Arg, typename Indices, std::enable_if_t<(not std::is_const_v<Arg>), int> = 0>
#endif
    static void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      constexpr std::size_t N = static_range_size_v<Indices>;
      static_assert(N == dynamic_size or N <= 2);

      using Scalar = scalar_type_of_t<Arg>;
      auto it = indices.begin();
      std::size_t i {*it};
      std::size_t j {N == 1 ? 1 : *++it};

      if constexpr (eigen_diagonal_expr<Arg>)
      {
        if (i == j)
          OpenKalman::set_component(arg, s, indices);
        else
          if (s != static_cast<Scalar>(0)) throw std::out_of_range("Cannot set non-diagonal element of a diagonal matrix to a non-zero value.");
      }
      else if constexpr (eigen_triangular_expr<Arg>)
      {
        if (triangular_matrix<Arg, TriangleType::lower> ? i >= j : j >= i)
          OpenKalman::set_component(arg, s, indices);
        else
          if (s != static_cast<Scalar>(0)) throw std::out_of_range("Cannot set elements of a triangular matrix to non-zero values outside the triangle.");
      }
      else
      {
        static_assert(eigen_self_adjoint_expr<Arg>);

        if (hermitian_adapter<Arg, HermitianAdapterType::lower> ? i >= j : j >= i)
          OpenKalman::set_component(arg, s, indices);
        else
          OpenKalman::set_component(arg, internal::constexpr_conj(s), j, i);
      }
    }


    // to_native_matrix inherited

    // make_default inherited

    // fill_components inherited

    // make_constant inherited

    // make_identity_matrix inherited

    // make_triangular_matrix inherited

    // make_hermitian_adapter inherited


    template<typename Arg, typename...Begin, typename...Size>
    static auto
    get_slice(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      auto dense = to_dense_object<scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
      static_assert(not eigen_diagonal_expr<decltype(dense)> and not eigen_triangular_expr<decltype(dense)> and
        not eigen_self_adjoint_expr<decltype(dense)>);
      return OpenKalman::get_slice(std::move(dense), begin, size);
    }


    template<typename Arg, typename Block, typename...Begin>
    static constexpr Arg&
    set_slice(Arg& arg, Block&& block, Begin...begin) = delete;


    // set_triangle not defined because it is handled by global set_triangle function


    template<typename Arg>
    static decltype(auto)
    to_diagonal(Arg&& arg)
    {
      // Note: the interface only needs to handle constant and dynamic-sized zero matrices.
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }


    template<typename Arg>
    static decltype(auto)
    diagonal_of(Arg&& arg)
    {
      // Note: the global diagonal_of function already handles all zero and constant cases.
      if constexpr (eigen_diagonal_expr<Arg>) return nested_object(std::forward<Arg>(arg));
      else return OpenKalman::diagonal_of(nested_object(std::forward<Arg>(arg)));
    }


    // broadcast


    template<typename...Ds, typename Operation, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
    {
      using Traits = library_interface<std::decay_t<nested_object_of_t<T>>>;
      return Traits::n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto)
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      using Traits = library_interface<std::decay_t<nested_object_of_t<T>>>;
      return Traits::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }

    // to_euclidean not defined

    // from_euclidean not defined

    // wrap_angles not defined

    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg)
    {
      if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        constexpr auto t = hermitian_adapter_type_of_v<Arg>;
        return make_hermitian_matrix<t>(OpenKalman::conjugate(nested_object(std::forward<Arg>(arg))));
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);
        constexpr auto t = triangle_type_of_v<Arg>;
        return make_triangular_matrix<t>(OpenKalman::conjugate(nested_object(std::forward<Arg>(arg))));
      }
      // Global conjugate function already handles DiagonalMatrix
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg)
    {
      if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (hermitian_matrix<nested_object_of_t<Arg>>)
        {
          return OpenKalman::transpose(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = (hermitian_adapter<Arg, HermitianAdapterType::lower> ? TriangleType::upper : TriangleType::lower);
          return make_hermitian_matrix<t>(OpenKalman::transpose(nested_object(std::forward<Arg>(arg))));
        }
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);
        if constexpr (triangular_matrix<nested_object_of_t<Arg>>)
        {
          return OpenKalman::transpose(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = triangular_matrix<Arg, TriangleType::lower> ? TriangleType::upper : TriangleType::lower;
          return make_triangular_matrix<t>(OpenKalman::transpose(nested_object(std::forward<Arg>(arg))));
        }
      }
      // Global transpose function already handles DiagonalMatrix
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg)
    {
      // Global conjugate function already handles SelfAdjointMatrix and DiagonalMatrix
      static_assert(eigen_triangular_expr<Arg>);

      constexpr auto t = triangular_matrix<Arg, TriangleType::lower> ? TriangleType::upper : TriangleType::lower;
      return make_triangular_matrix<t>(OpenKalman::adjoint(nested_object(std::forward<Arg>(arg))));
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg)
    {
      // The general determinant function already handles TriangularMatrix and DiagonalMatrix.
      static_assert(eigen_self_adjoint_expr<Arg>);
      return OpenKalman::determinant(to_dense_object(std::forward<Arg>(arg)));
    }


    template<typename Arg, typename...Args>
    static constexpr auto
    sum(Arg&& arg, Args&&...args)
    {
      return library_interface<Nested>::sum(std::forward<Arg>(arg), std::forward<Args>(args)...);
    }


    template<typename A, typename B>
    static constexpr auto
    contract(A&& a, B&& b)
    {
      return library_interface<std::decay_t<nested_object_of_t<T>>>::contract(std::forward<A>(a), std::forward<B>(b));
    }


    // contract_in_place


    template<TriangleType triangle_type, typename A>
    static constexpr auto
    cholesky_factor(A&& a)
    {
      static_assert(not eigen_diagonal_expr<A>); // DiagonalMatrix case should be handled by cholesky_factor function

      if constexpr (eigen_self_adjoint_expr<A>)
      {
        constexpr HermitianAdapterType h = triangle_type == TriangleType::upper ? HermitianAdapterType::upper : HermitianAdapterType::lower;
        if constexpr (hermitian_adapter<A, h>)
          return cholesky_factor<triangle_type>(nested_object(std::forward<A>(a)));
        else
          return cholesky_factor<triangle_type>(adjoint(nested_object(std::forward<A>(a))));
      }
      else
      {
        static_assert(eigen_triangular_expr<A>);
        if constexpr (triangular_matrix<A, triangle_type>)
          return OpenKalman::cholesky_factor<triangle_type>(nested_object(std::forward<A>(a)));
        else
          return OpenKalman::cholesky_factor<triangle_type>(to_diagonal(diagonal_of(std::forward<A>(a))));
      }
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
    {
      auto&& n = nested_object(std::forward<A>(a));
      using Trait = interface::library_interface<std::decay_t<decltype(n)>>;
      if constexpr (eigen_self_adjoint_expr<A>)
      {
        auto&& m = Trait::template rank_update_hermitian<significant_triangle>(std::forward<decltype(n)>(n), std::forward<U>(u), alpha);
        return make_hermitian_matrix<significant_triangle>(std::forward<decltype(m)>(m));
      }
      else
      {
        static_assert(eigen_diagonal_expr<A>);
        return Trait::template rank_update_hermitian<significant_triangle>(to_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      static_assert(eigen_diagonal_expr<A>);
      using N = std::decay_t<nested_object_of_t<T>>;
      return library_interface<N>::template rank_update_triangular<triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b)
    {
      using N = std::decay_t<nested_object_of_t<T>>;
      return library_interface<N>::template solve<must_be_unique, must_be_exact>(to_native_matrix(std::forward<A>(a)), std::forward<B>(b));
    }


    template<typename A>
    static constexpr decltype(auto)
    LQ_decomposition(A&& a)
    {
      using N = std::decay_t<nested_object_of_t<T>>;
      return library_interface<N>::LQ_decomposition(to_native_matrix(std::forward<A>(a)));
    }


    template<typename A>
    static constexpr decltype(auto)
    QR_decomposition(A&& a)
    {
      using N = std::decay_t<nested_object_of_t<T>>;
      return library_interface<N>::QR_decomposition(to_native_matrix(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_ADAPTERS_INTERFACE_HPP
