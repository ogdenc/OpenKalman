/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Concepts for testing whether library-interface functions are defined for a particular object.
 */

#ifndef OPENKALMAN_INTERFACES_DEFINED_HPP
#define OPENKALMAN_INTERFACES_DEFINED_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to traits of a particular array, or generalized tensor within a library.
   * \details This traits class must be specialized for any \ref indexible object (matrix, tensor, etc.)
   * from a linear algebra library. Each different type of objects in a library will typically have its own specialization.
   * \tparam T An object, such as a matrix, array, or tensor, with components addressable by indices.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct indexible_object_traits;


  /**
   * \brief An interface to various routines from the library associated with \ref indexible object T.
   * \details This traits class must be specialized for any object (matrix, tensor, etc.) from a linear algebra library.
   * Typically, only one specialization would be necessary for all objects within a given library.
   * \tparam T An \ref indexible object from a given library.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct library_interface;


  // --------------------------- //
  //  convert_to_self_contained  //
  // --------------------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept convert_to_self_contained_defined_for = requires (Arg arg) {
    interface::indexible_object_traits<T>::convert_to_self_contained(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct convert_to_self_contained_defined_for_impl : std::false_type {};

    template<typename T, typename Arg>
    struct convert_to_self_contained_defined_for_impl<T, Arg,
      std::void_t<decltype(interface::indexible_object_traits<T>::convert_to_self_contained(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool convert_to_self_contained_defined_for = detail::convert_to_self_contained_defined_for_impl<T, Arg>::value;
#endif


  // ------------- //
  //  LibraryBase  //
  // ------------- //

#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Derived, typename LibraryObject>
  concept LibraryBase_defined_for = requires {
    typename interface::library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>;
  };
#else
  namespace detail
  {
    template<typename Derived, typename LibraryObject, typename = void>
    struct LibraryBase_defined_for_impl : std::false_type {};

    template<typename Derived, typename LibraryObject>
    struct LibraryBase_defined_for_impl<Derived, LibraryObject,
      std::void_t<typename interface::library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>>>
      : std::true_type {};
  }

  template<typename Derived, typename LibraryObject>
  constexpr bool LibraryBase_defined_for = detail::LibraryBase_defined_for_impl<Derived, LibraryObject>::value;
#endif


  // --------------- //
  //  get_component  //
  // --------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename Indices>
  concept get_component_defined_for = requires (Arg arg, Indices indices) {
      interface::library_interface<T>::get_component(std::forward<Arg>(arg), std::forward<Indices>(indices));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename Indices, typename = void>
    struct get_component_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename Indices>
    struct get_component_defined_for_impl<T, Arg, Indices, std::void_t<
      decltype(interface::library_interface<T>::get_component(std::declval<Arg>(), std::declval<Indices>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename Indices>
  constexpr bool get_component_defined_for = detail::get_component_defined_for_impl<T, Arg, Indices>::value;
#endif


  // --------------- //
  //  set_component  //
  // --------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename Scalar, typename Indices>
  concept set_component_defined_for = requires (Arg arg, Scalar scalar, Indices indices) {
      interface::library_interface<T>::set_component(std::forward<Arg>(arg), std::forward<Scalar>(scalar), std::forward<Indices>(indices));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename Scalar, typename Indices, typename = void>
    struct set_component_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename Scalar, typename Indices>
    struct set_component_defined_for_impl<T, Arg, Scalar, Indices, std::void_t<
      decltype(interface::library_interface<T>::set_component(std::declval<Arg>(), std::declval<Scalar>(), std::declval<Indices>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename Scalar, typename Indices>
  constexpr bool set_component_defined_for = detail::set_component_defined_for_impl<T, Arg, Scalar, Indices>::value;
#endif


  // ------------------ //
  //  to_native_matrix  //
  // ------------------ //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept to_native_matrix_defined_for = requires (Arg arg) {
      interface::library_interface<T>::to_native_matrix(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct to_native_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct to_native_matrix_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::to_native_matrix(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool to_native_matrix_defined_for = detail::to_native_matrix_defined_for_impl<T, Arg>::value;
#endif


  // make_default not defined

  // fill_with_elements not defined


  // ---------------------- //
  //  make_constant_matrix  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<typename T, typename C, typename...Ds>
  concept make_constant_matrix_defined_for = requires (C c, Ds...ds) {
      interface::library_interface<T>::make_constant_matrix(std::forward<C>(c), std::forward<Ds>(ds)...);
    };
#else
  namespace detail
  {
    template<typename T, typename C, typename = void, typename...Ds>
    struct make_constant_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename C, typename...Ds>
    struct make_constant_matrix_defined_for_impl<T, C, std::void_t<
      decltype(interface::library_interface<T>::make_constant_matrix(std::declval<C>(), std::declval<Ds>()...))>, Ds...>
      : std::true_type {};
  }

  template<typename T, typename C, typename...Ds>
  constexpr bool make_constant_matrix_defined_for = detail::make_constant_matrix_defined_for_impl<T, C, void, Ds...>::value;
#endif


  // ---------------------- //
  //  make_identity_matrix  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<typename T, typename Scalar, typename D>
  concept make_identity_matrix_defined_for = requires (D d) {
      interface::library_interface<T>::template make_identity_matrix<Scalar>(std::forward<D>(d));
    };
#else
  namespace detail
  {
    template<typename T, typename Scalar, typename D, typename = void>
    struct make_identity_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename Scalar, typename D>
    struct make_identity_matrix_defined_for_impl<T, Scalar, D, std::void_t<
      decltype(interface::library_interface<T>::template make_identity_matrix<Scalar>(std::declval<D&&>()))>>
      : std::true_type {};
  }

  template<typename T, typename Scalar, typename D>
  constexpr bool make_identity_matrix_defined_for = detail::make_identity_matrix_defined_for_impl<T, Scalar, D>::value;
#endif


  // ------------------------ //
  //  make_triangular_matrix  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T, TriangleType triangle_type, typename Arg>
  concept make_triangular_matrix_defined_for = requires (Arg arg) {
      interface::library_interface<T>::template make_triangular_matrix<triangle_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename Arg, typename = void>
    struct make_triangular_matrix_defined_for_impl: std::false_type {};

    template<typename T, TriangleType triangle_type, typename Arg>
    struct make_triangular_matrix_defined_for_impl<T, triangle_type, Arg, std::void_t<
      decltype(interface::library_interface<T>::template make_triangular_matrix<triangle_type>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, TriangleType triangle_type, typename Arg>
  constexpr bool make_triangular_matrix_defined_for = detail::make_triangular_matrix_defined_for_impl<T, triangle_type, Arg>::value;
#endif


  // ------------------------ //
  //  make_hermitian_adapter  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T, HermitianAdapterType adapter_type, typename Arg>
  concept make_hermitian_adapter_defined_for = requires (Arg arg) {
      interface::library_interface<T>::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, HermitianAdapterType adapter_type, typename Arg, typename = void>
    struct make_hermitian_adapter_defined_for_impl: std::false_type {};

    template<typename T, HermitianAdapterType adapter_type, typename Arg>
    struct make_hermitian_adapter_defined_for_impl<T, adapter_type, Arg, std::void_t<
      decltype(interface::library_interface<T>::template make_hermitian_adapter<adapter_type>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, HermitianAdapterType adapter_type, typename Arg>
  constexpr bool make_hermitian_adapter_defined_for = detail::make_hermitian_adapter_defined_for_impl<T, adapter_type, Arg>::value;
#endif


  // get_block not defined

  // set_block not defined


  // -------------- //
  //  set_triangle  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T, TriangleType triangle_type, typename A, typename B>
  concept set_triangle_defined_for = requires(A a, B b) {
    interface::library_interface<T>::template set_triangle<triangle_type>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename A, typename B, typename = void>
    struct set_triangle_defined_for_impl : std::false_type {};

    template<typename T, TriangleType triangle_type, typename A, typename B>
    struct set_triangle_defined_for_impl<T, triangle_type, A, B, std::void_t<decltype(
        interface::library_interface<std::decay_t<A>>::template set_triangle<triangle_type>(std::declval<A>(), std::declval<B>()))>>
      : std::true_type {};
  }

  template<typename T, TriangleType triangle_type, typename A, typename B>
  constexpr bool set_triangle_defined_for = detail::set_triangle_defined_for_impl<T, triangle_type, A, B>::value;
#endif


  // ------------- //
  //  to_diagonal  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept to_diagonal_defined_for = requires (Arg arg) {
      interface::library_interface<T>::to_diagonal(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct to_diagonal_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct to_diagonal_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::to_diagonal(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool to_diagonal_defined_for = detail::to_diagonal_defined_for_impl<T, Arg>::value;
#endif


  // ------------- //
  //  diagonal_of  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept diagonal_of_defined_for = requires (Arg arg) {
      interface::library_interface<T>::diagonal_of(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct diagonal_of_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct diagonal_of_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::diagonal_of(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool diagonal_of_defined_for = detail::diagonal_of_defined_for_impl<T, Arg>::value;
#endif


  // ----------- //
  //  broadcast  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename...Factors>
  concept broadcast_defined_for = requires(Arg arg, Factors...factors) {
    interface::library_interface<T>::broadcast(std::forward<Arg>(arg), std::forward<Factors>(factors)...);
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void, typename...Factors>
    struct broadcast_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename...Factors>
    struct broadcast_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::broadcast(std::declval<Arg>(), std::declval<Factors>()...))>, Factors...>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename...Factors>
  constexpr bool broadcast_defined_for = detail::broadcast_defined_for_impl<T, Arg, void, Factors...>::value;
#endif


  // ----------------- //
  //  n_ary_operation  //
  // ----------------- //

#ifdef __cpp_concepts
  template<typename T, typename DTup, typename Op, typename...Args>
  concept n_ary_operation_defined_for = requires(DTup d_tup, Op op, Args...args) {
    interface::library_interface<T>::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
  };
#else
  namespace detail
  {
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
    struct n_ary_operation_defined_for_impl: std::false_type {};

    template<typename T, typename DTup, typename Op, typename...Args>
    struct n_ary_operation_defined_for_impl<T, DTup, Op, std::void_t<
      decltype(interface::library_interface<T>::n_ary_operation(std::declval<DTup>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
      : std::true_type {};
  }

  template<typename T, typename DTup, typename Op, typename...Args>
  constexpr bool n_ary_operation_defined_for = detail::n_ary_operation_defined_for_impl<T, DTup, Op, void, Args...>::value;
#endif


  // -------- //
  //  reduce  //
  // -------- //

#ifdef __cpp_concepts
  template<typename T, typename BinaryFunction, typename Arg, std::size_t...indices>
  concept reduce_defined_for = requires (BinaryFunction op, Arg arg) {
      interface::library_interface<T>::template reduce<indices...>(std::forward<BinaryFunction>(op), std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename BinaryFunction, typename Arg, typename = void, std::size_t...indices>
    struct reduce_defined_for_impl: std::false_type {};

    template<typename T, typename BinaryFunction, typename Arg, std::size_t...indices>
    struct reduce_defined_for_impl<T, BinaryFunction, Arg, std::void_t<
      decltype(interface::library_interface<T>::template reduce<indices...>(std::declval<BinaryFunction>(), std::declval<Arg>()))>, indices...>
      : std::true_type {};
  }

  template<typename T, typename BinaryFunction, typename Arg, std::size_t...indices>
  constexpr bool reduce_defined_for = detail::reduce_defined_for_impl<T, BinaryFunction, Arg, void, indices...>::value;
#endif


  // -------------- //
  //  to_euclidean  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename C>
  concept to_euclidean_defined_for = requires (Arg arg, C c) {
      interface::library_interface<T>::to_euclidean(std::forward<Arg>(arg), std::forward<C>(c));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename C, typename = void>
    struct to_euclidean_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename C>
    struct to_euclidean_defined_for_impl<T, Arg, C, std::void_t<
      decltype(interface::library_interface<T>::to_euclidean(std::declval<Arg>(), std::declval<C>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename C>
  constexpr bool to_euclidean_defined_for = detail::to_euclidean_defined_for_impl<T, Arg, C>::value;
#endif


  // ---------------- //
  //  from_euclidean  //
  // ---------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename C>
  concept from_euclidean_defined_for = requires (Arg arg, C c) {
      interface::library_interface<T>::from_euclidean(std::forward<Arg>(arg), std::forward<C>(c));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename C, typename = void>
    struct from_euclidean_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename C>
    struct from_euclidean_defined_for_impl<T, Arg, C, std::void_t<
      decltype(interface::library_interface<T>::from_euclidean(std::declval<Arg>(), std::declval<C>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename C>
  constexpr bool from_euclidean_defined_for = detail::from_euclidean_defined_for_impl<T, Arg, C>::value;
#endif


  // ------------- //
  //  wrap_angles  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename C>
  concept wrap_angles_defined_for = requires (Arg arg, C c) {
      interface::library_interface<T>::wrap_angles(std::forward<Arg>(arg), std::forward<C>(c));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename C, typename = void>
    struct wrap_angles_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename C>
    struct wrap_angles_defined_for_impl<T, Arg, C, std::void_t<
      decltype(interface::library_interface<T>::wrap_angles(std::declval<Arg>(), std::declval<C>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename C>
  constexpr bool wrap_angles_defined_for = detail::wrap_angles_defined_for_impl<T, Arg, C>::value;
#endif


  // ----------- //
  //  conjugate  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept conjugate_defined_for = requires (Arg arg) {
      interface::library_interface<T>::conjugate(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct conjugate_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct conjugate_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::conjugate(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool conjugate_defined_for = detail::conjugate_defined_for_impl<T, Arg>::value;
#endif


  // ----------- //
  //  transpose  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept transpose_defined_for = requires (Arg arg) {
      interface::library_interface<T>::transpose(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct transpose_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct transpose_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::transpose(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool transpose_defined_for = detail::transpose_defined_for_impl<T, Arg>::value;
#endif


  // --------- //
  //  adjoint  //
  // --------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept adjoint_defined_for = requires (Arg arg) {
      interface::library_interface<T>::adjoint(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct adjoint_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct adjoint_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::adjoint(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool adjoint_defined_for = detail::adjoint_defined_for_impl<T, Arg>::value;
#endif


  // ------------- //
  //  determinant  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept determinant_defined_for = requires (Arg arg) {
      interface::library_interface<T>::determinant(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct determinant_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct determinant_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::determinant(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool determinant_defined_for = detail::determinant_defined_for_impl<T, Arg>::value;
#endif


  // ----- //
  //  sum  //
  // ----- //

#ifdef __cpp_concepts
  template<typename T, typename...Args>
  concept sum_defined_for = requires(Args...args) {
    interface::library_interface<T>::sum(std::forward<Args>(args)...);
  };
#else
  namespace detail
  {
    template<typename T, typename = void, typename...Args>
    struct sum_defined_for_impl : std::false_type {};

    template<typename T, typename...Args>
    struct sum_defined_for_impl<T, std::void_t<decltype(
        interface::library_interface<T>::sum(std::declval<Args>()...))>, Args...>
      : std::true_type {};
  }

  template<typename T, typename...Args>
  constexpr bool sum_defined_for = detail::sum_defined_for_impl<T, void, Args...>::value;
#endif


  // ---------- //
  //  contract  //
  // ---------- //

#ifdef __cpp_concepts
  template<typename T, typename A, typename B>
  concept contract_defined_for = requires(A a, B b) {
    interface::library_interface<T>::contract(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, typename A, typename B, typename = void>
    struct contract_defined_for_impl : std::false_type {};

    template<typename T, typename A, typename B>
    struct contract_defined_for_impl<T, A, B, std::void_t<decltype(
        interface::library_interface<T>::contract(std::declval<A>(), std::declval<B>()))>>
      : std::true_type {};
  }

  template<typename T, typename A, typename B>
  constexpr bool contract_defined_for = detail::contract_defined_for_impl<T, A, B>::value;
#endif


  // ------------------- //
  //  contract_in_place  //
  // ------------------- //

#ifdef __cpp_concepts
  template<typename T, bool on_the_right, typename A, typename B>
  concept contract_in_place_defined_for = requires(A a, B b) {
    interface::library_interface<T>::template contract_in_place<on_the_right>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, bool on_the_right, typename A, typename B, typename = void>
    struct contract_in_place_defined_for_impl : std::false_type {};

    template<typename T, bool on_the_right, typename A, typename B>
    struct contract_in_place_defined_for_impl<T, on_the_right, A, B, std::void_t<decltype(
        interface::library_interface<T>::template contract_in_place<on_the_right>(std::declval<A>(), std::declval<B>()))>>
      : std::true_type {};
  }

  template<typename T, bool on_the_right, typename A, typename B>
  constexpr bool contract_in_place_defined_for = detail::contract_in_place_defined_for_impl<T, on_the_right, A, B>::value;
#endif


  // ----------------- //
  //  cholesky_factor  //
  // ----------------- //

#ifdef __cpp_concepts
  template<typename T, TriangleType triangle_type, typename Arg>
  concept cholesky_factor_defined_for = requires (Arg arg) {
      interface::library_interface<T>::template cholesky_factor<triangle_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename Arg, typename = void>
    struct cholesky_factor_defined_for_impl: std::false_type {};

    template<typename T, TriangleType triangle_type, typename Arg>
    struct cholesky_factor_defined_for_impl<T, triangle_type, Arg, std::void_t<
      decltype(interface::library_interface<T>::template cholesky_factor<triangle_type>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, TriangleType triangle_type, typename Arg>
  constexpr bool cholesky_factor_defined_for = detail::cholesky_factor_defined_for_impl<T, triangle_type, Arg>::value;
#endif


  // -------------------------- //
  //  rank_update_self_adjoint  //
  // -------------------------- //

#ifdef __cpp_concepts
  template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
  concept rank_update_self_adjoint_defined_for = requires (A a, U u, Alpha alpha) {
      interface::library_interface<T>::template rank_update_self_adjoint<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
    };
#else
  namespace detail
  {
    template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha, typename = void>
    struct rank_update_self_adjoint_defined_for_impl: std::false_type {};

    template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    struct rank_update_self_adjoint_defined_for_impl<T, significant_triangle, A, U, Alpha, std::void_t<
      decltype(interface::library_interface<T>::template rank_update_self_adjoint<significant_triangle>(
        std::declval<A>(), std::declval<U>(), std::declval<Alpha>()))>>
      : std::true_type {};
  }

  template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
  constexpr bool rank_update_self_adjoint_defined_for = detail::rank_update_self_adjoint_defined_for_impl<T, significant_triangle, A, U, Alpha>::value;
#endif


  // ------------------------ //
  //  rank_update_triangular  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T, TriangleType triangle_type, typename A, typename U, typename Alpha>
  concept rank_update_triangular_defined_for = requires (A a, U u, Alpha alpha) {
      interface::library_interface<T>::template rank_update_triangular<triangle_type>(std::forward<A>(a), std::forward<U>(u), alpha);
    };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename A, typename U, typename Alpha, typename = void>
    struct rank_update_triangular_defined_for_impl: std::false_type {};

    template<typename T, TriangleType triangle_type, typename A, typename U, typename Alpha>
    struct rank_update_triangular_defined_for_impl<T, triangle_type, A, U, Alpha, std::void_t<
      decltype(interface::library_interface<T>::template rank_update_triangular<triangle_type>(
        std::declval<A>(), std::declval<U>(), std::declval<Alpha>()))>>
      : std::true_type {};
  }

  template<typename T, TriangleType triangle_type, typename A, typename U, typename Alpha>
  constexpr bool rank_update_triangular_defined_for = detail::rank_update_triangular_defined_for_impl<T, triangle_type, A, U, Alpha>::value;
#endif


  // ------- //
  //  solve  //
  // ------- //

#ifdef __cpp_concepts
  template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B>
  concept solve_defined_for = requires(A a, B b) {
    interface::library_interface<T>::template solve<must_be_unique, must_be_exact>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B, typename = void>
    struct solve_defined_for_impl : std::false_type {};

    template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B>
    struct solve_defined_for_impl<T, must_be_unique, must_be_exact, A, B, std::void_t<decltype(
        interface::library_interface<T>::template solve<must_be_unique, must_be_exact>(std::declval<A>(), std::declval<B>()))>>
      : std::true_type {};
  }

  template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B>
  constexpr bool solve_defined_for = detail::solve_defined_for_impl<T, must_be_unique, must_be_exact, A, B>::value;
#endif


  // ------------------ //
  //  LQ_decomposition  //
  // ------------------ //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept LQ_decomposition_defined_for = requires (Arg arg) {
      interface::library_interface<T>::LQ_decomposition(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct LQ_decomposition_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct LQ_decomposition_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::LQ_decomposition(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool LQ_decomposition_defined_for = detail::LQ_decomposition_defined_for_impl<T, Arg>::value;
#endif


  // ------------------ //
  //  LQ_decomposition  //
  // ------------------ //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept QR_decomposition_defined_for = requires (Arg arg) {
      interface::library_interface<T>::QR_decomposition(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct QR_decomposition_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct QR_decomposition_defined_for_impl<T, Arg, std::void_t<
      decltype(interface::library_interface<T>::QR_decomposition(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool QR_decomposition_defined_for = detail::QR_decomposition_defined_for_impl<T, Arg>::value;
#endif


} // namespace OpenKalman::interface


#endif //OPENKALMAN_INTERFACES_DEFINED_HPP
