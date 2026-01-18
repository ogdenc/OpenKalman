/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Concepts for testing whether \ref object_traits or \ref library_interface definitions exist for a particular object.
 */

#ifndef OPENKALMAN_INTERFACES_DEFINED_HPP
#define OPENKALMAN_INTERFACES_DEFINED_HPP

#include <type_traits>
#include <utility>
#include "object_traits.hpp"
#include "library_interface.hpp"
#include "linear-algebra/enumerations.hpp"

namespace OpenKalman::interface
{
  // *************************************************** //
  //                    object_traits                    //
  // *************************************************** //

  // ------------------------ //
  //  get_pattern_collection  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T>
  concept get_pattern_collection_defined_for =
    requires(T t) { object_traits<std::remove_cvref_t<T>>::get_pattern_collection(t); };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct get_pattern_collection_defined_for_impl : std::false_type {};

    template<typename T>
    struct get_pattern_collection_defined_for_impl<T,
      std::void_t<decltype(object_traits<stdex::remove_cvref_t<T>>::get_pattern_collection(std::declval<T>()))>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool get_pattern_collection_defined_for = detail::get_pattern_collection_defined_for_impl<T>::value;
#endif


  // -------------- //
  //  get_constant  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept get_constant_defined_for =
    requires (T t) { object_traits<std::remove_cvref_t<T>>::get_constant(t); };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct get_constant_defined_for_impl : std::false_type {};

    template<typename T>
    struct get_constant_defined_for_impl<T,
      std::void_t<decltype(object_traits<stdex::remove_cvref_t<T>>::get_constant(std::declval<T>()))>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool get_constant_defined_for = detail::get_constant_defined_for_impl<T>::value;
#endif


  // ----------- //
  //  is_square  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, applicability b = applicability::guaranteed>
  concept is_square_defined_for = std::convertible_to<
    decltype(object_traits<std::remove_cvref_t<T>>::template is_square<b>), bool>;
#else
  namespace detail
  {
    template<typename T, applicability b, typename = void>
    struct is_square_defined_for_impl : std::false_type {};

    template<typename T, applicability b>
    struct is_square_defined_for_impl<T, b, std::enable_if_t<stdex::convertible_to<
          decltype(object_traits<stdex::remove_cvref_t<T>>::template is_square<b>), bool>>>
      : std::true_type {};
  }

  template<typename T, applicability b = applicability::guaranteed>
  constexpr bool is_square_defined_for = detail::is_square_defined_for_impl<T, b>::value;
#endif


  // ----------------------- //
  //  is_triangular_adapter  //
  // ----------------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept is_triangular_adapter_defined_for = std::convertible_to<
    decltype(object_traits<std::remove_cvref_t<T>>::is_triangular_adapter), bool>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_triangular_adapter_defined_for_impl : std::false_type {};

    template<typename T>
    struct is_triangular_adapter_defined_for_impl<T, std::enable_if_t<stdex::convertible_to<
          decltype(object_traits<stdex::remove_cvref_t<T>>::is_triangular_adapter), bool>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool is_triangular_adapter_defined_for = detail::is_triangular_adapter_defined_for_impl<T>::value;
#endif


  // -------------- //
  //  is_hermitian  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept is_hermitian_defined_for = std::convertible_to<
    decltype(object_traits<std::remove_cvref_t<T>>::is_hermitian), bool>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_hermitian_defined_for_impl : std::false_type {};

    template<typename T>
    struct is_hermitian_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
          decltype(object_traits<stdex::remove_cvref_t<T>>::is_hermitian), bool>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool is_hermitian_defined_for = detail::is_hermitian_defined_for_impl<T>::value;


  template<typename T, typename = void>
  struct is_explicitly_hermitian : std::false_type {};

  template<typename T>
  struct is_explicitly_hermitian<T, std::enable_if_t<object_traits<stdex::remove_cvref_t<T>>::is_hermitian>>
    : std::true_type {};
#endif


  // ------------------------ //
  //  hermitian_adapter_type  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T>
  concept hermitian_adapter_type_defined_for = std::convertible_to<
    decltype(object_traits<std::remove_cvref_t<T>>::hermitian_adapter_type), HermitianAdapterType>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct hermitian_adapter_type_defined_for_impl : std::false_type {};

    template<typename T>
    struct hermitian_adapter_type_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
          decltype(object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type), HermitianAdapterType>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool hermitian_adapter_type_defined_for = detail::hermitian_adapter_type_defined_for_impl<T>::value;
#endif


  // ******************************************************* //
  //                    library_interface                    //
  // ******************************************************* //

  // -------------------------- //
  //  library_base_defined_for  //
  // -------------------------- //

#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Derived, typename LibraryObject>
  concept library_base_defined_for = requires {
    typename library_interface<std::remove_cvref_t<LibraryObject>>::template library_base<std::decay_t<Derived>>;
  };
#else
  namespace detail
  {
    template<typename Derived, typename LibraryObject, typename = void>
    struct LibraryBase_defined_for_impl : std::false_type {};

    template<typename Derived, typename LibraryObject>
    struct LibraryBase_defined_for_impl<Derived, LibraryObject,
      std::void_t<typename library_interface<stdex::remove_cvref_t<LibraryObject>>::template library_base<std::decay_t<Derived>>>>
      : std::true_type {};
  }

  template<typename Derived, typename LibraryObject>
  constexpr bool library_base_defined_for = detail::LibraryBase_defined_for_impl<Derived, LibraryObject>::value;
#endif


  // ----------- //
  //  copy_from  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename LHS, typename RHS>
  concept copy_from_defined_for = requires(LHS lhs, RHS rhs) {
      library_interface<std::remove_cvref_t<LHS>>::copy(lhs, std::forward<RHS>(rhs));
    };
#else
  namespace detail
  {
    template<typename LHS, typename RHS, typename = void>
    struct copy_from_defined_for_impl : std::false_type {};

    template<typename LHS, typename RHS>
    struct copy_from_defined_for_impl<LHS, RHS,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<LHS>>::copy(std::declval<LHS>(), std::declval<RHS>()))>>
      : std::true_type {};
  }

  template<typename LHS, typename RHS>
  constexpr bool copy_from_defined_for = detail::copy_from_defined_for_impl<LHS, RHS>::value;
#endif


  // ------------- //
  //  to_diagonal  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename Arg>
  concept to_diagonal_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename Arg, typename = void>
    struct to_diagonal_defined_for_impl: std::false_type {};

    template<typename Arg>
    struct to_diagonal_defined_for_impl<Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<Arg>>::to_diagonal(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg>
  constexpr bool to_diagonal_defined_for = detail::to_diagonal_defined_for_impl<Arg>::value;
#endif


  // ------------- //
  //  diagonal_of  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename Arg>
  concept diagonal_of_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename Arg, typename = void>
    struct diagonal_of_defined_for_impl: std::false_type {};

    template<typename Arg>
    struct diagonal_of_defined_for_impl<Arg, std::void_t<
      decltype(library_interface<stdex::remove_cvref_t<Arg>>::diagonal_of(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg>
  constexpr bool diagonal_of_defined_for = detail::diagonal_of_defined_for_impl<Arg>::value;
#endif


  // ----------- //
  //  conjugate  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename Arg>
  concept conjugate_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<Arg>>::conjugate(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename Arg, typename = void>
    struct conjugate_defined_for_impl: std::false_type {};

    template<typename Arg>
    struct conjugate_defined_for_impl<Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<Arg>>::conjugate(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg>
  constexpr bool conjugate_defined_for = detail::conjugate_defined_for_impl<Arg>::value;
#endif


  // ----------- //
  //  transpose  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename Arg, std::size_t indexa, std::size_t indexb>
  concept transpose_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<Arg>>::template transpose<indexa, indexb>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename Arg, std::size_t indexa, std::size_t indexb, typename = void>
    struct transpose_defined_for_impl: std::false_type {};

    template<typename Arg, std::size_t indexa, std::size_t indexb>
    struct transpose_defined_for_impl<Arg, indexa, indexb,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<Arg>>::template transpose<indexa, indexb>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg, std::size_t indexa, std::size_t indexb>
  constexpr bool transpose_defined_for = detail::transpose_defined_for_impl<Arg, indexa, indexb>::value;
#endif


  // --------------------- //
  //  conjugate_transpose  //
  // --------------------- //

#ifdef __cpp_concepts
  template<typename Arg>
  concept conjugate_transpose_defined_for = requires (Arg arg) {
    library_interface<std::remove_cvref_t<Arg>>::conjugate_transpose(std::forward<Arg>(arg));
  };
#else
  namespace detail
  {
    template<typename Arg, typename = void>
    struct conjugate_transpose_defined_for_impl: std::false_type {};

    template<typename Arg>
    struct conjugate_transpose_defined_for_impl<Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<Arg>>::conjugate_transpose(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg>
  constexpr bool conjugate_transpose_defined_for = detail::conjugate_transpose_defined_for_impl<Arg>::value;
#endif


  // ------------------ //
  //  to_native_matrix  //
  // ------------------ //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept to_native_matrix_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::to_native_matrix(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct to_native_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct to_native_matrix_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::to_native_matrix(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool to_native_matrix_defined_for = detail::to_native_matrix_defined_for_impl<T, Arg>::value;
#endif


  // -------------- //
  //  make_default  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T, typename layout, typename Scalar, typename D>
  concept make_default_defined_for = requires(D d) {
    library_interface<std::remove_cvref_t<T>>::template make_default<layout, Scalar>(std::forward<D>(d));
  };
#else
  namespace detail
  {
    template<typename T, typename layout, typename Scalar, typename D, typename = void>
    struct make_default_defined_for_impl : std::false_type {};

    template<typename T, typename layout, typename Scalar, typename D>
    struct make_default_defined_for_impl<T, layout, Scalar, D,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template make_default<layout, Scalar>(std::declval<D>()))>>
      : std::true_type {};
  }

  template<typename T, typename layout, typename Scalar, typename D>
  constexpr bool make_default_defined_for = detail::make_default_defined_for_impl<T, layout, Scalar, D>::value;
#endif


  // ----------------- //
  //  fill_components  //
  // ----------------- //

#ifdef __cpp_concepts
  template<typename T, typename layout, typename Arg, typename...Scalars>
  concept fill_components_defined_for = requires(Arg arg, Scalars...scalars) {
    library_interface<std::remove_cvref_t<T>>::template fill_components<layout>(std::forward<Arg>(arg), std::forward<Scalars>(scalars)...);
  };
#else
  namespace detail
  {
    template<typename T, typename layout, typename Arg, typename = void, typename...Scalars>
    struct fill_components_defined_for_impl : std::false_type {};

    template<typename T, typename layout, typename Arg, typename...Scalars>
    struct fill_components_defined_for_impl<T, layout, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template fill_components<layout>(std::declval<Arg>(), std::declval<Scalars>()...))>, Scalars...>
      : std::true_type {};
  }

  template<typename T, typename layout, typename Arg, typename...Scalars>
  constexpr bool fill_components_defined_for = detail::fill_components_defined_for_impl<T, layout, Arg, void, Scalars...>::value;
#endif


  // ------------------------ //
  //  make_triangular_matrix  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T, triangle_type tri, typename Arg>
  concept make_triangular_matrix_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::template make_triangular_matrix<tri>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, triangle_type tri, typename Arg, typename = void>
    struct make_triangular_matrix_defined_for_impl: std::false_type {};

    template<typename T, triangle_type tri, typename Arg>
    struct make_triangular_matrix_defined_for_impl<T, tri, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template make_triangular_matrix<tri>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, triangle_type tri, typename Arg>
  constexpr bool make_triangular_matrix_defined_for = detail::make_triangular_matrix_defined_for_impl<T, tri, Arg>::value;
#endif


  // ------------------------ //
  //  make_hermitian_adapter  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T, HermitianAdapterType adapter_type, typename Arg>
  concept make_hermitian_adapter_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, HermitianAdapterType adapter_type, typename Arg, typename = void>
    struct make_hermitian_adapter_defined_for_impl: std::false_type {};

    template<typename T, HermitianAdapterType adapter_type, typename Arg>
    struct make_hermitian_adapter_defined_for_impl<T, adapter_type, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template make_hermitian_adapter<adapter_type>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, HermitianAdapterType adapter_type, typename Arg>
  constexpr bool make_hermitian_adapter_defined_for = detail::make_hermitian_adapter_defined_for_impl<T, adapter_type, Arg>::value;
#endif


  // -------------- //
  //  to_euclidean  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept to_euclidean_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::to_euclidean(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct to_euclidean_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct to_euclidean_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::to_euclidean(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool to_euclidean_defined_for = detail::to_euclidean_defined_for_impl<T, Arg>::value;
#endif


  // ---------------- //
  //  from_euclidean  //
  // ---------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename V>
  concept from_euclidean_defined_for = requires (Arg arg, V v) {
      library_interface<std::remove_cvref_t<T>>::from_euclidean(std::forward<Arg>(arg), std::forward<V>(v));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename V, typename = void>
    struct from_euclidean_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename V>
    struct from_euclidean_defined_for_impl<T, Arg, V,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::from_euclidean(std::declval<Arg>(), std::declval<V>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename V>
  constexpr bool from_euclidean_defined_for = detail::from_euclidean_defined_for_impl<T, Arg, V>::value;
#endif


  // ------------- //
  //  wrap_angles  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept wrap_angles_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::wrap_angles(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct wrap_angles_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct wrap_angles_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::wrap_angles(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool wrap_angles_defined_for = detail::wrap_angles_defined_for_impl<T, Arg>::value;
#endif


  // ----------- //
  //  get_slice  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename BeginTup, typename SizeTup>
  concept get_slice_defined_for = requires(Arg arg, BeginTup begin_tup, SizeTup size_tup) {
    library_interface<std::remove_cvref_t<T>>::set_slice(std::forward<Arg>(arg), begin_tup, size_tup);
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename BeginTup, typename SizeTup, typename = void>
    struct get_slice_defined_for_impl : std::false_type {};

    template<typename T, typename Arg, typename BeginTup, typename SizeTup>
    struct get_slice_defined_for_impl<T, Arg, BeginTup, SizeTup,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::set_slice(std::declval<Arg>(), std::declval<BeginTup>(), std::declval<SizeTup>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename BeginTup, typename SizeTup>
  constexpr bool get_slice_defined_for = detail::get_slice_defined_for_impl<T, Arg, BeginTup, SizeTup>::value;
#endif


  // ----------- //
  //  set_slice  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename Block, typename...Begin>
  concept set_slice_defined_for = requires(Arg arg, Block block, Begin...begin) {
    library_interface<std::remove_cvref_t<T>>::set_slice(std::forward<Arg>(arg), std::forward<Block>(block), std::forward<Begin>(begin)...);
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename Block, typename = void, typename...Begin>
    struct set_slice_defined_for_impl : std::false_type {};

    template<typename T, typename Arg, typename Block, typename...Begin>
    struct set_slice_defined_for_impl<T, Arg, Block,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::set_slice(std::declval<Arg>(), std::declval<Block>(), std::declval<Begin>()...))>, Begin...>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename Block, typename...Begin>
  constexpr bool set_slice_defined_for = detail::set_slice_defined_for_impl<T, Arg, Block, Begin...>::value;
#endif


  // -------------- //
  //  set_triangle  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T, triangle_type tri, typename A, typename B>
  concept set_triangle_defined_for = requires(A a, B b) {
    library_interface<std::remove_cvref_t<T>>::template set_triangle<tri>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, triangle_type tri, typename A, typename B, typename = void>
    struct set_triangle_defined_for_impl : std::false_type {};

    template<typename T, triangle_type tri, typename A, typename B>
    struct set_triangle_defined_for_impl<T, tri, A, B,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template set_triangle<tri>(std::declval<A>(), std::declval<B>()))>>
      : std::true_type {};
  }

  template<typename T, triangle_type tri, typename A, typename B>
  constexpr bool set_triangle_defined_for = detail::set_triangle_defined_for_impl<T, tri, A, B>::value;
#endif


  // ----------- //
  //  broadcast  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename...Factors>
  concept broadcast_defined_for = requires(Arg arg, Factors...factors) {
    library_interface<std::remove_cvref_t<T>>::broadcast(std::forward<Arg>(arg), std::forward<Factors>(factors)...);
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void, typename...Factors>
    struct broadcast_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename...Factors>
    struct broadcast_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::broadcast(std::declval<Arg>(), std::declval<Factors>()...))>, Factors...>
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
    library_interface<std::remove_cvref_t<T>>::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
  };
#else
  namespace detail
  {
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
    struct n_ary_operation_defined_for_impl: std::false_type {};

    template<typename T, typename DTup, typename Op, typename...Args>
    struct n_ary_operation_defined_for_impl<T, DTup, Op,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::n_ary_operation(std::declval<DTup>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
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
      library_interface<std::remove_cvref_t<T>>::template reduce<indices...>(std::forward<BinaryFunction>(op), std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename BinaryFunction, typename Arg, typename = void, std::size_t...indices>
    struct reduce_defined_for_impl: std::false_type {};

    template<typename T, typename BinaryFunction, typename Arg, std::size_t...indices>
    struct reduce_defined_for_impl<T, BinaryFunction, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template reduce<indices...>(std::declval<BinaryFunction>(), std::declval<Arg>()))>, indices...>
      : std::true_type {};
  }

  template<typename T, typename BinaryFunction, typename Arg, std::size_t...indices>
  constexpr bool reduce_defined_for = detail::reduce_defined_for_impl<T, BinaryFunction, Arg, void, indices...>::value;
#endif


  // ------------- //
  //  determinant  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg>
  concept determinant_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::determinant(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct determinant_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct determinant_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::determinant(std::declval<Arg>()))>>
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
    library_interface<std::remove_cvref_t<T>>::sum(std::forward<Args>(args)...);
  };
#else
  namespace detail
  {
    template<typename T, typename = void, typename...Args>
    struct sum_defined_for_impl : std::false_type {};

    template<typename T, typename...Args>
    struct sum_defined_for_impl<T,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::sum(std::declval<Args>()...))>, Args...>
      : std::true_type {};
  }

  template<typename T, typename...Args>
  constexpr bool sum_defined_for = detail::sum_defined_for_impl<T, void, Args...>::value;
#endif


  // ---------------- //
  //  scalar_product  //
  // ---------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename S>
  concept scalar_product_defined_for = requires(Arg arg, S s) {
    library_interface<std::remove_cvref_t<T>>::scalar_product(std::forward<Arg>(arg), std::forward<S>(s));
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename S, typename = void>
    struct scalar_product_defined_for_impl : std::false_type {};

    template<typename T, typename Arg, typename S>
    struct scalar_product_defined_for_impl<T, Arg, S,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::scalar_product(std::declval<Arg>(), std::declval<S>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename S>
  constexpr bool scalar_product_defined_for = detail::scalar_product_defined_for_impl<T, Arg, S>::value;
#endif


  // ---------------- //
  //  scalar_product  //
  // ---------------- //

#ifdef __cpp_concepts
  template<typename T, typename Arg, typename S>
  concept scalar_quotient_defined_for = requires(Arg arg, S s) {
    library_interface<std::remove_cvref_t<T>>::scalar_quotient(std::forward<Arg>(arg), std::forward<S>(s));
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename S, typename = void>
    struct scalar_quotient_defined_for_impl : std::false_type {};

    template<typename T, typename Arg, typename S>
    struct scalar_quotient_defined_for_impl<T, Arg, S,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::scalar_quotient(std::declval<Arg>(), std::declval<S>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg, typename S>
  constexpr bool scalar_quotient_defined_for = detail::scalar_quotient_defined_for_impl<T, Arg, S>::value;
#endif


  // ---------- //
  //  contract  //
  // ---------- //

#ifdef __cpp_concepts
  template<typename T, typename A, typename B>
  concept contract_defined_for = requires(A a, B b) {
    library_interface<std::remove_cvref_t<T>>::contract(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, typename A, typename B, typename = void>
    struct contract_defined_for_impl : std::false_type {};

    template<typename T, typename A, typename B>
    struct contract_defined_for_impl<T, A, B,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::contract(std::declval<A>(), std::declval<B>()))>>
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
    library_interface<std::remove_cvref_t<T>>::template contract_in_place<on_the_right>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, bool on_the_right, typename A, typename B, typename = void>
    struct contract_in_place_defined_for_impl : std::false_type {};

    template<typename T, bool on_the_right, typename A, typename B>
    struct contract_in_place_defined_for_impl<T, on_the_right, A, B,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template contract_in_place<on_the_right>(std::declval<A>(), std::declval<B>()))>>
      : std::true_type {};
  }

  template<typename T, bool on_the_right, typename A, typename B>
  constexpr bool contract_in_place_defined_for = detail::contract_in_place_defined_for_impl<T, on_the_right, A, B>::value;
#endif


  // ----------------- //
  //  cholesky_factor  //
  // ----------------- //

#ifdef __cpp_concepts
  template<typename T, triangle_type tri, typename Arg>
  concept cholesky_factor_defined_for = requires (Arg arg) {
      library_interface<std::remove_cvref_t<T>>::template cholesky_factor<tri>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, triangle_type tri, typename Arg, typename = void>
    struct cholesky_factor_defined_for_impl: std::false_type {};

    template<typename T, triangle_type tri, typename Arg>
    struct cholesky_factor_defined_for_impl<T, tri, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template cholesky_factor<tri>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, triangle_type tri, typename Arg>
  constexpr bool cholesky_factor_defined_for = detail::cholesky_factor_defined_for_impl<T, tri, Arg>::value;
#endif


  // ----------------------- //
  //  rank_update_hermitian  //
  // ----------------------- //

#ifdef __cpp_concepts
  template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
  concept rank_update_self_adjoint_defined_for = requires (A a, U u, Alpha alpha) {
      library_interface<std::remove_cvref_t<T>>::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
    };
#else
  namespace detail
  {
    template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha, typename = void>
    struct rank_update_self_adjoint_defined_for_impl: std::false_type {};

    template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    struct rank_update_self_adjoint_defined_for_impl<T, significant_triangle, A, U, Alpha,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template rank_update_hermitian<significant_triangle>(std::declval<A>(), std::declval<U>(), std::declval<Alpha>()))>>
      : std::true_type {};
  }

  template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
  constexpr bool rank_update_self_adjoint_defined_for = detail::rank_update_self_adjoint_defined_for_impl<T, significant_triangle, A, U, Alpha>::value;
#endif


  // ------------------------ //
  //  rank_update_triangular  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T, triangle_type tri, typename A, typename U, typename Alpha>
  concept rank_update_triangular_defined_for = requires (A a, U u, Alpha alpha) {
      library_interface<std::remove_cvref_t<T>>::template rank_update_triangular<tri>(std::forward<A>(a), std::forward<U>(u), alpha);
    };
#else
  namespace detail
  {
    template<typename T, triangle_type tri, typename A, typename U, typename Alpha, typename = void>
    struct rank_update_triangular_defined_for_impl: std::false_type {};

    template<typename T, triangle_type tri, typename A, typename U, typename Alpha>
    struct rank_update_triangular_defined_for_impl<T, tri, A, U, Alpha,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template rank_update_triangular<tri>(std::declval<A>(), std::declval<U>(), std::declval<Alpha>()))>>
      : std::true_type {};
  }

  template<typename T, triangle_type tri, typename A, typename U, typename Alpha>
  constexpr bool rank_update_triangular_defined_for = detail::rank_update_triangular_defined_for_impl<T, tri, A, U, Alpha>::value;
#endif


  // ------- //
  //  solve  //
  // ------- //

#ifdef __cpp_concepts
  template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B>
  concept solve_defined_for = requires(A a, B b) {
    library_interface<std::remove_cvref_t<T>>::template solve<must_be_unique, must_be_exact>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B, typename = void>
    struct solve_defined_for_impl : std::false_type {};

    template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B>
    struct solve_defined_for_impl<T, must_be_unique, must_be_exact, A, B,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::template solve<must_be_unique, must_be_exact>(std::declval<A>(), std::declval<B>()))>>
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
      library_interface<std::remove_cvref_t<T>>::LQ_decomposition(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct LQ_decomposition_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct LQ_decomposition_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::LQ_decomposition(std::declval<Arg>()))>>
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
      library_interface<std::remove_cvref_t<T>>::QR_decomposition(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct QR_decomposition_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct QR_decomposition_defined_for_impl<T, Arg,
      std::void_t<decltype(library_interface<stdex::remove_cvref_t<T>>::QR_decomposition(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool QR_decomposition_defined_for = detail::QR_decomposition_defined_for_impl<T, Arg>::value;
#endif


}


#endif
