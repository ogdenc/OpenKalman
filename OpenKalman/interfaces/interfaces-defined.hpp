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


  // ------------- //
  //  scalar_type  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept scalar_type_defined_for = scalar_type<typename indexible_object_traits<std::decay_t<T>>::scalar_type>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct scalar_type_defined_for_impl : std::false_type {};

    template<typename T>
    struct scalar_type_defined_for_impl<T, std::enable_if_t<scalar_type<typename indexible_object_traits<std::decay_t<T>>::scalar_type>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool scalar_type_defined_for = detail::scalar_type_defined_for_impl<T>::value;
#endif


  // --------------- //
  //  count_indices  //
  // --------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept count_indices_defined_for = requires (T t) {
    {indexible_object_traits<std::decay_t<T>>::count_indices(t)} -> index_value;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct count_indices_defined_for_impl : std::false_type {};

    template<typename T>
    struct count_indices_defined_for_impl<T, std::enable_if_t<
      index_value<decltype(indexible_object_traits<std::decay_t<T>>::count_indices(std::declval<T>()))>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool count_indices_defined_for = detail::count_indices_defined_for_impl<T>::value;
#endif


  // ----------------------------- //
  //  get_vector_space_descriptor  //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept get_vector_space_descriptor_defined_for = requires (T t, std::integral_constant<std::size_t, 0> n) {
    {indexible_object_traits<std::decay_t<T>>::get_vector_space_descriptor(t, n)} -> vector_space_descriptor;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct get_vector_space_descriptor_defined_for_impl : std::false_type {};

    template<typename T>
    struct get_vector_space_descriptor_defined_for_impl<T, std::enable_if_t<vector_space_descriptor<
      decltype(indexible_object_traits<std::decay_t<T>>::get_vector_space_descriptor(
        std::declval<T>(), std::integral_constant<std::size_t, 0>{}))>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool get_vector_space_descriptor_defined_for = detail::get_vector_space_descriptor_defined_for_impl<T>::value;
#endif


  // --------------- //
  //  nested_object  //
  // --------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept nested_object_defined_for = requires (T&& t) {
    {indexible_object_traits<std::decay_t<T>>::nested_object(std::forward<T>(t))} -> count_indices_defined_for;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct nested_object_defined_for_impl : std::false_type {};

    template<typename T>
    struct nested_object_defined_for_impl<T, std::enable_if_t<
      count_indices_defined_for<decltype(indexible_object_traits<std::decay_t<T>>::nested_object(std::declval<T&&>()))>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool nested_object_defined_for = detail::nested_object_defined_for_impl<T>::value;
#endif


  // --------------------------- //
  //  convert_to_self_contained  //
  // --------------------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept convert_to_self_contained_defined_for = requires (T&& t) {
    {indexible_object_traits<std::decay_t<T>>::convert_to_self_contained(std::forward<T>(t))} -> count_indices_defined_for;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct convert_to_self_contained_defined_for_impl : std::false_type {};

    template<typename T>
    struct convert_to_self_contained_defined_for_impl<T, std::enable_if_t<
      count_indices_defined_for<decltype(indexible_object_traits<std::decay_t<T>>::convert_to_self_contained(std::declval<T&&>()))>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool convert_to_self_contained_defined_for = detail::convert_to_self_contained_defined_for_impl<T>::value;
#endif


  // -------------- //
  //  get_constant  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
  concept get_constant_defined_for = requires(T t) {
    {indexible_object_traits<std::decay_t<T>>::get_constant(t)} -> scalar_constant<c>;
  };
#else
  namespace detail
  {
    template<typename T, CompileTimeStatus c, typename = void>
    struct get_constant_defined_for_impl : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct get_constant_defined_for_impl<T, c, std::enable_if_t<scalar_constant<
      decltype(indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>())), c>>>
      : std::true_type {};

  } // namespace detail

  template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
  constexpr bool get_constant_defined_for = detail::get_constant_defined_for_impl<T, c>::value;
#endif


  // ----------------------- //
  //  get_constant_diagonal  //
  // ----------------------- //

#ifdef __cpp_concepts
  template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
  concept get_constant_diagonal_defined_for = requires(T t) {
    {indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(t)} -> scalar_constant<c>;
  };
#else
  namespace detail
  {
    template<typename T, CompileTimeStatus c, typename = void>
    struct get_constant_diagonal_defined_for_impl : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct get_constant_diagonal_defined_for_impl<T, c, std::enable_if_t<scalar_constant<
      decltype(indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>())), c>>>
      : std::true_type {};

  } // namespace detail

  template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
  constexpr bool get_constant_diagonal_defined_for = detail::get_constant_diagonal_defined_for_impl<std::decay_t<T>, c>::value;
#endif


  // ----------------- //
  //  one_dimensional  //
  // ----------------- //

#ifdef __cpp_concepts
  template<typename T, Likelihood b = Likelihood::definitely>
  concept one_dimensional_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>), bool>;
#else
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct one_dimensional_defined_for_impl : std::false_type {};

    template<typename T, Likelihood b>
    struct one_dimensional_defined_for_impl<T, b, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>), bool>>>
      : std::true_type {};
  }

  template<typename T, Likelihood b = Likelihood::definitely>
  constexpr bool one_dimensional_defined_for = detail::one_dimensional_defined_for_impl<T, b>::value;


  template<typename T, Likelihood b, typename = void>
  struct is_explicitly_one_dimensional : std::false_type {};

  template<typename T, Likelihood b>
  struct is_explicitly_one_dimensional<T, b, std::enable_if_t<indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>>>
    : std::true_type {};
#endif


  // ----------- //
  //  is_square  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, Likelihood b = Likelihood::definitely>
  concept is_square_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::template is_square<b>), bool>;
#else
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_square_defined_for_impl : std::false_type {};

    template<typename T, Likelihood b>
    struct is_square_defined_for_impl<T, b, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::template is_square<b>), bool>>>
      : std::true_type {};
  }

  template<typename T, Likelihood b = Likelihood::definitely>
  constexpr bool is_square_defined_for = detail::is_square_defined_for_impl<T, b>::value;


  template<typename T, Likelihood b, typename = void>
  struct is_explicitly_square : std::false_type {};

  template<typename T, Likelihood b>
  struct is_explicitly_square<T, b, std::enable_if_t<indexible_object_traits<std::decay_t<T>>::template is_square<b>>>
    : std::true_type {};
#endif


  // --------------- //
  //  is_triangular  //
  // --------------- //

#ifdef __cpp_concepts
  template<typename T, TriangleType t, Likelihood b = Likelihood::definitely>
  concept is_triangular_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::template is_triangular<b>), bool>;
#else
  namespace detail
  {
    template<typename T, TriangleType t, Likelihood b, typename = void>
    struct is_triangular_defined_for_impl : std::false_type {};

    template<typename T, TriangleType t, Likelihood b>
    struct is_triangular_defined_for_impl<T, t, b, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::template is_triangular<b>), bool>>>
      : std::true_type {};
  }

  template<typename T, TriangleType t, Likelihood b = Likelihood::definitely>
  constexpr bool is_triangular_defined_for = detail::is_triangular_defined_for_impl<T, t, b>::value;


  template<typename T, TriangleType t, Likelihood b, typename = void>
  struct is_explicitly_triangular : std::false_type {};

  template<typename T, TriangleType t, Likelihood b>
  struct is_explicitly_triangular<T, t, b, std::enable_if_t<indexible_object_traits<std::decay_t<T>>::template is_triangular<t, b>>>
    : std::true_type {};
#endif


  // ----------------------- //
  //  is_triangular_adapter  //
  // ----------------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept is_triangular_adapter_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::is_triangular_adapter), bool>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_triangular_adapter_defined_for_impl : std::false_type {};

    template<typename T>
    struct is_triangular_adapter_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::is_triangular_adapter), bool>>>
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
    decltype(indexible_object_traits<std::decay_t<T>>::is_hermitian), bool>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_hermitian_defined_for_impl : std::false_type {};

    template<typename T>
    struct is_hermitian_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::is_hermitian), bool>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool is_hermitian_defined_for = detail::is_hermitian_defined_for_impl<T>::value;
  
  
  template<typename T, typename = void>
  struct is_explicitly_hermitian : std::false_type {};

  template<typename T>
  struct is_explicitly_hermitian<T, std::enable_if_t<indexible_object_traits<std::decay_t<T>>::is_hermitian>>
    : std::true_type {};
#endif


  // ------------------------ //
  //  hermitian_adapter_type  //
  // ------------------------ //

#ifdef __cpp_concepts
  template<typename T>
  concept hermitian_adapter_type_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::hermitian_adapter_type), HermitianAdapterType>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct hermitian_adapter_type_defined_for_impl : std::false_type {};

    template<typename T>
    struct hermitian_adapter_type_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::hermitian_adapter_type), HermitianAdapterType>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool hermitian_adapter_type_defined_for = detail::hermitian_adapter_type_defined_for_impl<T>::value;
#endif


  // ------------- //
  //  is_writable  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept is_writable_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::is_writable), bool>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_writable_defined_for_impl : std::false_type {};

    template<typename T>
    struct is_writable_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::is_writable), bool>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool is_writable_defined_for = detail::is_writable_defined_for_impl<T>::value;


  template<typename T, typename = void>
  struct is_explicitly_writable : std::false_type {};

  template<typename T>
  struct is_explicitly_writable<T, std::enable_if_t<indexible_object_traits<std::decay_t<T>>::is_writable>> : std::true_type {};
#endif


  // ---------- //
  //  raw_data  //
  // ---------- //

#ifdef __cpp_concepts
  template<typename T>
  concept raw_data_defined_for = requires (T t) {
    requires std::is_pointer_v<decltype(indexible_object_traits<std::decay_t<T>>::raw_data(t))>;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct raw_data_defined_for_impl : std::false_type {};

    template<typename T>
    struct raw_data_defined_for_impl<T, std::enable_if_t<
      (std::is_pointer_v<decltype(indexible_object_traits<std::decay_t<T>>::raw_data(std::declval<T>()))>)>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool raw_data_defined_for = detail::raw_data_defined_for_impl<T>::value;
#endif


  // -------- //
  //  layout  //
  // -------- //

#ifdef __cpp_concepts
  template<typename T>
  concept layout_defined_for = requires(T t) {
    {indexible_object_traits<std::decay_t<T>>::layout} -> std::convertible_to<const Layout>;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct layout_defined_for_impl : std::false_type {};

    template<typename T>
    struct layout_defined_for_impl<T, std::enable_if_t<std::is_convertible_v<
      std::decay_t<decltype(indexible_object_traits<std::decay_t<T>>::layout)>, const Layout>>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool layout_defined_for = detail::layout_defined_for_impl<T>::value;
#endif


  // --------- //
  //  strides  //
  // --------- //

#ifdef __cpp_concepts
  template<typename T>
  concept strides_defined_for = requires (T t) {
    requires std::is_pointer_v<decltype(indexible_object_traits<std::decay_t<T>>::strides(t))>;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct strides_defined_for_impl : std::false_type {};

    template<typename T>
    struct strides_defined_for_impl<T, std::enable_if_t<
        (std::is_pointer_v<decltype(indexible_object_traits<std::decay_t<T>>::strides(std::declval<T>()))>)>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool strides_defined_for = detail::strides_defined_for_impl<T>::value;
#endif


  // ------------- //
  //  LibraryBase  //
  // ------------- //

#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Derived, typename LibraryObject>
  concept LibraryBase_defined_for = requires {
    typename library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>;
  };
#else
  namespace detail
  {
    template<typename Derived, typename LibraryObject, typename = void>
    struct LibraryBase_defined_for_impl : std::false_type {};

    template<typename Derived, typename LibraryObject>
    struct LibraryBase_defined_for_impl<Derived, LibraryObject,
      std::void_t<typename library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>>>
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
      library_interface<T>::get_component(std::forward<Arg>(arg), std::forward<Indices>(indices));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename Indices, typename = void>
    struct get_component_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename Indices>
    struct get_component_defined_for_impl<T, Arg, Indices, std::void_t<
      decltype(library_interface<T>::get_component(std::declval<Arg>(), std::declval<Indices>()))>>
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
      library_interface<T>::set_component(std::forward<Arg>(arg), std::forward<Scalar>(scalar), std::forward<Indices>(indices));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename Scalar, typename Indices, typename = void>
    struct set_component_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename Scalar, typename Indices>
    struct set_component_defined_for_impl<T, Arg, Scalar, Indices, std::void_t<
      decltype(library_interface<T>::set_component(std::declval<Arg>(), std::declval<Scalar>(), std::declval<Indices>()))>>
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
      library_interface<T>::to_native_matrix(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct to_native_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct to_native_matrix_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::to_native_matrix(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool to_native_matrix_defined_for = detail::to_native_matrix_defined_for_impl<T, Arg>::value;
#endif


  // make_default not defined

  // fill_components not defined


  // ---------------------- //
  //  make_constant  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<typename T, typename C, typename...Ds>
  concept make_constant_matrix_defined_for = requires (C c, Ds...ds) {
      library_interface<T>::make_constant(std::forward<C>(c), std::forward<Ds>(ds)...);
    };
#else
  namespace detail
  {
    template<typename T, typename C, typename = void, typename...Ds>
    struct make_constant_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename C, typename...Ds>
    struct make_constant_matrix_defined_for_impl<T, C, std::void_t<
      decltype(library_interface<T>::make_constant(std::declval<C>(), std::declval<Ds>()...))>, Ds...>
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
      library_interface<T>::template make_identity_matrix<Scalar>(std::forward<D>(d));
    };
#else
  namespace detail
  {
    template<typename T, typename Scalar, typename D, typename = void>
    struct make_identity_matrix_defined_for_impl: std::false_type {};

    template<typename T, typename Scalar, typename D>
    struct make_identity_matrix_defined_for_impl<T, Scalar, D, std::void_t<
      decltype(library_interface<T>::template make_identity_matrix<Scalar>(std::declval<D&&>()))>>
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
      library_interface<T>::template make_triangular_matrix<triangle_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename Arg, typename = void>
    struct make_triangular_matrix_defined_for_impl: std::false_type {};

    template<typename T, TriangleType triangle_type, typename Arg>
    struct make_triangular_matrix_defined_for_impl<T, triangle_type, Arg, std::void_t<
      decltype(library_interface<T>::template make_triangular_matrix<triangle_type>(std::declval<Arg>()))>>
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
      library_interface<T>::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, HermitianAdapterType adapter_type, typename Arg, typename = void>
    struct make_hermitian_adapter_defined_for_impl: std::false_type {};

    template<typename T, HermitianAdapterType adapter_type, typename Arg>
    struct make_hermitian_adapter_defined_for_impl<T, adapter_type, Arg, std::void_t<
      decltype(library_interface<T>::template make_hermitian_adapter<adapter_type>(std::declval<Arg>()))>>
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
    library_interface<T>::template set_triangle<triangle_type>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename A, typename B, typename = void>
    struct set_triangle_defined_for_impl : std::false_type {};

    template<typename T, TriangleType triangle_type, typename A, typename B>
    struct set_triangle_defined_for_impl<T, triangle_type, A, B, std::void_t<decltype(
        library_interface<std::decay_t<A>>::template set_triangle<triangle_type>(std::declval<A>(), std::declval<B>()))>>
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
      library_interface<T>::to_diagonal(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct to_diagonal_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct to_diagonal_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::to_diagonal(std::declval<Arg>()))>>
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
      library_interface<T>::diagonal_of(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct diagonal_of_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct diagonal_of_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::diagonal_of(std::declval<Arg>()))>>
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
    library_interface<T>::broadcast(std::forward<Arg>(arg), std::forward<Factors>(factors)...);
  };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void, typename...Factors>
    struct broadcast_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename...Factors>
    struct broadcast_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::broadcast(std::declval<Arg>(), std::declval<Factors>()...))>, Factors...>
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
    library_interface<T>::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
  };
#else
  namespace detail
  {
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
    struct n_ary_operation_defined_for_impl: std::false_type {};

    template<typename T, typename DTup, typename Op, typename...Args>
    struct n_ary_operation_defined_for_impl<T, DTup, Op, std::void_t<
      decltype(library_interface<T>::n_ary_operation(std::declval<DTup>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
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
      library_interface<T>::template reduce<indices...>(std::forward<BinaryFunction>(op), std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename BinaryFunction, typename Arg, typename = void, std::size_t...indices>
    struct reduce_defined_for_impl: std::false_type {};

    template<typename T, typename BinaryFunction, typename Arg, std::size_t...indices>
    struct reduce_defined_for_impl<T, BinaryFunction, Arg, std::void_t<
      decltype(library_interface<T>::template reduce<indices...>(std::declval<BinaryFunction>(), std::declval<Arg>()))>, indices...>
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
      library_interface<T>::to_euclidean(std::forward<Arg>(arg), std::forward<C>(c));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename C, typename = void>
    struct to_euclidean_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename C>
    struct to_euclidean_defined_for_impl<T, Arg, C, std::void_t<
      decltype(library_interface<T>::to_euclidean(std::declval<Arg>(), std::declval<C>()))>>
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
      library_interface<T>::from_euclidean(std::forward<Arg>(arg), std::forward<C>(c));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename C, typename = void>
    struct from_euclidean_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename C>
    struct from_euclidean_defined_for_impl<T, Arg, C, std::void_t<
      decltype(library_interface<T>::from_euclidean(std::declval<Arg>(), std::declval<C>()))>>
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
      library_interface<T>::wrap_angles(std::forward<Arg>(arg), std::forward<C>(c));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename C, typename = void>
    struct wrap_angles_defined_for_impl: std::false_type {};

    template<typename T, typename Arg, typename C>
    struct wrap_angles_defined_for_impl<T, Arg, C, std::void_t<
      decltype(library_interface<T>::wrap_angles(std::declval<Arg>(), std::declval<C>()))>>
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
      library_interface<T>::conjugate(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct conjugate_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct conjugate_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::conjugate(std::declval<Arg>()))>>
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
      library_interface<T>::transpose(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct transpose_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct transpose_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::transpose(std::declval<Arg>()))>>
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
      library_interface<T>::adjoint(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct adjoint_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct adjoint_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::adjoint(std::declval<Arg>()))>>
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
      library_interface<T>::determinant(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct determinant_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct determinant_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::determinant(std::declval<Arg>()))>>
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
    library_interface<T>::sum(std::forward<Args>(args)...);
  };
#else
  namespace detail
  {
    template<typename T, typename = void, typename...Args>
    struct sum_defined_for_impl : std::false_type {};

    template<typename T, typename...Args>
    struct sum_defined_for_impl<T, std::void_t<decltype(
        library_interface<T>::sum(std::declval<Args>()...))>, Args...>
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
    library_interface<T>::contract(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, typename A, typename B, typename = void>
    struct contract_defined_for_impl : std::false_type {};

    template<typename T, typename A, typename B>
    struct contract_defined_for_impl<T, A, B, std::void_t<decltype(
        library_interface<T>::contract(std::declval<A>(), std::declval<B>()))>>
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
    library_interface<T>::template contract_in_place<on_the_right>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, bool on_the_right, typename A, typename B, typename = void>
    struct contract_in_place_defined_for_impl : std::false_type {};

    template<typename T, bool on_the_right, typename A, typename B>
    struct contract_in_place_defined_for_impl<T, on_the_right, A, B, std::void_t<decltype(
        library_interface<T>::template contract_in_place<on_the_right>(std::declval<A>(), std::declval<B>()))>>
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
      library_interface<T>::template cholesky_factor<triangle_type>(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename Arg, typename = void>
    struct cholesky_factor_defined_for_impl: std::false_type {};

    template<typename T, TriangleType triangle_type, typename Arg>
    struct cholesky_factor_defined_for_impl<T, triangle_type, Arg, std::void_t<
      decltype(library_interface<T>::template cholesky_factor<triangle_type>(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, TriangleType triangle_type, typename Arg>
  constexpr bool cholesky_factor_defined_for = detail::cholesky_factor_defined_for_impl<T, triangle_type, Arg>::value;
#endif


  // -------------------------- //
  //  rank_update_hermitian  //
  // -------------------------- //

#ifdef __cpp_concepts
  template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
  concept rank_update_self_adjoint_defined_for = requires (A a, U u, Alpha alpha) {
      library_interface<T>::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
    };
#else
  namespace detail
  {
    template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha, typename = void>
    struct rank_update_self_adjoint_defined_for_impl: std::false_type {};

    template<typename T, HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    struct rank_update_self_adjoint_defined_for_impl<T, significant_triangle, A, U, Alpha, std::void_t<
      decltype(library_interface<T>::template rank_update_hermitian<significant_triangle>(
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
      library_interface<T>::template rank_update_triangular<triangle_type>(std::forward<A>(a), std::forward<U>(u), alpha);
    };
#else
  namespace detail
  {
    template<typename T, TriangleType triangle_type, typename A, typename U, typename Alpha, typename = void>
    struct rank_update_triangular_defined_for_impl: std::false_type {};

    template<typename T, TriangleType triangle_type, typename A, typename U, typename Alpha>
    struct rank_update_triangular_defined_for_impl<T, triangle_type, A, U, Alpha, std::void_t<
      decltype(library_interface<T>::template rank_update_triangular<triangle_type>(
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
    library_interface<T>::template solve<must_be_unique, must_be_exact>(std::forward<A>(a), std::forward<B>(b));
  };
#else
  namespace detail
  {
    template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B, typename = void>
    struct solve_defined_for_impl : std::false_type {};

    template<typename T, bool must_be_unique, bool must_be_exact, typename A, typename B>
    struct solve_defined_for_impl<T, must_be_unique, must_be_exact, A, B, std::void_t<decltype(
        library_interface<T>::template solve<must_be_unique, must_be_exact>(std::declval<A>(), std::declval<B>()))>>
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
      library_interface<T>::LQ_decomposition(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct LQ_decomposition_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct LQ_decomposition_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::LQ_decomposition(std::declval<Arg>()))>>
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
      library_interface<T>::QR_decomposition(std::forward<Arg>(arg));
    };
#else
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct QR_decomposition_defined_for_impl: std::false_type {};

    template<typename T, typename Arg>
    struct QR_decomposition_defined_for_impl<T, Arg, std::void_t<
      decltype(library_interface<T>::QR_decomposition(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename T, typename Arg>
  constexpr bool QR_decomposition_defined_for = detail::QR_decomposition_defined_for_impl<T, Arg>::value;
#endif


} // namespace OpenKalman::interface


#endif //OPENKALMAN_INTERFACES_DEFINED_HPP
