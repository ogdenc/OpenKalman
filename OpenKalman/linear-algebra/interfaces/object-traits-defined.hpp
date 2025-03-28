/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Concepts for testing whether \ref indexible_object_traits are defined for a particular object.
 */

#ifndef OPENKALMAN_OBJECT_TRAITS_DEFINED_HPP
#define OPENKALMAN_OBJECT_TRAITS_DEFINED_HPP

#include <type_traits>
#include "basics/global-definitions.hpp"
#include "values/concepts/number.hpp"
#include "values/concepts/index.hpp"
#include "linear-algebra/interfaces/default/indexible_object_traits.hpp"

namespace OpenKalman::interface
{
  // ------------- //
  //  scalar_type  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T>
  concept scalar_type_defined_for = value::number<typename indexible_object_traits<std::decay_t<T>>::scalar_type>;
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct scalar_type_defined_for_impl : std::false_type {};

    template<typename T>
    struct scalar_type_defined_for_impl<T, std::enable_if_t<value::number<typename indexible_object_traits<std::decay_t<T>>::scalar_type>>>
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
    {indexible_object_traits<std::decay_t<T>>::count_indices(t)} -> value::index;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct count_indices_defined_for_impl : std::false_type {};

    template<typename T>
    struct count_indices_defined_for_impl<T, std::enable_if_t<
      value::index<decltype(indexible_object_traits<std::decay_t<T>>::count_indices(std::declval<T>()))>>>
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
  concept get_vector_space_descriptor_defined_for = requires (T t) {
    {indexible_object_traits<std::decay_t<T>>::get_vector_space_descriptor(
      t, std::integral_constant<std::size_t, 0>{})} -> coordinate::coordinate::pattern;
  };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct get_vector_space_descriptor_defined_for_impl : std::false_type {};

    template<typename T>
    struct get_vector_space_descriptor_defined_for_impl<T, std::enable_if_t<coordinate::coordinate::pattern<
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


  // -------------- //
  //  get_constant  //
  // -------------- //

#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct get_constant_return_type { using type = std::monostate; };


#ifdef __cpp_concepts
  template<typename T> requires requires(T t) { {indexible_object_traits<std::decay_t<T>>::get_constant(t)}; }
  struct get_constant_return_type<T>
#else
  template<typename T>
  struct get_constant_result_type<T, std::void_t<
    decltype(indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>()))>>
#endif
  {
    using type = decltype(indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>()));
  };


  // ----------------------- //
  //  get_constant_diagonal  //
  // ----------------------- //

#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct get_constant_diagonal_return_type { using type = std::monostate; };


#ifdef __cpp_concepts
  template<typename T> requires requires(T t) { {indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(t)}; }
  struct get_constant_diagonal_return_type<T>
#else
  template<typename T>
  struct get_constant_diagonal_result_type<T, std::void_t<
    decltype(indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>()))>>
#endif
  {
    using type = decltype(indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>()));
  };


  // ----------------- //
  //  one_dimensional  //
  // ----------------- //

#ifdef __cpp_concepts
  template<typename T, Applicability b = Applicability::guaranteed>
  concept one_dimensional_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>), bool>;
#else
  namespace detail
  {
    template<typename T, Applicability b, typename = void>
    struct one_dimensional_defined_for_impl : std::false_type {};

    template<typename T, Applicability b>
    struct one_dimensional_defined_for_impl<T, b, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>), bool>>>
      : std::true_type {};
  }

  template<typename T, Applicability b = Applicability::guaranteed>
  constexpr bool one_dimensional_defined_for = detail::one_dimensional_defined_for_impl<T, b>::value;


  template<typename T, Applicability b, typename = void>
  struct is_explicitly_one_dimensional : std::false_type {};

  template<typename T, Applicability b>
  struct is_explicitly_one_dimensional<T, b, std::enable_if_t<indexible_object_traits<std::decay_t<T>>::template one_dimensional<b>>>
    : std::true_type {};
#endif


  // ----------- //
  //  is_square  //
  // ----------- //

#ifdef __cpp_concepts
  template<typename T, Applicability b = Applicability::guaranteed>
  concept is_square_defined_for = std::convertible_to<
    decltype(indexible_object_traits<std::decay_t<T>>::template is_square<b>), bool>;
#else
  namespace detail
  {
    template<typename T, Applicability b, typename = void>
    struct is_square_defined_for_impl : std::false_type {};

    template<typename T, Applicability b>
    struct is_square_defined_for_impl<T, b, std::enable_if_t<std::is_convertible_v<
          decltype(indexible_object_traits<std::decay_t<T>>::template is_square<b>), bool>>>
      : std::true_type {};
  }

  template<typename T, Applicability b = Applicability::guaranteed>
  constexpr bool is_square_defined_for = detail::is_square_defined_for_impl<T, b>::value;
#endif


  // --------------- //
  //  is_triangular  //
  // --------------- //

#ifdef __cpp_concepts
  template<typename T, TriangleType t>
  concept is_triangular_defined_for = requires { indexible_object_traits<std::decay_t<T>>::template is_triangular<t>; };
#else
  namespace detail
  {
    template<typename T, TriangleType t, typename = void>
    struct is_triangular_defined_for_impl : std::false_type {};

    template<typename T, TriangleType t>
    struct is_triangular_defined_for_impl<T, t, std::void_t<decltype(indexible_object_traits<std::decay_t<T>>::template is_triangular<t>)>>
      : std::true_type {};
  }

  template<typename T, TriangleType t>
  constexpr bool is_triangular_defined_for = detail::is_triangular_defined_for_impl<T, t>::value;
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
  concept raw_data_defined_for = requires (T t) { {*indexible_object_traits<std::decay_t<T>>::raw_data(t)} -> value::value; };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct raw_data_defined_for_impl : std::false_type {};

    template<typename T>
    struct raw_data_defined_for_impl<T, std::enable_if_t<
        value::value<decltype(*indexible_object_traits<std::decay_t<T>>::raw_data(std::declval<T>()))>>>
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
  concept strides_defined_for = requires (T t) { indexible_object_traits<std::decay_t<T>>::strides(t); };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct strides_defined_for_impl : std::false_type {};

    template<typename T>
    struct strides_defined_for_impl<T, std::void_t<decltype(indexible_object_traits<std::decay_t<T>>::strides(std::declval<T>()))>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool strides_defined_for = detail::strides_defined_for_impl<T>::value;
#endif


} // namespace OpenKalman::interface


#endif //OPENKALMAN_OBJECT_TRAITS_DEFINED_HPP
