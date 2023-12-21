/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref self_contained.
 */

#ifndef OPENKALMAN_SELF_CONTAINED_HPP
#define OPENKALMAN_SELF_CONTAINED_HPP


namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct self_contained_impl : std::false_type {};


    template<typename Tup, std::size_t...I>
    constexpr bool no_lvalue_ref_dependencies(std::index_sequence<I...>)
    {
      return (self_contained_impl<std::tuple_element_t<I, Tup>>::value and ...);
    }


#ifdef __cpp_concepts
    template<typename T> requires (not std::is_lvalue_reference_v<T>) and
      (detail::no_lvalue_ref_dependencies<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>(
        std::make_index_sequence<std::tuple_size_v<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>> {}))
    struct self_contained_impl<T> : std::true_type {};
#else
    template<typename T>
    struct self_contained_impl<T, std::enable_if_t<
      (std::tuple_size<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>::value >= 0)>>
      : std::bool_constant<(not std::is_lvalue_reference_v<T>) and
          no_lvalue_ref_dependencies<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>(
          std::make_index_sequence<std::tuple_size_v<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>> {})> {};
#endif


    template<typename Tup, std::size_t...I>
    constexpr bool all_lvalue_ref_dependencies_impl(std::index_sequence<I...>)
    {
      return ((sizeof...(I) > 0) and ... and std::is_lvalue_reference_v<std::tuple_element_t<I, Tup>>);
    }


    template<typename T, std::size_t...I>
    constexpr bool no_recursive_runtime_parameters(std::index_sequence<I...>)
    {
      using Traits = interface::indexible_object_traits<T>;
      return ((not Traits::has_runtime_parameters) and ... and
        no_recursive_runtime_parameters<std::decay_t<std::tuple_element_t<I, typename Traits::dependents>>>(
          std::make_index_sequence<std::tuple_size_v<typename interface::indexible_object_traits<std::decay_t<std::tuple_element_t<I, typename Traits::dependents>>>::dependents>> {}
          ));
    }

#ifdef __cpp_concepts
    template<typename T>
    concept all_lvalue_ref_dependencies =
      no_recursive_runtime_parameters<std::decay_t<T>>(
        std::make_index_sequence<std::tuple_size_v<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>> {}) and
      all_lvalue_ref_dependencies_impl<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>(
        std::make_index_sequence<std::tuple_size_v<typename interface::indexible_object_traits<std::decay_t<T>>::dependents>> {});
#else
    template<typename T, typename = void>
    struct has_no_runtime_parameters_impl : std::false_type {};

    template<typename T>
    struct has_no_runtime_parameters_impl<T, std::enable_if_t<not interface::indexible_object_traits<T>::has_runtime_parameters>>
      : std::true_type {};


    template<typename T, typename = void>
    struct all_lvalue_ref_dependencies_detail : std::false_type {};

    template<typename T>
    struct all_lvalue_ref_dependencies_detail<T, std::void_t<typename interface::indexible_object_traits<T>::dependents>>
      : std::bool_constant<has_no_runtime_parameters_impl<T>::value and
        (all_lvalue_ref_dependencies_impl<typename interface::indexible_object_traits<T>::dependents>(
          std::make_index_sequence<std::tuple_size_v<typename interface::indexible_object_traits<T>::dependents>> {}))> {};

    template<typename T>
    constexpr bool all_lvalue_ref_dependencies = all_lvalue_ref_dependencies_detail<std::decay_t<T>>::value;
#endif
  } // namespace detail


  /**
   * \brief Specifies that a type is a self-contained matrix or expression.
   * \details A value is self-contained if it can be created in a function and returned as the result.
   * \tparam T The object in question
   * \tparam Ts An optional set of objects that T depends on. T is self-contained if all of Ts are either
   * lvalue references or depend only on lvalue references.
   * \sa make_self_contained, equivalent_self_contained_t
   * \internal \sa indexible_object_traits
   */
  template<typename T, typename...Ts>
#ifdef __cpp_concepts
  concept self_contained =
#else
  constexpr bool self_contained =
#endif
    detail::self_contained_impl<T>::value or
    ((sizeof...(Ts) > 0) and ... and (std::is_lvalue_reference_v<Ts> or detail::all_lvalue_ref_dependencies<Ts>));


} // namespace OpenKalman

#endif //OPENKALMAN_SELF_CONTAINED_HPP
