/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general functions relating to elements.
 */

#ifndef OPENKALMAN_ELEMENT_FUNCTIONS_HPP
#define OPENKALMAN_ELEMENT_FUNCTIONS_HPP

#ifdef __cpp_lib_ranges
#include<ranges>
//#else
#include<algorithm>
#endif

namespace OpenKalman
{
  // \todo Add functions that return stl-compatible iterators.

  // =================== //
  //  Element functions  //
  // =================== //

  namespace detail
  {
    template<std::size_t N, typename Indices>
    constexpr decltype(auto) truncate_indices(const Indices& indices)
    {
      if constexpr (static_range_size_v<Indices> != dynamic_size and N != dynamic_size and static_range_size_v<Indices> > N)
      {
#ifdef __cpp_lib_ranges
        if (std::ranges::any_of(std::views::drop(indices, N), [](const auto& x){ return x != 0; }))
          throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
        return std::views::take(indices, N);
#else
        auto ad = indices.begin();
        std::advance(ad, N);
        if (std::any_of(ad, indices.end(), [](const auto& x){ return x != 0; }))
          throw std::invalid_argument {"Component access: one or more trailing indices are not 0."};
        std::array<std::size_t, N> ret;
        std::copy_n(indices.begin(), N, ret.begin());
        return ret;
#endif
      }
      else return indices;
    }


    template<typename Arg, typename Indices>
    constexpr decltype(auto)
    get_element_impl(Arg&& arg, const Indices& indices)
    {
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      return Trait::get_component(std::forward<Arg>(arg), detail::truncate_indices<index_count_v<Arg>>(indices));
    }
  } // namespace detail


  /**
   * \brief Get a component of an object at a particular set of indices.
   * \tparam Arg The object to be accessed.
   * \tparam Indices A sized input range containing the indices.
   * \return a \ref scalar_constant
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
    (static_range_size_v<Indices> == dynamic_size or index_count_v<Arg> == dynamic_size or static_range_size_v<Indices> >= index_count_v<Arg>) and
    (not empty_object<Arg>)
  constexpr scalar_constant decltype(auto)
#else
  template<typename Arg, typename Indices, std::enable_if_t<indexible<Arg> and index_value<decltype(*std::declval<Indices>().begin())> and
    (static_range_size<Indices>::value == dynamic_size or index_count<Arg>::value == dynamic_size or static_range_size<Indices>::value >= index_count<Arg>::value) and
    (not empty_object<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  get_element(Arg&& arg, const Indices& indices)
  {
    return detail::get_element_impl(std::forward<Arg>(arg), indices);
  }


  /**
   * \overload
   * \brief Get a component of an object using an initializer list.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_value Indices> requires (not empty_object<Arg>)
  constexpr scalar_constant decltype(auto)
#else
  template<typename Arg, typename Indices, std::enable_if_t<indexible<Arg> and index_value<Indices> and
    (not empty_object<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  get_element(Arg&& arg, const std::initializer_list<Indices>& indices)
  {
    return detail::get_element_impl(std::forward<Arg>(arg), indices);
  }


  namespace detail
  {
    template<typename Arg, typename...V, std::size_t...Ix>
    constexpr bool static_indices_within_bounds_impl(std::index_sequence<Ix...>)
    {
      return ([]{
        if constexpr (static_index_value<V>) return (std::decay_t<V>::value < index_dimension_of_v<Arg, Ix>);
        else return true;
      }() and ...);
    }


    template<typename Arg, typename...I>
    struct static_indices_within_bounds
      : std::bool_constant<(detail::static_indices_within_bounds_impl<Arg, I...>(std::index_sequence_for<I...>{}))> {};

  } // namespace detail


  /**
   * \overload
   * \brief Get a component of an object using a fixed number of indices.
   * \details The number of indices must be at least <code>index_count_v&lt;Arg&gt;</code>. If the indices are
   * integral constants, the function performs compile-time bounds checking to the extent possible.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_value...I> requires (index_count_v<Arg> == dynamic_size or sizeof...(I) >= index_count_v<Arg>) and
  (not empty_object<Arg>) and detail::static_indices_within_bounds<Arg, I...>::value
  constexpr scalar_constant decltype(auto)
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and
    (index_count<Arg>::value == dynamic_size or sizeof...(I) >= index_count<Arg>::value) and
    (not empty_object<Arg>) and detail::static_indices_within_bounds<Arg, I...>::value, int> = 0>
  constexpr decltype(auto)
#endif
  get_element(Arg&& arg, I&&...i)
  {
    auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
    return detail::get_element_impl(std::forward<Arg>(arg), indices);
  }


  namespace detail
  {
    template<typename Arg, typename Indices>
    constexpr Arg&&
    set_element_impl(Arg&& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      Trait::set_component(arg, s, detail::truncate_indices<index_count_v<Arg>>(indices));
      return std::forward<Arg>(arg);
    }
  } // namespace detail


  /**
   * \brief Set a component of an object at a particular set of indices.
   * \tparam Arg The object to be accessed.
   * \tparam Indices An input range object containing the indices.
   * \return The modified Arg
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<indexible Arg, std::ranges::input_range Indices> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and index_value<std::ranges::range_value_t<Indices>> and
    (static_range_size_v<Indices> == dynamic_size or index_count_v<Arg> == dynamic_size or static_range_size_v<Indices> >= index_count_v<Arg>) and
    (not empty_object<Arg>) and
    interface::set_component_defined_for<std::decay_t<Arg>, std::add_lvalue_reference_t<Arg>, const scalar_type_of_t<Arg>&, const Indices&>
#else
  template<typename Arg, typename Indices, std::enable_if_t<indexible<Arg> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and index_value<decltype(*std::declval<Indices>().begin())> and
    (static_range_size<Indices>::value == dynamic_size or index_count<Arg>::value == dynamic_size or static_range_size<Indices>::value >= index_count<Arg>::value) and
    (not empty_object<Arg>) and
    interface::set_component_defined_for<std::decay_t<Arg>, typename std::add_lvalue_reference<Arg>::type, const typename scalar_type_of<Arg>::type&, const Indices&>, int> = 0>
#endif
  inline Arg&&
  set_element(Arg&& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
  {
    return detail::set_element_impl(std::forward<Arg>(arg), s, indices);
  }


  /**
   * \overload
   * \brief Set a component of an object using an initializer list.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_value Indices> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not empty_object<Arg>) and
    interface::set_component_defined_for<std::decay_t<Arg>,
      std::add_lvalue_reference_t<Arg>, const scalar_type_of_t<Arg>&, const std::initializer_list<Indices>&>
#else
  template<typename Arg, typename Indices, std::enable_if_t<indexible<Arg> and index_value<Indices> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not empty_object<Arg>) and
    interface::set_component_defined_for<std::decay_t<Arg>,
      typename std::add_lvalue_reference<Arg>::type, const typename scalar_type_of<Arg>::type&, const std::initializer_list<Indices>&>, int> = 0>
#endif
  inline Arg&&
  set_element(Arg&& arg, const scalar_type_of_t<Arg>& s, const std::initializer_list<Indices>& indices)
  {
    return detail::set_element_impl(std::forward<Arg>(arg), s, indices);
  }


  /**
   * \overload
   * \brief Set a component of an object using a fixed number of indices.
   * \details The number of indices must be at least <code>index_count_v&lt;Arg&gt;</code>. If the indices are
   * integral constants, the function performs compile-time bounds checking to the extent possible.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_value...I> requires (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (not empty_object<Arg>) and detail::static_indices_within_bounds<Arg, I...>::value and
    interface::set_component_defined_for<std::decay_t<Arg>,
      std::add_lvalue_reference_t<Arg>, const scalar_type_of_t<Arg>&, const std::array<std::size_t, sizeof...(I)>&>
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and (index_value<I> and ...) and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not empty_object<Arg>) and
    detail::static_indices_within_bounds<Arg, I...>::value and
    interface::set_component_defined_for<std::decay_t<Arg>, typename std::add_lvalue_reference<Arg>::type,
      const typename scalar_type_of<Arg>::type&, const std::array<std::size_t, sizeof...(I)>&>, int> = 0>
#endif
  inline Arg&&
  set_element(Arg&& arg, const scalar_type_of_t<Arg>& s, I&&...i)
  {
    const auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
    return detail::set_element_impl(std::forward<Arg>(arg), s, indices);
  }


  namespace internal
  {
    namespace detail
    {
      template<typename T, std::size_t N, std::size_t...I>
      constexpr bool may_hold_components_impl(std::index_sequence<I...>)
      {
        constexpr auto dims = ((dynamic_dimension<T, I> ? 1 : index_dimension_of_v<T, I>) * ... * 1);
        if constexpr (N == 0) return dims == 0;
        else if constexpr (dims == 0) return false;
        else return N % dims == 0;
      }
    } // namespace detail


    template<typename T, typename...Components>
#ifdef __cpp_concepts
    concept may_hold_components = indexible<T> and (std::convertible_to<Components, const scalar_type_of_t<T>> and ...) and
#else
    constexpr bool may_hold_components = indexible<T> and (std::is_convertible_v<Components, const scalar_type_of_t<T>> and ...) and
#endif
      detail::may_hold_components_impl<T, sizeof...(Components)>(std::make_index_sequence<index_count_v<T>> {});

  } // namespace internal


  /**
   * \overload
   * \brief Set all the components of an object from a list of scalar values.
   * \details The scalar components are listed in the specified layout order, as follows:
   * - \ref Layout::left: column-major;
   * - \ref Layout::right: row-major (the default).
   * \tparam layout The \ref Layout of Args and the resulting object (\ref Layout::right if unspecified).
   * \param arg The object to be modified.
   * \param s Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<Layout layout = Layout::right, writable Arg, scalar_type ... S>
    requires (layout == Layout::right or layout == Layout::left) and internal::may_hold_components<Arg, S...>
  inline writable auto
#else
  template<Layout layout = Layout::right, typename Arg, typename...S, std::enable_if_t<
    writable<Arg> and (scalar_type<S> and ...) and
    (layout == Layout::right or layout == Layout::left) and internal::may_hold_components<Arg, S...>, int> = 0>
  inline Arg&&
#endif
  set_elements(Arg&& arg, S...s)
  {
    if constexpr (sizeof...(S) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = scalar_type_of_t<Arg>;
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      return Trait::template fill_with_elements<layout>(std::forward<Arg>(arg), static_cast<const Scalar>(s)...);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
