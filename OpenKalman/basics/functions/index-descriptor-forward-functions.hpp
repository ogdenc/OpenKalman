/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward functions for index descriptors.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_FUNCTIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_FUNCTIONS_HPP

#include <type_traits>


namespace OpenKalman
{
  // ------------------------- //
  //   get_dimension_size_of   //
  // ------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_dimension_size_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return dimension_size_of_v<T>;
    else {
      interface::DynamicIndexDescriptorTraits ret{t};
      return ret.get_size();
    }
  }


  // ----------------------------------- //
  //   get_euclidean_dimension_size_of   //
  // ----------------------------------- //

  /**
   * \brief Get the Euclidean dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_euclidean_dimension_size_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return euclidean_dimension_size_of_v<T>;
    else {
      interface::DynamicIndexDescriptorTraits ret{t};
      return ret.get_euclidean_size();
    }
  }


  // ------------------------------------------- //
  //   get_index_descriptor_component_count_of   //
  // ------------------------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_index_descriptor_component_count_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return index_descriptor_components_of_v<T>;
    else {
      interface::DynamicIndexDescriptorTraits ret{t};
      return ret.get_component_count();
    }
  }


  // ------------------------------------- //
  //   get_index_descriptor_is_euclidean   //
  // ------------------------------------- //

  /**
   * \brief Determine, at runtime, whether \ref index_descriptor T is untyped.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr bool
  get_index_descriptor_is_euclidean(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return euclidean_index_descriptor<T>;
    else {
      interface::DynamicIndexDescriptorTraits ret{t};
      return ret.is_euclidean();
    }
  }


  // ------------------------ //
  //   to_euclidean_element   //
  // ------------------------ //

  /**
   * \brief Maps an element from coordinates in modular space to coordinates in Euclidean space.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
   * \param start The starting location of the angle within any larger set of index type descriptors
   */
#ifdef __cpp_concepts
  constexpr scalar_type auto
  to_euclidean_element(const index_descriptor auto& t, const auto& g, std::size_t euclidean_local_index, std::size_t start)
  requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
  template<typename T, typename G, std::enable_if_t<index_descriptor<T> and
    scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
  constexpr auto to_euclidean_element(const T& t, const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
  {
    using T_d = std::decay_t<decltype(t)>;
    if constexpr (euclidean_index_descriptor<T_d>)
      return g(start + euclidean_local_index);
    else if constexpr (fixed_index_descriptor<T_d>)
      return interface::FixedIndexDescriptorTraits<T_d>::to_euclidean_element(g, euclidean_local_index, start);
    else
      return interface::DynamicIndexDescriptorTraits<T_d>{t}.to_euclidean_element(g, euclidean_local_index, start);
  }


  // -------------------------- //
  //   from_euclidean_element   //
  // -------------------------- //

  /**
   * \brief The inverse of <code>to_euclidean_element</code>. Maps coordinates in Euclidean space back into modular space.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param local_index A local index accessing the coordinate in modular space.
   * \param euclidean_start The starting location in Euclidean space within any larger set of index type descriptors
   */
#ifdef __cpp_concepts
  constexpr scalar_type auto
  from_euclidean_element(const index_descriptor auto& t, const auto& g, std::size_t local_index, std::size_t euclidean_start)
  requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
  template<typename T, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
  constexpr auto from_euclidean_element(const T& t, const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
  {
    using T_d = std::decay_t<decltype(t)>;
    if constexpr (euclidean_index_descriptor<T_d>)
      return g(euclidean_start + local_index);
    else if constexpr (fixed_index_descriptor<T_d>)
      return interface::FixedIndexDescriptorTraits<T_d>::from_euclidean_element(g, local_index, euclidean_start);
    else
      return interface::DynamicIndexDescriptorTraits<T_d>{t}.from_euclidean_element(g, local_index, euclidean_start);
  }


  // -------------------- //
  //   wrap_get_element   //
  // -------------------- //

  /**
   * \brief Gets an element from a matrix or tensor object and wraps the result.
   * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
   * or in other words, performing <code>to_euclidean_element</code> followed by <code>from_euclidean_element<code>.
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param local_index A local index accessing the element.
   * \param start The starting location of the element within any larger set of index type descriptors.
   */
#ifdef __cpp_concepts
  constexpr scalar_type auto wrap_get_element(const index_descriptor auto& t, const auto& g, std::size_t local_index, std::size_t start)
  requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
  template<typename T, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
  constexpr auto wrap_get_element(const T& t, const G& g, std::size_t local_index, std::size_t start)
#endif
  {
    using T_d = std::decay_t<decltype(t)>;
    if constexpr (euclidean_index_descriptor<T_d>)
      return g(start + local_index);
    else if constexpr (fixed_index_descriptor<T_d>)
      return interface::FixedIndexDescriptorTraits<T_d>::wrap_get_element(g, local_index, start);
    else
      return interface::DynamicIndexDescriptorTraits<T_d>{t}.wrap_get_element(g, local_index, start);
  }


  // -------------------- //
  //   wrap_set_element   //
  // -------------------- //

  /**
   * \brief Set an angle and then wrapping.
   * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
   * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
   * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
   * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
   * \param x The new value to be set.
   * \param local_index A local index accessing the element.
   * \param start The starting location of the element within any larger set of index type descriptors.
   */
#ifdef __cpp_concepts
  constexpr void wrap_set_element(const index_descriptor auto& t, const auto& s, const auto& g,
    const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
  requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
  template<typename T, typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
    std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
  constexpr void wrap_set_element(const T& t, const S& s, const G& g,
    const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x, std::size_t local_index, std::size_t start)
#endif
  {
    using T_d = std::decay_t<decltype(t)>;
    if constexpr (euclidean_index_descriptor<T_d>)
      s(x, start + local_index);
    else if constexpr (fixed_index_descriptor<T_d>)
      interface::FixedIndexDescriptorTraits<T_d>::wrap_set_element(s, g, x, local_index, start);
    else
      interface::DynamicIndexDescriptorTraits<T_d>{t}.wrap_set_element(s, g, x, local_index, start);
  }


} // namespace OpenKalman


#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_FUNCTIONS_HPP
