/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Defaults for \ref static_vector_space_descriptor_traits.
 */

#ifndef OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP

#include <type_traits>
#include "basics/values/values.hpp"


namespace OpenKalman::interface
{
  /**
   * \brief Interfaces for a \ref static_vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct static_vector_space_descriptor_traits
  {
#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief The number of dimensions at compile time.
    static constexpr std::size_t size = 0;


    /// \brief The number of dimensions after transforming to Euclidean space.
    static constexpr std::size_t euclidean_size = 0;


    /// \brief The number of atomic component parts at compile time.
    static constexpr std::size_t component_count = 0;


    /// \brief The type of the \ref vector_space_descriptor when \ref indexible objects having respective vector_space_descriptor T are subtracted.
    /// \details For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Dimensions<1>.
    /// So if <code>T</code> is Distance, the resulting <code>difference_type</code> will be Dimensions<1>.
    using difference_type = std::decay_t<T>;


    /// \brief Whether the \ref vector_space_descriptor object is known at compile time to describe Euclidean coordinates (and in this case, size == euclidean_size).
    static constexpr bool always_euclidean = false;


    /**
     * \brief Maps an element from coordinates in modular space to coordinates in Euclidean space.
     * \note For \ref static_vector_space_descriptor, this must be a static function.
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr value::number auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> value::number; } = delete;
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start);
#endif


    /**
     * \brief The inverse of <code>to_euclidean_element</code>. Maps coordinates in Euclidean space back into modular space.
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the coordinate in modular space.
     * \param euclidean_start The starting location in Euclidean space within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr value::number auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> value::number; } = delete;
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start);
#endif


    /**
     * \brief Gets an element from a matrix or tensor object and wraps the result.
     * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
     * or in other words, performing <code>to_euclidean_element</code> followed by <code>from_euclidean_element<code>.
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the element.
     * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    static constexpr value::number auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> value::number; } = delete;
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const G& g, std::size_t local_index, std::size_t start);
#endif


    /**
     * \brief Set an angle and then wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref value::number
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param x The new value to be set.
     * \param local_index A local index accessing the element.
     * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> value::number; } = delete;
#else
    template<typename S, typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start) = delete;
#endif
#endif // DOXYGEN_SHOULD_SKIP_THIS
  };


/**
 * \internal
 * \brief \ref static_vector_space_descriptor_traits for integral values.
 */
#ifdef __cpp_concepts
  template<value::static_index T>
  struct static_vector_space_descriptor_traits<T>
#else
  template<typename T>
  struct static_vector_space_descriptor_traits<T, std::enable_if_t<value::static_index<T>>>
#endif
  {
    static constexpr std::size_t size = static_cast<std::size_t>(T{});


    static constexpr std::size_t euclidean_size = size;


    static constexpr std::size_t component_count = size;


    using difference_type = T;


    static constexpr bool always_euclidean = true;


#ifdef __cpp_concepts
    static constexpr value::number auto to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
    {
      return g(start + euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr value::number auto from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      return g(euclidean_start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr value::number auto get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      return g(start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr void set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> value::number; }
#else
    template<typename S, typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      s(x, start + local_index);
    }

  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
