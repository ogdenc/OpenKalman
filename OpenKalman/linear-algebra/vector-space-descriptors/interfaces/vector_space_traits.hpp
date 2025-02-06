/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref vector_space_traits.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
#define OPENKALMAN_VECTOR_SPACE_TRAITS_HPP

#include <type_traits>
#include <typeindex>
#include "linear-algebra/values/concepts/index.hpp"

namespace OpenKalman::interface
{
  /**
   * \brief Traits for \ref vector_space_descriptor objects.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct vector_space_traits
  {
    static constexpr bool is_specialized = false;


    /**
     * \brief The scalar type associated with T (e.g., double, int).
     * \details <code>OpenKalman::value::number&lt;scalar_type&gt;</code> must be satisfied.
     * \note Optional if \ref descriptor::static_vector_space_descriptor<T>. Defaults to <code>double</code>.
     * \sa descriptor::scalar_type_of
     */
    using scalar_type = double;


    /**
     * \brief The number of dimensions at compile time.
     */
#ifdef __cpp_concepts
    static constexpr value::index auto
#else
    static constexpr auto
#endif
    size(const T& t)
    {
      return std::integral_constant<std::size_t, 0_uz>{};
    }


    /**
     * \brief The number of dimensions after transforming to Euclidean space.
     */
#ifdef __cpp_concepts
    static constexpr value::index auto
#else
    static constexpr auto
#endif
    euclidean_size(const T& t)
    {
      return std::integral_constant<std::size_t, 0_uz>{};
    }


    /**
     * \brief Whether the \ref vector_space_descriptor object describes Euclidean coordinates (and in this case, size == euclidean_size).
     */
#ifdef __cpp_concepts
    static constexpr std::convertible_to<bool> auto
#else
    static constexpr auto
#endif
    is_euclidean(const T& t)
    {
      return std::integral_constant<bool, true>{};
    }


    /**
     * \brief The std::type_index for type T.
     * \details If this is omitted, the hash code will be <code>typeid(t)</code>.
     */
    static constexpr std::type_index
    type_index(const T& t)
    {
        return std::type_index{typeid(t)};
    }


    /**
     * \brief Returns a range that maps components of any modular space to a corresponding component in Euclidean space.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param euclidean_index A local index accessing the coordinate in Euclidean space
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_index)
#endif
    {
      return g(euclidean_index);
    }


    /**
     * \brief The inverse of <code>to_euclidean_element</code>. Maps coordinates in Euclidean space back into modular space.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param index A local index accessing the coordinate in modular space.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& index)
#endif
    {
      return g(index);
    }


    /**
     * \brief Gets an element from a matrix or tensor object and wraps the result.
     * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
     * or in other words, performing <code>to_euclidean_element</code> followed by <code>from_euclidean_element<code>.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param index A local index accessing the element.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& index)
#endif
    {
      return g(index);
    }


    /**
     * \brief Set an angle and then wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param x The new value to be set.
     * \param index A local index accessing the element.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const value::value auto& x, const value::index auto& index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, std::enable_if_t<value::value<X> and value::index<L> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& index)
#endif
    {
      return g(index);
    }


  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
