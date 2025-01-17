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
 * \brief Definition of \ref vector_space_traits.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
#define OPENKALMAN_VECTOR_SPACE_TRAITS_HPP

#include <type_traits>
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
#ifdef DOXYGEN_SHOULD_SKIP_THIS
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
    size(const T&);


    /**
     * \brief The number of dimensions after transforming to Euclidean space.
     */
#ifdef __cpp_concepts
    static constexpr value::index auto
#else
    static constexpr auto
#endif
    euclidean_size(const T&);


    /**
     * \brief A \ref descriptor::vector_space_descriptor_collection containing the coordinate groups within T.
     * \details The collection represents the canonical form of T.
     * Two types are equivalent, by definition, if their canonical equivalent types are the same.
     */
#ifdef __cpp_concepts
    static constexpr descriptor::vector_space_descriptor_collection auto
#else
    static constexpr auto
#endif
    collection(const T&);


    /**
     * \brief Whether the \ref vector_space_descriptor object describes Euclidean coordinates (and in this case, size == euclidean_size).
     */
#ifdef __cpp_concepts
    static constexpr std::convertible_to<bool> auto
#else
    static constexpr auto
#endif
    is_euclidean(const T&);


    /**
     * \brief The std::type_index for type T.
     * \details If this is omitted, the hash code will be <code>typeid(t)</code>.
     */
    static constexpr std::type_index
    type_index(const T& t);


    /**
     * \brief Returns a range that maps components of any modular space to a corresponding component in Euclidean space.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T&, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires { {g(start)} -> value::value; };
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, const S&>::type> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T&, const Getter& g, const L& euclidean_local_index, const S& start);
#endif


    /**
     * \brief The inverse of <code>to_euclidean_element</code>. Maps coordinates in Euclidean space back into modular space.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the coordinate in modular space.
     * \param euclidean_start The starting location in Euclidean space within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires { {g(euclidean_start)} -> value::value; };
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, const S&>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T&, const Getter& g, const L& local_index, const S& euclidean_start);
#endif


    /**
     * \brief Gets an element from a matrix or tensor object and wraps the result.
     * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
     * or in other words, performing <code>to_euclidean_element</code> followed by <code>from_euclidean_element<code>.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the element.
     * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires { {g(start)} -> value::value; };
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, const S&>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T&, const Getter& g, const L& local_index, const S& start);
#endif


    /**
     * \brief Set an angle and then wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \note Optional if T is a \ref euclidean_vector_space_descriptor.
     * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
     * \param g An element getter mapping an index i of type std::size_t to an element
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param x The new value to be set.
     * \param local_index A local index accessing the element.
     * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T&, const auto& s, const auto& g, const value::value auto& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires { s(x, start); s(g(start), start); };
#else
    template<typename Setter, typename Getter, typename X, typename L, typename S, std::enable_if_t<
      value::value<X> and value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const X&, const S&>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, const S&>::type, const S&>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T&, const Setter& s, const Getter& g, const X& x, const L& local_index, const S& start);
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS
  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
