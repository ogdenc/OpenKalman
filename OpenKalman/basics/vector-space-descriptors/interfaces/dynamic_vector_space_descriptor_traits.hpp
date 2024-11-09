/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref dynamic_vector_space_descriptor_traits.
 */

#ifndef OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP

#include <type_traits>

namespace OpenKalman::vector_space_descriptors
{
  /**
   * \brief Traits for a \ref dynamic_vector_space_descriptor.
   * \details The traits must define all the functions as indicated here.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct dynamic_vector_space_descriptor_traits
  {
    /// \brief Constructs a traits objects based on a parameter for which runtime traits are needed.
    /// \param t A runtime object of type T.
    explicit constexpr dynamic_vector_space_descriptor_traits(const std::decay_t<T>& t) = delete;


    /// \brief Get the dimension size, at runtime, of t.
    /// \details May be non-static if T is a \ref dynamic_vector_space_descriptor.
    [[nodiscard]] std::size_t get_size() const = delete;


    /// \brief Get the dimension size, at runtime (if transforming to Euclidean space), of t.
    /// \details May be non-static if T is a \ref dynamic_vector_space_descriptor.
    [[nodiscard]] std::size_t get_euclidean_size() const = delete;


    /// \brief Get the number of atomic component parts of the \ref vector_space_descriptor object.
    /// \details May be non-static if T is a \ref dynamic_vector_space_descriptor.
    [[nodiscard]] std::size_t get_component_count() const = delete;


    /// \brief Whether the \ref vector_space_descriptor is euclidean at runtime.
    /// \note May be non-static if T is a \ref dynamic_vector_space_descriptor.
    [[nodiscard]] bool is_euclidean() const = delete;


#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief Whether the \ref vector_space_descriptor object is known at compile time to describe Euclidean coordinates.
    static constexpr bool always_euclidean = false;
#endif


    /**
     * \copydoc static_vector_space_descriptor_traits::to_euclidean_element
     */
#ifdef __cpp_concepts
    constexpr scalar_type auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start) const
    requires requires (std::size_t i){ {g(i)} -> scalar_type; } = delete;
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    constexpr auto
    to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) const = delete;
#endif


    /**
     * \copydoc static_vector_space_descriptor_traits::from_euclidean_element
     */
#ifdef __cpp_concepts
    constexpr scalar_type auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start) const
    requires requires (std::size_t i){ {g(i)} -> scalar_type; } = delete;
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) const = delete;
#endif


    /**
     * \copydoc static_vector_space_descriptor_traits::get_wrapped_component
     */
#ifdef __cpp_concepts
    constexpr scalar_type auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start) const
    requires requires (std::size_t i){ {g(i)} -> scalar_type; } = delete;
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    constexpr auto
    get_wrapped_component(const G& g, std::size_t local_index, std::size_t start) const = delete;
#endif


    /**
     * \copydoc static_vector_space_descriptor_traits::set_wrapped_component
     */
#ifdef __cpp_concepts
    constexpr void
    set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start) const
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; } = delete;
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    constexpr void
    set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start) const = delete;
#endif

  };


  // ----------------- //
  //  deduction guide  //
  // ----------------- //

  template<typename T>
  dynamic_vector_space_descriptor_traits(T&&) -> dynamic_vector_space_descriptor_traits<std::decay_t<T>>;


} // namespace OpenKalman::vector_space_descriptors

#endif //OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
