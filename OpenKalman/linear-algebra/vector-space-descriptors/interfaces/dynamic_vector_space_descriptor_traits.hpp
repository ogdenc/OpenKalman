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
#include "linear-algebra/values/values.hpp"

namespace OpenKalman::interface
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
#ifdef DOXYGEN_SHOULD_SKIP_THIS
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


    /// \brief Whether the \ref vector_space_descriptor object is known at compile time to describe Euclidean coordinates.
    static constexpr bool always_euclidean = false;


    /**
     * \copydoc static_vector_space_descriptor_traits::to_euclidean_element
     */
#ifdef __cpp_concepts
    constexpr value::number auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start) const
    requires requires (std::size_t i){ {g(i)} -> value::number; } = delete;
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    constexpr auto
    to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) const = delete;
#endif


    /**
     * \copydoc static_vector_space_descriptor_traits::from_euclidean_element
     */
#ifdef __cpp_concepts
    constexpr value::number auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start) const
    requires requires (std::size_t i){ {g(i)} -> value::number; } = delete;
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) const = delete;
#endif


    /**
     * \copydoc static_vector_space_descriptor_traits::get_wrapped_component
     */
#ifdef __cpp_concepts
    constexpr value::number auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start) const
    requires requires (std::size_t i){ {g(i)} -> value::number; } = delete;
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
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
    requires requires (std::size_t i){ s(x, i); {x} -> value::number; } = delete;
#else
    template<typename S, typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    constexpr void
    set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start) const = delete;
#endif
#endif // DOXYGEN_SHOULD_SKIP_THIS
  };


  /**
   * \internal
   * \brief Deduction guide for \ref dynamic_vector_space_descriptor_traits
   */
  template<typename T>
  dynamic_vector_space_descriptor_traits(T&&) -> dynamic_vector_space_descriptor_traits<std::decay_t<T>>;


  /**
   * \internal
   * \brief \ref dynamic_vector_space_descriptor_traits for integral values.
   */
#ifdef __cpp_concepts
  template<value::index T> requires value::dynamic<T>
  struct dynamic_vector_space_descriptor_traits<T>
#else
  template<typename T>
  struct dynamic_vector_space_descriptor_traits<T, std::enable_if_t<value::index<T> and value::dynamic<T>>>
#endif
  {
    explicit constexpr dynamic_vector_space_descriptor_traits(const T& t) : m_integral {t} {};


    [[nodiscard]] constexpr std::size_t get_size() const { return m_integral; }


    [[nodiscard]] constexpr std::size_t get_euclidean_size() const { return m_integral; }


    [[nodiscard]] constexpr std::size_t get_component_count() const { return m_integral; }


    [[nodiscard]] constexpr bool is_euclidean() const { return true; }


    static constexpr bool always_euclidean = true;


#ifdef __cpp_concepts
    static constexpr value::number auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
    {
      return g(start + euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr value::number auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      return g(euclidean_start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr value::number auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> value::number; }
#else
    template<typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      return g(start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> value::number; }
#else
    template<typename S, typename G, std::enable_if_t<value::number<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const S& s, const G& g,
      const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x, std::size_t local_index, std::size_t start)
#endif
    {
      s(x, start + local_index);
    }

  private:

    const T& m_integral;

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
