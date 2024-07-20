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
 * \brief \ref vector_space_descriptor interfaces for integral values.
 */

#ifndef OPENKALMAN_INTEGRAL_INTERFACES_HPP
#define OPENKALMAN_INTEGRAL_INTERFACES_HPP

#include <type_traits>

namespace OpenKalman
{
  /*
   * Fixed
   */
#ifdef __cpp_concepts
  template<static_index_value T>
  struct fixed_vector_space_descriptor_traits<T>
#else
  template<typename T>
  struct fixed_vector_space_descriptor_traits<T, std::enable_if_t<static_index_value<T>>>
#endif
  {
    static constexpr std::size_t size = static_cast<std::size_t>(T{});


    static constexpr std::size_t euclidean_size = size;


    static constexpr std::size_t component_count = size;


    using difference_type = T;


    static constexpr bool always_euclidean = true;


#ifdef __cpp_concepts
    static constexpr scalar_type auto to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
    {
      return g(start + euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      return g(euclidean_start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      return g(start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr void set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      s(x, start + local_index);
    }

  };


  /*
   * Dynamic
   */
#ifdef __cpp_concepts
  template<dynamic_index_value T>
  struct dynamic_vector_space_descriptor_traits<T>
#else
  template<typename T>
  struct dynamic_vector_space_descriptor_traits<T, std::enable_if_t<dynamic_index_value<T>>>
#endif
  {
    explicit constexpr dynamic_vector_space_descriptor_traits(const T& t) : m_integral {t} {};


    [[nodiscard]] constexpr std::size_t get_size() const { return m_integral; }


    [[nodiscard]] constexpr std::size_t get_euclidean_size() const { return m_integral; }


    [[nodiscard]] constexpr std::size_t get_component_count() const { return m_integral; }


    [[nodiscard]] constexpr bool is_euclidean() const { return true; }


    static constexpr bool always_euclidean = true;


#ifdef __cpp_concepts
    static constexpr scalar_type auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
    {
      return g(start + euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      return g(euclidean_start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
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
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
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


} // namespace OpenKalman

#endif //OPENKALMAN_INTEGRAL_INTERFACES_HPP
