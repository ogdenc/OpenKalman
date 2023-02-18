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
 * \brief Traits for integral index descriptors.
 */

#ifndef OPENKALMAN_INTEGRAL_INTERFACE_TRAITS_HPP
#define OPENKALMAN_INTEGRAL_INTERFACE_TRAITS_HPP

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for fixed std::integral_constant.
   */
#ifdef __cpp_concepts
  template<typename T, T N> requires (N >= 0) and (N != dynamic_size)
  struct FixedIndexDescriptorTraits<std::integral_constant<T, N>>
#else
  template<typename T, T N>
  struct FixedIndexDescriptorTraits<std::integral_constant<T, N>, std::enable_if_t<(N >= 0) and N != dynamic_size>>
#endif
  {
    static constexpr std::size_t size = N;
    static constexpr std::size_t euclidean_size = N;
    static constexpr std::size_t component_count = N;
    using difference_type = std::integral_constant<T, N>;
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
    static constexpr scalar_type auto wrap_get_element(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto wrap_get_element(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      return g(start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr void wrap_set_element(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void wrap_set_element(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      s(x, start + local_index);
    }

  };


#ifdef __cpp_concepts
  template<std::integral T>
  struct DynamicIndexDescriptorTraits<T>
#else
  template<typename T>
  struct DynamicIndexDescriptorTraits<T, std::enable_if_t<std::is_integral_v<T>>>
#endif
  {
    explicit constexpr DynamicIndexDescriptorTraits(const std::decay_t<T>& t) : m_integral {t} {};

    [[nodiscard]] constexpr std::size_t get_size() const { return m_integral; }

    [[nodiscard]] constexpr std::size_t get_euclidean_size() const { return m_integral; }

    [[nodiscard]] constexpr std::size_t get_component_count() const { return m_integral; }

    static constexpr bool is_euclidean() { return true; }


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
    static constexpr scalar_type auto wrap_get_element(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto wrap_get_element(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      return g(start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr void wrap_set_element(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void wrap_set_element(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      s(x, start + local_index);
    }

  private:
    const std::decay_t<T>& m_integral;

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_INTEGRAL_INTERFACE_TRAITS_HPP
