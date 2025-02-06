/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for StaticDescriptor class specializations and associated aliases.
 */

#ifndef OPENKALMAN_STATICDESCRIPTOR_HPP
#define OPENKALMAN_STATICDESCRIPTOR_HPP

#include <array>
#include <functional>
#include <typeindex>
#include "basics/utils.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_component_collection.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"
#include "internal/Any.hpp"

namespace OpenKalman::descriptor
{
#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct StaticDescriptor
  {
#ifndef __cpp_concepts
    static_assert((static_vector_space_descriptor<Cs> and ...));
#endif
  }; // struct StaticDescriptor

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for StaticDescriptor.
   */
  template<typename...Cs>
  struct coordinate_set_traits<descriptor::StaticDescriptor<Cs...>>
  {
  private:

    using T = descriptor::StaticDescriptor<Cs...>;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto
    size(const T&)
    {
      return std::integral_constant<std::size_t, (0 + ... + descriptor::dimension_size_of_v<Cs>)>{};
    }


    static constexpr auto
    euclidean_size(const T&)
    {
      return std::integral_constant<std::size_t, (0 + ... + descriptor::euclidean_dimension_size_of_v<Cs>)>{};
    }


    static constexpr auto
    is_euclidean(const T&)
    {
      return std::bool_constant<(... and descriptor::euclidean_vector_space_descriptor<Cs>)>{};
    }

  private:

    template<typename...Ts>
    static constexpr auto combine_euclidean()
    {
      constexpr std::size_t N = (0 + ... + descriptor::dimension_size_of_v<Ts>);
      if constexpr (N == 0) return std::tuple {};
      else return std::tuple {descriptor::Dimensions<N>{}};
    }


    template<typename Arg>
    static constexpr auto canonical_tuple_collection(const Arg& arg)
    {
      constexpr auto s = std::tuple_size_v<Arg>;
      if constexpr (s < 2)
      {
        return arg;
      }
      else
      {
        using A0 = std::tuple_element_t<0, Arg>;
        using A1 = std::tuple_element_t<1, Arg>;
        if constexpr (descriptor::euclidean_vector_space_descriptor<A0> and descriptor::euclidean_vector_space_descriptor<A1>)
        {
          return canonical_tuple_collection(std::tuple_cat(combine_euclidean<A0, A1>(), internal::tuple_slice<2, s>(arg)));
        }
        else if constexpr (descriptor::euclidean_vector_space_descriptor<A0>)
        {
          return std::tuple_cat(
            std::tuple_cat(combine_euclidean<A0>(), std::tuple {std::get<1>(arg)}),
            canonical_tuple_collection(OpenKalman::internal::tuple_slice<2, s>(arg)));
        }
        else
        {
          return std::tuple_cat(
            std::tuple {std::get<0>(arg)},
            canonical_tuple_collection(OpenKalman::internal::tuple_slice<1, s>(arg)));
        }
      }
    }

  public:

    static constexpr auto
    component_collection(const T&)
    {
      return canonical_tuple_collection(internal::tuple_flatten(std::tuple {descriptor::internal::get_component_collection(Cs{})...}));
    }

  private:

    using ThisCollection = std::decay_t<decltype(component_collection(T{}))>;


    template<std::size_t...Ix>
    static constexpr auto
    component_start_indices_fixed(std::index_sequence<Ix...>)
    {
      auto i = std::integral_constant<std::size_t, (0 + ... + descriptor::dimension_size_of_v<std::tuple_element_t<Ix, ThisCollection>>)>{};
      auto e = std::integral_constant<std::size_t, (0 + ... + descriptor::euclidean_dimension_size_of_v<std::tuple_element_t<Ix, ThisCollection>>)>{};
      return std::tuple {i, e};
    }


    template<std::size_t N = 0>
    static constexpr auto
    component_start_indices_dynamic(std::size_t n, std::size_t i = 0, std::size_t e = 0)
    {
      if constexpr (N >= std::tuple_size_v<ThisCollection>)
      {
        return std::tuple {static_cast<std::size_t>(size(T{})), static_cast<std::size_t>(euclidean_size(T{}))};
      }
      else
      {
        if (N < n)
        {
          using E = std::tuple_element_t<N, ThisCollection>;
          return component_start_indices_dynamic<N + 1>(n, i + descriptor::dimension_size_of_v<E>, e + descriptor::euclidean_dimension_size_of_v<E>);
        }
        return std::tuple {i, e};
      }
    }

  public:

#ifdef __cpp_concepts
    template<value::index N>
#else
    template<typename N, std::enable_if_t<value::index<N>, int> = 0>
#endif
    static constexpr auto
    component_start_indices(const T&, N n)
    {
      if constexpr (value::fixed<N>)
      {
        return component_start_indices_fixed(std::make_index_sequence<value::fixed_number_of_v<N>>{});
      }
      else
      {
        return component_start_indices_dynamic(n);
      }
    }

  private:

    template<std::size_t c = 0, std::size_t i = 0, std::size_t...ix>
    static constexpr auto index_table_impl(std::index_sequence<ix...> seq = std::index_sequence<>{})
    {
      if constexpr (c < std::tuple_size_v<ThisCollection>)
      {
        if constexpr (i >= descriptor::dimension_size_of_v<std::tuple_element_t<c, ThisCollection>>)
          return index_table_impl<c + 1, 0>(seq);
        else
          return index_table_impl<c, i + 1>(std::index_sequence<ix..., c>{});
      }
      else return std::array<std::size_t, sizeof...(ix)> {ix...};
    }


    static constexpr auto index_table_v = index_table_impl();

  public:

    template<typename Arg>
    static constexpr decltype(auto)
    index_table(const T&)
    {
      return index_table_v;
    }

  private:

    template<std::size_t c = 0, std::size_t i = 0, std::size_t...ix>
    static constexpr auto euclidean_index_table_impl(std::index_sequence<ix...> seq = std::index_sequence<>{})
    {
      if constexpr (c < std::tuple_size_v<ThisCollection>)
      {
        if constexpr (i >= descriptor::euclidean_dimension_size_of_v<std::tuple_element_t<c, ThisCollection>>)
          return euclidean_index_table_impl<c + 1, 0>(seq);
        else
          return euclidean_index_table_impl<c, i + 1>(std::index_sequence<ix..., c>{});
      }
      else return std::array<std::size_t, sizeof...(ix)> {ix...};
    }


    static constexpr auto euclidean_index_table_v = euclidean_index_table_impl();

  public:

    template<typename Arg>
    static constexpr decltype(auto)
    euclidean_index_table(const T&)
    {
      return euclidean_index_table_v;
    }

  private:

    // ------- Function tables ------- //


    template<typename Scalar>
    using GArr = std::array<Scalar(*)(const std::function<Scalar(std::size_t)>&, std::size_t), std::tuple_size_v<ThisCollection>>;


    template<typename Scalar>
    using SArr = std::array<void(*)(const std::function<void(const Scalar&, std::size_t)>&,
      const std::function<Scalar(std::size_t)>&, const Scalar&, std::size_t), std::tuple_size_v<ThisCollection>>;


    template<typename Scalar> static constexpr auto
    to_euclidean_array = std::apply(
      [](const auto&...ds)
      {
        return GArr<Scalar>{ [](const auto& g, std::size_t euclidean_local_index)
          { return descriptor::to_euclidean_element(decltype(ds){}, g, euclidean_local_index); }... };
      }, component_collection(T{}));


    template<typename Scalar> static constexpr auto
    from_euclidean_array = std::apply(
      [](const auto&...ds)
      {
        return GArr<Scalar>{ [](const auto& g, std::size_t local_index)
          { return descriptor::from_euclidean_element(decltype(ds){}, g, local_index); }... };
      }, component_collection(T{}));


    template<typename Scalar> static constexpr auto
    wrap_get_array = std::apply(
      [](const auto&...ds)
      {
        return GArr<Scalar>{ [](const auto& g, std::size_t local_index)
          { return descriptor::get_wrapped_component(decltype(ds){}, g, local_index); }... };
      }, component_collection(T{}));


    template<typename Scalar> static constexpr auto
    wrap_set_array = std::apply(
      [](const auto&...ds)
      {
        return SArr<Scalar>{ [](const auto& s, const auto& g, const Scalar& x, std::size_t local_index)
          { return descriptor::set_wrapped_component(decltype(ds){}, s, g, x, local_index); }... };
      }, component_collection(T{}));

  public:

#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto local_e = static_cast<std::size_t>(euclidean_local_index);
      auto c = euclidean_index_table_v[local_e];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_i](std::size_t i) { return g(comp_i + i); };
      return to_euclidean_array<Scalar>[c](new_g, local_e - comp_e);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto local_i = static_cast<std::size_t>(local_index);
      auto c = index_table_v[local_i];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_e](std::size_t e) { return g(comp_e + e); };
      return from_euclidean_array<Scalar>[c](new_g, local_i - comp_i);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto local_i = static_cast<std::size_t>(local_index);
      auto c = index_table_v[local_i];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_i](std::size_t i) { return g(comp_i + i); };
      return wrap_get_array<Scalar>[c](new_g, local_i - comp_i);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const value::value auto& x, const value::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, std::enable_if_t<value::value<X> and value::index<L> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index)
#endif
    {
      using Scalar = std::decay_t<decltype(x)>;
      auto local_i = static_cast<std::size_t>(local_index);
      auto c = index_table_v[local_i];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_i](std::size_t i) { return g(comp_i + i); };
      auto new_s = [&s, comp_i](const Scalar& x, std::size_t i) { s(x, comp_i + i); };
      wrap_set_array<std::decay_t<decltype(x)>>[c](new_s, new_g, x, local_i - comp_i);
    }

  };


}// namespace OpenKalman::interface


#endif //OPENKALMAN_STATICDESCRIPTOR_HPP
