/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref descriptor::internal::Replicate.
 */

#ifndef OPENKALMAN_DESCRIPTOR_REPLICATE_HPP
#define OPENKALMAN_DESCRIPTOR_REPLICATE_HPP

#include <type_traits>
#include <array>
#include <functional>

#include "basics/utils.hpp"
#include "basics/internal/iota_range.hpp"
#include "basics/internal/iota_tuple.hpp"
#include "basics/internal/get_collection_size.hpp"
#include "linear-algebra/values/concepts/index.hpp"
#include "linear-algebra/values/functions/internal/get_collection_element.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_size.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_size.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_component_collection.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_is_euclidean.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_type_index.hpp"
#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/traits/scalar_type_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/Any.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief A type representing the replication of other \ref vector_space_descriptor objects.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor C, value::index Factor>
#else
  template<typename C, typename Factor>
#endif
  struct Replicate
  {
    /// Default constructor
#ifdef __cpp_concepts
    constexpr Replicate() requires static_vector_space_descriptor<C> and value::fixed<Factor> = default;
#else
    template<typename mC = C, typename mN = Factor, std::enable_if_t<
      static_vector_space_descriptor<mC> and value::fixed<mN>, int> = 0>
    constexpr Replicate() {};
#endif


    /**
     * \brief Construct from a \ref vector_space_descriptor and a replication factor.
     */
#ifdef __cpp_concepts
    template <vector_space_descriptor Arg, value::index I> requires
      std::constructible_from<C, Arg&&> and std::constructible_from<Factor, I&&>
#else
    template<typename Arg, typename I, std::enable_if_t<vector_space_descriptor<Arg> and value::index<I> and
      std::is_constructible_v<C, Arg&&> and std::is_constructible_v<Factor, I&&>, int> = 0>
#endif
    explicit constexpr
    Replicate(Arg&& arg, I&& i)
      : my_coordinates {std::forward<Arg>(arg)}, my_factor {std::forward<I>(i)} {}

  private:

    C my_coordinates;
    Factor my_factor;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::vector_space_traits;

  };


  template<typename Arg, typename I>
  Replicate(Arg&&, const I&) -> Replicate<std::conditional_t<static_vector_space_descriptor<Arg>, std::decay_t<Arg>, Arg>, I>;

} // namespace OpenKalman::descriptor::internal


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for descriptor::internal::Reverse.
   */
  template<typename C, typename Factor>
  struct coordinate_set_traits<descriptor::internal::Replicate<C, Factor>>
  {
  private:

    using T = descriptor::internal::Replicate<C, Factor>;

  public:

    static constexpr bool is_specialized = true;


    using scalar_type = descriptor::scalar_type_of_t<C>;


    static constexpr auto
    size(const T& t)
    {
      return value::operation {std::multiplies<>{}, descriptor::get_size(t.my_coordinates), t.my_factor};
    }


    static constexpr auto
    euclidean_size(const T& t)
    {
      return value::operation {std::multiplies<>{}, descriptor::get_euclidean_size(t.my_coordinates), t.my_factor};
    }


    static constexpr auto
    is_euclidean(const T& t)
    {
      return descriptor::get_is_euclidean(t.my_coordinates);
    }

  private:

    template<typename Arg, std::size_t...Ix>
    static constexpr auto
    tuple_to_range_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      return std::array {descriptor::internal::Any<scalar_type> {std::get<Ix>(std::forward<Arg>(arg))}...};
    }


    template<typename Arg>
    static constexpr auto
    tuple_to_range(Arg&& arg)
    {
      if constexpr (descriptor::vector_space_descriptor_range<Arg>)
        return std::forward<Arg>(arg);
      else
        return tuple_to_range_impl(std::forward<Arg>(arg), std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{});
    }

  public:

    constexpr auto
    component_collection(const T& t)
    {
      if constexpr (descriptor::dimension_size_of_v<C> == 0)
      {
        return std::tuple {};
      }
      else if constexpr (descriptor::euclidean_vector_space_descriptor<C>)
      {
        auto s = value::operation {std::multiplies<>{}, descriptor::get_size(t.my_coordinates), t.my_factor};
        return std::array {descriptor::Dimensions {s}};
      }
      else if constexpr (descriptor::static_vector_space_descriptor<C> and value::fixed<Factor>)
      {
        constexpr std::size_t n = value::fixed_number_of_v<Factor>;
        using Coll =  std::decay_t<decltype(descriptor::internal::get_component_collection(t.my_coordinates))>;
        constexpr std::size_t s = std::tuple_size_v<Coll>;
        using c_first = std::tuple_element_t<0, Coll>;
        using c_last = std::tuple_element_t<s - 1, Coll>;
        if constexpr (descriptor::euclidean_vector_space_descriptor<c_last> and
          descriptor::euclidean_vector_space_descriptor<c_first> and n >= 2)
        {
          auto f = [&t](auto fixed_i)
          {
            decltype(auto) coll = descriptor::internal::get_component_collection(t.my_coordinates);
            constexpr std::size_t i = value::fixed_number_of_v<decltype(fixed_i)>;
            using std::get;
            if constexpr (i < s - 2)
              return get<i>(std::forward<Coll>(coll));
            else if constexpr (i == (s - 1) * n - 1)
              return get<s * n - 1>(std::forward<Coll>(coll));
            else if constexpr (i % (s - 1) == s - 1)
              return descriptor::Dimensions<descriptor::dimension_size_of_v<c_first> + descriptor::dimension_size_of_v<c_last>>{};
            else
              return get<i % (s - 1)>(std::forward<Coll>(coll));
          };
          return value::internal::transform_collection(OpenKalman::internal::iota_tuple<0_uz, (s - 1) * (n - 1) + s>(), std::move(f));
        }
        else
        {
          auto f = [&t](auto fixed_i)
          {
            decltype(auto) coll = descriptor::internal::get_component_collection(t.my_coordinates);
            using Coll = decltype(coll);
            auto coll_size = internal::get_collection_size(coll);
            auto local_i = value::operation {std::modulus<>{}, fixed_i, coll_size};
            return value::internal::get_collection_element(std::forward<Coll>(coll), local_i);
          };
          return value::internal::transform_collection(OpenKalman::internal::iota_tuple<0_uz, s * n>(), std::move(f));
        }
      }
      else
      {
        auto s = descriptor::get_component_count(t.my_coordinates);
        auto n = t.my_factor;
        auto f = [&t](auto i)
        {
          using Common = descriptor::internal::Any<scalar_type>;
          decltype(auto) coll = descriptor::internal::get_component_collection(t.my_coordinates);
          using Coll = decltype(coll);
          auto coll_size = internal::get_collection_size(coll);
          auto local_i = value::operation {std::modulus<>{}, i, coll_size};
          return value::internal::get_collection_element<Common>(std::forward<Coll>(coll), local_i);
        };
        return value::internal::transform_collection(OpenKalman::internal::iota_range(0_uz, s * n),std::move(f));
      }
    }


#ifdef __cpp_concepts
    template<value::index N>
#else
    template<typename N, std::enable_if_t<value::index<N>, int> = 0>
#endif
    static constexpr auto
    component_start_indices(const T& t, N n)
    {
      auto size_local_t = descriptor::get_component_count(t.my_coordinates);
      auto size_i = descriptor::get_size(t.my_coordinates);
      auto size_e = descriptor::get_euclidean_size(t.my_coordinates);
      auto factor = value::operation {std::divides<>{}, n, size_local_t};
      auto local_t = value::operation {std::modulus<>{}, n, size_local_t};
      auto [local_i, local_e] = descriptor::internal::get_component_start_indices(t.my_coordinates, local_t);
      auto i = value::operation {std::multiplies<>{}, factor, size_i} + local_i;
      auto e = value::operation {std::multiplies<>{}, factor, size_e} + local_e;
      return std::tuple {i, e};
    }


#ifdef __cpp_concepts
    template<value::index N>
#else
    template<typename N, std::enable_if_t<value::index<N>, int> = 0>
#endif
    static constexpr auto
    euclidean_component_start_indices(const T& t, N n)
    {
      auto size_local_t = descriptor::get_component_count(t.my_coordinates);
      auto size_i = descriptor::get_size(t.my_coordinates);
      auto size_e = descriptor::get_euclidean_size(t.my_coordinates);
      auto factor = value::operation {std::divides<>{}, n, size_local_t};
      auto local_t = value::operation {std::modulus<>{}, n, size_local_t};
      auto [local_i, local_e] = descriptor::internal::get_component_start_indices(t.my_coordinates, local_t);
      auto i = value::operation {std::multiplies<>{}, factor, size_i} + local_i;
      auto e = value::operation {std::multiplies<>{}, factor, size_e} + local_e;
      return std::tuple {i, e};
    }


#ifdef __cpp_concepts
    template<value::index I>
#else
    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    static constexpr auto
#endif
    static constexpr auto
    index_table(const T& t, I i)
    {
      std::size_t n = descriptor::get_size(t.my_coordinates);
      return descriptor::internal::get_index_table(t.my_coordinates, i % n);
    }


#ifdef __cpp_concepts
    template<value::index I>
#else
    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    static constexpr auto
#endif
    static constexpr auto
    euclidean_index_table(const T& t, I i)
    {
      std::size_t n = descriptor::get_euclidean_size(t.my_coordinates);
      return descriptor::internal::get_euclidean_index_table(t.my_coordinates, i % n);
    }


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
      std::size_t n = descriptor::get_euclidean_size(t.my_coordinates);
      auto new_g = [&g, offset = euclidean_local_index / n](std::size_t i) { return g(offset + i); };
      return descriptor::to_euclidean_element(t.my_coordinates, new_g, euclidean_local_index % n);
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
      auto n = descriptor::get_size(t.my_coordinates);
      auto new_g = [&g, offset = local_index / n](std::size_t i) { return g(offset + i); };
      return descriptor::from_euclidean_element(t.my_coordinates, new_g, local_index % n);
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
      auto n = descriptor::get_size(t.my_coordinates);
      auto new_g = [&g, offset = local_index / n](std::size_t i) { return g(offset + i); };
      return descriptor::get_wrapped_component(t.my_coordinates, new_g, local_index % n);
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
      auto n = descriptor::get_size(t.my_coordinates);
      auto new_g = [&g, offset = local_index / n](std::size_t i) { return g(offset + i); };
      auto new_s = [&s, offset = local_index / n](const Scalar& x, std::size_t i) { s(x, offset + i); };
      return descriptor::set_wrapped_component(t.my_coordinates, new_s, new_g, x, local_index % n);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_DESCRIPTOR_REPLICATE_HPP
